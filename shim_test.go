package shim

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

// translate runs Request and decodes the result for inspection.
func translate(t *testing.T, body string, opts Options) antRequest {
	t.Helper()
	out, err := Request([]byte(body), opts)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var req antRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatalf("translated body is not JSON: %v\n%s", err, out)
	}
	return req
}

func TestAnthropicRequestBasics(t *testing.T) {
	req := translate(t, `{
		"model":"claude-x","stream":true,"temperature":0.2,"top_p":0.9,
		"stop":["END","  \n"],"n":2,"presence_penalty":1,"response_format":{"type":"json_object"},
		"stream_options":{"include_usage":true},"min_p":0.05,
		"messages":[
			{"role":"system","content":"be terse"},
			{"role":"developer","content":[{"type":"text","text":"and kind"}]},
			{"role":"user","content":"hi"}
		]}`, Options{})

	if req.Model != "claude-x" || !req.Stream {
		t.Errorf("model/stream = %q/%v", req.Model, req.Stream)
	}
	if req.MaxTokens != defaultStreamMaxTokens {
		t.Errorf("max_tokens = %d, want the default %d when nothing is set", req.MaxTokens, defaultStreamMaxTokens)
	}
	if len(req.StopSequences) != 1 || req.StopSequences[0] != "END" {
		t.Errorf("stop_sequences = %q, want [END] (whitespace-only dropped)", req.StopSequences)
	}
	if len(req.System) != 2 || req.System[0].Text != "be terse" || req.System[1].Text != "and kind" {
		t.Errorf("system = %+v", req.System)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" || req.Messages[0].Content[0].Text != "hi" {
		t.Errorf("messages = %+v", req.Messages)
	}

	// Dropped fields must not reach Anthropic at all.
	raw, _ := Request([]byte(`{"model":"m","n":2,"seed":1,"user":"u","logit_bias":{},
		"messages":[{"role":"user","content":"x"}]}`), Options{})
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	for _, k := range []string{"n", "seed", "user", "logit_bias", "stream_options", "response_format", "min_p"} {
		if _, ok := m[k]; ok {
			t.Errorf("field %q leaked into the Anthropic body", k)
		}
	}
}

func TestAnthropicRequestMaxTokens(t *testing.T) {
	cases := []struct {
		name string
		body string
		opts Options
		want int
	}{
		{"default constant", `{}`, Options{}, defaultMaxTokens},
		{"model cap fallback", `{}`, Options{MaxTokens: 777}, 777},
		{"max_tokens", `{"max_tokens":100}`, Options{MaxTokens: 777}, 100},
		{"max_completion_tokens wins", `{"max_tokens":100,"max_completion_tokens":200}`, Options{}, 200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := strings.Replace(c.body, "{", `{"model":"m","messages":[{"role":"user","content":"x"}],`, 1)
			body = strings.Replace(body, ",}", "}", 1)
			if got := translate(t, body, c.opts).MaxTokens; got != c.want {
				t.Errorf("max_tokens = %d, want %d", got, c.want)
			}
		})
	}
}

// A full tool round trip: the assistant's tool_calls become tool_use blocks,
// the tool replies become ONE user turn of tool_result blocks, and the user's
// follow-up merges into that same turn after them.
func TestAnthropicRequestToolRoundTrip(t *testing.T) {
	req := translate(t, `{"model":"m","messages":[
		{"role":"user","content":"weather in two cities"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"wx","arguments":"{\"city\":\"Oslo\"}"}},
			{"id":"call_2","type":"function","function":{"name":"wx","arguments":"{\"city\":"}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":"cold"},
		{"role":"tool","tool_call_id":"call_2","content":[{"type":"text","text":"bad args"}]},
		{"role":"user","content":"thanks"}
	],"tools":[{"type":"function","function":{"name":"wx","description":"weather",
		"parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
		{"type":"function","function":{"name":"noargs"}}]}`, Options{})

	if len(req.Messages) != 3 {
		t.Fatalf("want user/assistant/user, got %d messages: %+v", len(req.Messages), req.Messages)
	}
	a := req.Messages[1]
	if a.Role != "assistant" || len(a.Content) != 2 {
		t.Fatalf("assistant turn = %+v", a)
	}
	if a.Content[0].Type != "tool_use" || a.Content[0].ID != "call_1" || a.Content[0].Name != "wx" ||
		string(a.Content[0].Input) != `{"city":"Oslo"}` {
		t.Errorf("tool_use 1 = %+v", a.Content[0])
	}
	if string(a.Content[1].Input) != `{}` {
		t.Errorf("truncated arguments must replay as {}, got %s", a.Content[1].Input)
	}
	u := req.Messages[2]
	if u.Role != "user" || len(u.Content) != 3 {
		t.Fatalf("merged user turn = %+v", u)
	}
	if u.Content[0].Type != "tool_result" || u.Content[0].ToolUseID != "call_1" || u.Content[0].Content[0].Text != "cold" ||
		u.Content[1].Type != "tool_result" || u.Content[1].ToolUseID != "call_2" ||
		u.Content[2].Type != "text" || u.Content[2].Text != "thanks" {
		t.Errorf("merged user turn content = %+v", u.Content)
	}
	if len(req.Tools) != 2 || req.Tools[0].Name != "wx" || req.Tools[0].Description != "weather" ||
		!strings.Contains(string(req.Tools[0].InputSchema), `"city"`) {
		t.Errorf("tools = %+v", req.Tools)
	}
	if string(req.Tools[1].InputSchema) != `{"type":"object","properties":{}}` {
		t.Errorf("a tool with no parameters needs an empty object schema, got %s", req.Tools[1].InputSchema)
	}
}

func TestAnthropicRequestRoleMerging(t *testing.T) {
	req := translate(t, `{"model":"m","messages":[
		{"role":"user","content":"a"},
		{"role":"user","content":[{"type":"text","text":"b"},{"type":"text","text":""}]},
		{"role":"assistant","content":""},
		{"role":"user","content":"c"},
		{"role":"assistant","content":"d"},
		{"role":"assistant","content":[{"type":"text","text":"e"}]},
		{"role":"tool","tool_call_id":"x","content":"r"},
		{"role":"user","content":"f"}
	]}`, Options{})
	var got []string
	for _, m := range req.Messages {
		var parts []string
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				parts = append(parts, "result")
			} else {
				parts = append(parts, b.Text)
			}
		}
		got = append(got, m.Role+":"+strings.Join(parts, ","))
	}
	want := []string{"user:a,b,c", "assistant:d,e", "user:r,f"}
	if strings.Join(got, " | ") != strings.Join(want, " | ") {
		t.Errorf("merged = %q, want %q (empty assistant dropped, empty text skipped)", got, want)
	}
}

func TestAnthropicRequestImages(t *testing.T) {
	req := translate(t, `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"what is this"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},
		{"type":"image_url","image_url":{"url":"https://example.com/cat.jpg","detail":"high"}}
	]}]}`, Options{})
	c := req.Messages[0].Content
	if len(c) != 3 {
		t.Fatalf("content = %+v", c)
	}
	if s := c[1].Source; c[1].Type != "image" || s == nil || s.Type != "base64" || s.MediaType != "image/png" || s.Data != "iVBORw0KGgo=" {
		t.Errorf("data URL image = %+v %+v", c[1], c[1].Source)
	}
	if s := c[2].Source; s == nil || s.Type != "url" || s.URL != "https://example.com/cat.jpg" {
		t.Errorf("https image = %+v", c[2].Source)
	}
}

func TestAnthropicRequestRefusals(t *testing.T) {
	cases := map[string]string{
		"audio part":        `{"model":"m","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]}`,
		"unknown role":      `{"model":"m","messages":[{"role":"function","content":"x"}]}`,
		"non-base64 data":   `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png,xyz"}}]}]}`,
		"ftp image":         `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"ftp://x/y.png"}}]}]}`,
		"image in system":   `{"model":"m","messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}]}`,
		"bad tool_choice":   `{"model":"m","tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"sometimes","messages":[]}`,
		"not json":          `nope`,
		"non-function tool": `{"model":"m","tools":[{"type":"web_search"}],"messages":[]}`,
	}
	for name, body := range cases {
		if _, err := Request([]byte(body), Options{}); err == nil {
			t.Errorf("%s: want an error, got none", name)
		}
	}
}

func TestAnthropicToolChoice(t *testing.T) {
	tools := `"tools":[{"type":"function","function":{"name":"f"}}],`
	cases := []struct {
		name  string
		extra string
		tools bool
		want  string // JSON of tool_choice, "" for absent
	}{
		{"absent", ``, true, ``},
		{"auto", `"tool_choice":"auto",`, true, `{"type":"auto"}`},
		{"none", `"tool_choice":"none",`, true, `{"type":"none"}`},
		{"required", `"tool_choice":"required",`, true, `{"type":"any"}`},
		{"named", `"tool_choice":{"type":"function","function":{"name":"f"}},`, true, `{"type":"tool","name":"f"}`},
		{"parallel off alone", `"parallel_tool_calls":false,`, true, `{"type":"auto","disable_parallel_tool_use":true}`},
		{"parallel off required", `"tool_choice":"required","parallel_tool_calls":false,`, true, `{"type":"any","disable_parallel_tool_use":true}`},
		{"parallel off none", `"tool_choice":"none","parallel_tool_calls":false,`, true, `{"type":"none"}`},
		{"no tools drops it", `"tool_choice":"required",`, false, ``},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := `{"model":"m",` + c.extra
			if c.tools {
				body += tools
			}
			body += `"messages":[{"role":"user","content":"x"}]}`
			req := translate(t, body, Options{})
			got := ""
			if req.ToolChoice != nil {
				b, _ := json.Marshal(req.ToolChoice)
				got = string(b)
			}
			if got != c.want {
				t.Errorf("tool_choice = %s, want %s", got, c.want)
			}
		})
	}
}

// markers lists where cache_control landed, as "tools[i]", "system[i]" and
// "msg[i].content[j]".
func markers(req antRequest) []string {
	var out []string
	for i, t := range req.Tools {
		if len(t.CacheControl) > 0 {
			out = append(out, fmt.Sprintf("tools[%d]", i))
		}
	}
	for i, b := range req.System {
		if len(b.CacheControl) > 0 {
			out = append(out, fmt.Sprintf("system[%d]", i))
		}
	}
	for i, m := range req.Messages {
		for j, b := range m.Content {
			if len(b.CacheControl) > 0 {
				out = append(out, fmt.Sprintf("msg[%d].content[%d]", i, j))
			}
		}
	}
	return out
}

func TestAnthropicCacheBreakpoints(t *testing.T) {
	conv := `"messages":[
		{"role":"system","content":"sys one"},
		{"role":"system","content":"sys two"},
		{"role":"user","content":"u1"},
		{"role":"assistant","content":"a1"},
		{"role":"user","content":"u2"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"c","content":"out"},
		{"role":"user","content":[{"type":"text","text":"u3a"},{"type":"text","text":"u3b"}]}
	]`
	tools := `"tools":[{"type":"function","function":{"name":"e"}},{"type":"function","function":{"name":"f"}}]`

	cases := []struct {
		name string
		body string
		ttl  string
		want []string
	}{
		{
			name: "tools, system, last two user turns",
			body: `{"model":"m",` + tools + `,` + conv + `}`,
			want: []string{"tools[1]", "system[1]", "msg[2].content[0]", "msg[4].content[2]"},
		},
		{
			name: "no tools: system and two turns",
			body: `{"model":"m",` + conv + `}`,
			want: []string{"system[1]", "msg[2].content[0]", "msg[4].content[2]"},
		},
		{
			name: "single user turn",
			body: `{"model":"m","messages":[{"role":"user","content":"only"}]}`,
			want: []string{"msg[0].content[0]"},
		},
		{
			// The caller already spent three of the four: only the
			// highest-priority automatic slot (the tool list) still fits.
			name: "caller markers count against the cap",
			body: `{"model":"m",` + tools + `,"messages":[
				{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]},
				{"role":"assistant","content":"x"},
				{"role":"user","content":[{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}]},
				{"role":"assistant","content":"y"},
				{"role":"user","content":[{"type":"text","text":"c","cache_control":{"type":"ephemeral"}}]}
			]}`,
			want: []string{"tools[1]", "msg[0].content[0]", "msg[2].content[0]", "msg[4].content[0]"},
		},
		{
			name: "caller already at the cap: nothing added",
			body: `{"model":"m",` + tools + `,"messages":[
				{"role":"system","content":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}]},
				{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]},
				{"role":"assistant","content":"x"},
				{"role":"user","content":[{"type":"text","text":"b","cache_control":{"type":"ephemeral"}},{"type":"text","text":"b2","cache_control":{"type":"ephemeral"}}]}
			]}`,
			want: []string{"system[0]", "msg[0].content[0]", "msg[2].content[0]", "msg[2].content[1]"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := translate(t, c.body, Options{CacheTTL: c.ttl})
			got := markers(req)
			if strings.Join(got, " ") != strings.Join(c.want, " ") {
				t.Errorf("breakpoints = %v, want %v", got, c.want)
			}
			if n := countBreakpoints(&req); n > maxBreakpoints {
				t.Errorf("%d breakpoints, cap is %d", n, maxBreakpoints)
			}
		})
	}
}

func TestAnthropicCacheTTL(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"x"}]}`
	if cc := string(translate(t, body, Options{}).Messages[0].Content[0].CacheControl); cc != `{"type":"ephemeral"}` {
		t.Errorf("default TTL marker = %s", cc)
	}
	if cc := string(translate(t, body, Options{CacheTTL: "1h"}).Messages[0].Content[0].CacheControl); cc != `{"type":"ephemeral","ttl":"1h"}` {
		t.Errorf("1h TTL marker = %s", cc)
	}
}

func TestAnthropicResponseTranslation(t *testing.T) {
	out, err := Response([]byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-x",
		"content":[
			{"type":"thinking","thinking":"hmm","signature":"sig"},
			{"type":"text","text":"Checking "},
			{"type":"text","text":"now."},
			{"type":"tool_use","id":"toolu_1","name":"wx","input":{"city":"Oslo"}}
		],
		"stop_reason":"tool_use",
		"usage":{"input_tokens":10,"cache_creation_input_tokens":200,"cache_read_input_tokens":3000,"output_tokens":42}}`), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var c oaiCompletion
	if err := json.Unmarshal(out, &c); err != nil {
		t.Fatal(err)
	}
	if c.Object != "chat.completion" || c.ID != "msg_1" || c.Model != "claude-x" || len(c.Choices) != 1 {
		t.Fatalf("envelope = %s", out)
	}
	m := c.Choices[0].Message
	if m.Content == nil || *m.Content != "Checking now." || m.ReasoningContent != "hmm" {
		t.Errorf("content/reasoning = %v/%q", m.Content, m.ReasoningContent)
	}
	if len(m.ToolCalls) != 1 || m.ToolCalls[0].ID != "toolu_1" || m.ToolCalls[0].Type != "function" ||
		m.ToolCalls[0].Function.Name != "wx" || m.ToolCalls[0].Function.Arguments != `{"city":"Oslo"}` {
		t.Errorf("tool_calls = %+v", m.ToolCalls)
	}
	if c.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q", c.Choices[0].FinishReason)
	}
	if c.Usage.PromptTokens != 3210 || c.Usage.CompletionTokens != 42 || c.Usage.TotalTokens != 3252 ||
		c.Usage.PromptTokensDetails.CachedTokens != 3000 {
		t.Errorf("usage = %+v", c.Usage)
	}

	// Tool call with no text: content is null, as OpenAI sends it.
	out, _ = Response([]byte(`{"id":"m","content":[{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use"}`), StreamOptions{})
	if !strings.Contains(string(out), `"content":null`) {
		t.Errorf("tool-only reply should carry content:null: %s", out)
	}
}

func TestAnthropicFinishReason(t *testing.T) {
	for in, want := range map[string]string{
		"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
		"tool_use": "tool_calls", "refusal": "content_filter", "pause_turn": "stop", "": "stop",
	} {
		if got := finishOf(in); got != want {
			t.Errorf("finishOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAnthropicErrorTranslation(t *testing.T) {
	out := Error([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	var e struct {
		Error struct{ Message, Type string } `json:"error"`
	}
	if json.Unmarshal(out, &e) != nil || e.Error.Message != "slow down" || e.Error.Type != "rate_limit_error" {
		t.Errorf("error = %s", out)
	}
	if got := string(Error([]byte("upstream exploded"))); got != "upstream exploded" {
		t.Errorf("a non-Anthropic body must pass through, got %q", got)
	}
}

// anthropicSSE is a realistic stream: thinking, text, a tool call whose
// arguments arrive in pieces, a ping, and usage split across message_start
// and message_delta.
const anthropicSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"usage":{"input_tokens":12,"cache_creation_input_tokens":100,"cache_read_input_tokens":5000,"output_tokens":1}}}` + "\n\n" +
	"event: ping\ndata: {\"type\":\"ping\"}\n\n" +
	`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me see"}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}` + "\n\n" +
	`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n" +
	`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world"}}` + "\n\n" +
	`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":1}` + "\n\n" +
	`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"wx","input":{}}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":""}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}` + "\n\n" +
	`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Oslo\"}"}}` + "\n\n" +
	`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":2}` + "\n\n" +
	`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":57}}` + "\n\n" +
	`event: message_stop` + "\n" + `data: {"type":"message_stop"}` + "\n\n"

// parsedStream is an OpenAI stream reassembled the way a client would.
type parsedStream struct {
	role, content, reasoning string
	toolID, toolName, args   string
	finish                   string
	usage                    *oaiUsage
	done                     bool
	errBody                  string
}

func parseOpenAIStream(t *testing.T, b []byte) parsedStream {
	t.Helper()
	var p parsedStream
	for _, ev := range strings.Split(string(b), "\n\n") {
		ev = strings.TrimSpace(ev)
		if ev == "" {
			continue
		}
		data, ok := strings.CutPrefix(ev, "data: ")
		if !ok {
			t.Fatalf("event without data: %q", ev)
		}
		if p.done {
			t.Fatalf("event after [DONE]: %q", ev)
		}
		if data == "[DONE]" {
			p.done = true
			continue
		}
		if strings.HasPrefix(data, `{"error"`) {
			p.errBody = data
			continue
		}
		var c oaiChunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("chunk not JSON: %v: %s", err, data)
		}
		if c.Object != "chat.completion.chunk" {
			t.Errorf("object = %q", c.Object)
		}
		if c.Usage != nil {
			p.usage = c.Usage
		}
		for _, ch := range c.Choices {
			if ch.Delta.Role != "" {
				p.role = ch.Delta.Role
			}
			if ch.Delta.Content != nil {
				p.content += *ch.Delta.Content
			}
			p.reasoning += ch.Delta.ReasoningContent
			for _, tc := range ch.Delta.ToolCalls {
				if tc.Index == nil || *tc.Index != 0 {
					t.Errorf("tool call delta index = %v, want 0", tc.Index)
				}
				if tc.ID != "" {
					p.toolID, p.toolName = tc.ID, tc.Function.Name
				}
				p.args += tc.Function.Arguments
			}
			if ch.FinishReason != nil {
				p.finish = *ch.FinishReason
			}
		}
	}
	return p
}

func TestAnthropicStreamTranslation(t *testing.T) {
	// Fed one byte at a time and all at once: event boundaries must not
	// depend on how the transport chunked the body.
	for _, step := range []int{1, 7, len(anthropicSSE)} {
		var out bytes.Buffer
		s := Stream{emit: func(b []byte) { out.Write(b) }}
		for i := 0; i < len(anthropicSSE); i += step {
			s.Feed([]byte(anthropicSSE[i:min(i+step, len(anthropicSSE))]))
		}
		s.Finish()
		p := parseOpenAIStream(t, out.Bytes())
		if p.role != "assistant" || p.content != "Hello world" || p.reasoning != "let me see" {
			t.Errorf("step %d: role/content/reasoning = %q/%q/%q", step, p.role, p.content, p.reasoning)
		}
		if p.toolID != "toolu_9" || p.toolName != "wx" || p.args != `{"city":"Oslo"}` {
			t.Errorf("step %d: tool = %q %q %q", step, p.toolID, p.toolName, p.args)
		}
		if p.finish != "tool_calls" || !p.done {
			t.Errorf("step %d: finish=%q done=%v", step, p.finish, p.done)
		}
		if p.usage == nil || p.usage.PromptTokens != 5112 || p.usage.CompletionTokens != 57 ||
			p.usage.PromptTokensDetails.CachedTokens != 5000 {
			t.Errorf("step %d: usage = %+v", step, p.usage)
		}
	}
}

func TestAnthropicStreamError(t *testing.T) {
	var out bytes.Buffer
	s := Stream{emit: func(b []byte) { out.Write(b) }}
	s.Feed([]byte(strings.SplitAfter(anthropicSSE, "\n\n")[0]))
	s.Feed([]byte("event: error\r\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\r\n\r\n"))
	s.Finish()
	p := parseOpenAIStream(t, out.Bytes())
	if !strings.Contains(p.errBody, `"message":"Overloaded"`) || !strings.Contains(p.errBody, `"type":"overloaded_error"`) || !p.done {
		t.Errorf("error = %q done=%v\n%s", p.errBody, p.done, out.String())
	}
}

func TestAnthropicStreamTruncated(t *testing.T) {
	// Upstream drops after the first text delta: usage so far is reported,
	// but no finish_reason and no [DONE], so the cut-off stays visible.
	cut := strings.Index(anthropicSSE, `"text":" world"`)
	var out bytes.Buffer
	s := Stream{emit: func(b []byte) { out.Write(b) }}
	s.Feed([]byte(anthropicSSE[:cut]))
	s.Finish()
	p := parseOpenAIStream(t, out.Bytes())
	if p.done || p.finish != "" || p.usage == nil || p.usage.PromptTokensDetails.CachedTokens != 5000 {
		t.Errorf("truncated stream: done=%v finish=%q usage=%+v", p.done, p.finish, p.usage)
	}
}

// --- full proxy path against a fake Anthropic ---

func TestEffort(t *testing.T) {
	for _, tc := range []struct{ model, level, want string }{
		{"claude-opus-5-5", "high", "high"},
		{"claude-opus-5-5", "none", "low"},
		{"claude-opus-5-5", "", ""},
		{"claude-haiku-4-5-20251001", "high", ""},
		{"claude-opus-5-5", "sometimes", ""},
	} {
		req := translate(t, `{"model":"`+tc.model+`","reasoning_effort":"`+tc.level+`","messages":[{"role":"user","content":"hi"}]}`, Options{})
		got := ""
		if req.OutputConfig != nil {
			got = req.OutputConfig.Effort
		}
		if got != tc.want {
			t.Errorf("%s %q: %q, want %q", tc.model, tc.level, got, tc.want)
		}
	}
}

func TestSamplingIsNotSent(t *testing.T) {
	b, err := Request([]byte(`{"model":"claude-opus-5-5","temperature":0.2,"top_p":0.9,"top_k":3,"messages":[{"role":"user","content":"hi"}]}`), Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"temperature", "top_p", "top_k"} {
		if strings.Contains(string(b), f) {
			t.Errorf("%s was sent: %s", f, b)
		}
	}
}

func TestMaxTokensDefaults(t *testing.T) {
	if r := translate(t, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, Options{}); r.MaxTokens != defaultStreamMaxTokens {
		t.Errorf("stream: %d", r.MaxTokens)
	}
	if r := translate(t, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, Options{}); r.MaxTokens != defaultMaxTokens {
		t.Errorf("plain: %d", r.MaxTokens)
	}
}

const oneTool = `"tools":[{"type":"function","function":{"name":"a","parameters":{"type":"object"}}},{"type":"function","function":{"name":"b","parameters":{"type":"object"}}}]`

func TestForcedChoice(t *testing.T) {
	msgs := `"messages":[{"role":"user","content":"hi"}]`
	named := `{"type":"function","function":{"name":"b"}}`
	for _, tc := range []struct {
		model, choice, want, name string
		tools                     int
	}{
		{"claude-opus-5", `"required"`, "any", "", 2},
		{"claude-opus-5", named, "tool", "b", 2},
		{"claude-opus-5-5", `"required"`, "auto", "", 2},
		{"claude-opus-5-5", named, "auto", "", 1},
		{"claude-sonnet-5-5", named, "auto", "", 1},
		{"claude-fable-5-1", named, "auto", "", 1},
		{"claude-mythos-5-1", named, "auto", "", 1},
		{"claude-opus-5-5", `{"type":"function","function":{"name":"gone"}}`, "auto", "", 2},
		{"claude-opus-5-5", `"none"`, "none", "", 2},
		{"claude-opus-5-5", `"auto"`, "auto", "", 2},
	} {
		r := translate(t, `{"model":"`+tc.model+`",`+msgs+`,`+oneTool+`,"tool_choice":`+tc.choice+`}`, Options{})
		if r.ToolChoice == nil || r.ToolChoice.Type != tc.want || r.ToolChoice.Name != tc.name || len(r.Tools) != tc.tools {
			t.Errorf("%s %s: %+v, %d tools", tc.model, tc.choice, r.ToolChoice, len(r.Tools))
		}
	}
}

func roles(r antRequest) string {
	var out []string
	for _, m := range r.Messages {
		out = append(out, m.Role[:1])
	}
	return strings.Join(out, "")
}

const toolCall = `{"role":"assistant","content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"a","arguments":"{}"}}]}`

// A transcript that was folded or interrupted still makes a valid request.
func TestRepair(t *testing.T) {
	for name, tc := range map[string]struct{ msgs, want string }{
		"starts with assistant": {`{"role":"assistant","content":"a"},{"role":"user","content":"b"}`, "uau"},
		"ends with assistant":   {`{"role":"user","content":"a"},{"role":"assistant","content":"b"}`, "uau"},
		"result never came":     {`{"role":"user","content":"a"},` + toolCall + `,{"role":"user","content":"b"}`, "uau"},
		"ends on a call":        {`{"role":"user","content":"a"},` + toolCall, "uau"},
		"call folded away":      {`{"role":"tool","tool_call_id":"gone","content":"x"},{"role":"user","content":"b"}`, "u"},
		"empty reply":           {`{"role":"user","content":"a"},{"role":"assistant","content":""},{"role":"user","content":"b"}`, "u"},
	} {
		if got := roles(translate(t, `{"model":"m","messages":[`+tc.msgs+`]}`, Options{})); got != tc.want {
			t.Errorf("%s: %s, want %s", name, got, tc.want)
		}
	}
	r := translate(t, `{"model":"m","messages":[{"role":"user","content":"a"},`+toolCall+`,{"role":"user","content":"b"}]}`, Options{})
	got := r.Messages[2].Content
	if got[0].Type != "tool_result" || got[0].ToolUseID != "1" || !got[0].IsError || got[1].Text != "b" {
		t.Errorf("missing result %+v", got)
	}
	r = translate(t, `{"model":"m","messages":[{"role":"tool","tool_call_id":"gone","content":"kept"},{"role":"user","content":"b"}]}`, Options{})
	if got := r.Messages[0].Content; len(got) != 2 || got[0].Type != "text" || got[0].Text != "kept" {
		t.Errorf("orphan %+v", got)
	}
}

func TestEagerInputStreamingOnlyOnAStream(t *testing.T) {
	r := translate(t, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],`+oneTool+`}`, Options{})
	if !r.Tools[0].EagerInputStreaming {
		t.Error("a stream did not ask for it")
	}
	r = translate(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],`+oneTool+`}`, Options{})
	if r.Tools[0].EagerInputStreaming {
		t.Error("a plain request asked for it")
	}
}

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("event: x\ndata: " + e + "\n\n")
	}
	return b.String()
}

func transcoded(t *testing.T, in string, opts StreamOptions) string {
	t.Helper()
	b, err := io.ReadAll(Transcode(strings.NewReader(in), opts))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestTranscode_NoArguments(t *testing.T) {
	out := transcoded(t, sse(`{"type":"message_start","message":{"id":"m","model":"x","usage":{"input_tokens":1,"cache_read_input_tokens":90,"cache_creation_input_tokens":9}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"a"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`), StreamOptions{})
	for _, want := range []string{`"arguments":"{}"`, `"finish_reason":"tool_calls"`, `"prompt_tokens":100`, `"cache_read_input_tokens":90`, `"cache_creation_input_tokens":9`, "data: [DONE]"} {
		if !strings.Contains(out, want) {
			t.Errorf("no %s in:\n%s", want, out)
		}
	}
}

func TestTranscode_Refusal(t *testing.T) {
	in := sse(`{"type":"message_start","message":{"id":"m","model":"x","usage":{}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"cyber","explanation":"no"}},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`)
	if out := transcoded(t, in, StreamOptions{}); !strings.Contains(out, `"finish_reason":"content_filter"`) {
		t.Errorf("default:\n%s", out)
	}
	out := transcoded(t, in, StreamOptions{RefusalAsError: true})
	if !strings.Contains(out, `"error"`) || !strings.Contains(out, "cyber") || strings.Contains(out, "finish_reason\":\"content_filter") {
		t.Errorf("as error:\n%s", out)
	}
}
