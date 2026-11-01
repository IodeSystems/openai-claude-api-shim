package shim

import (
	"encoding/json"
	"strings"
	"testing"
)

func reversed(t *testing.T, body string) map[string]any {
	t.Helper()
	b, err := ReverseRequest([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestReverseRequest(t *testing.T) {
	out := reversed(t, `{"model":"m","max_tokens":9,"stream":true,
		"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=1; cch=2;\nBe brief."}],
		"tools":[{"name":"exec","description":"d","input_schema":{"type":"object"}},{"type":"web_search_20260209","name":"web_search"}],
		"tool_choice":{"type":"any","disable_parallel_tool_use":true},
		"messages":[
		 {"role":"user","content":"hi"},
		 {"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"exec","input":{"a":1}}]},
		 {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":"boom"},{"type":"text","text":"next"}]},
		 {"role":"system","content":"env"}]}`)
	msgs := out["messages"].([]any)
	role := func(i int) map[string]any { return msgs[i].(map[string]any) }
	if role(0)["role"] != "system" || role(0)["content"] != "Be brief." {
		t.Errorf("system %v", role(0))
	}
	if calls := role(2)["tool_calls"].([]any); len(calls) != 1 || role(2)["content"] != "ok" {
		t.Errorf("assistant %v", role(2))
	}
	if role(3)["role"] != "tool" || role(3)["tool_call_id"] != "t1" || role(3)["content"] != "Error: boom" {
		t.Errorf("result %v", role(3))
	}
	if role(4)["role"] != "user" || role(4)["content"] != "next" {
		t.Errorf("turn %v", role(4))
	}
	if role(5)["role"] != "user" || role(5)["content"] != "env" {
		t.Errorf("a late system message is a user message: %v", role(5))
	}
	if tools := out["tools"].([]any); len(tools) != 1 {
		t.Errorf("a server tool was sent: %v", tools)
	}
	if out["tool_choice"] != "required" || out["parallel_tool_calls"] != false || out["stream"] != true {
		t.Errorf("%v", out)
	}
}

func TestReverseResponse(t *testing.T) {
	b, err := ReverseResponse([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"reasoning_content":"r","content":"hi","tool_calls":[{"id":"a.b","function":{"name":"exec","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}`), "m", "corrallm")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"stop_reason":"tool_use"`, `"signature":"corrallm"`, `"id":"a_b"`, `"input_tokens":6`, `"cache_read_input_tokens":4`, `"id":"msg_1"`} {
		if !strings.Contains(s, want) {
			t.Errorf("no %s in %s", want, s)
		}
	}
}

func TestReverseStream(t *testing.T) {
	var out strings.Builder
	s := NewReverseStream(func(b []byte) { out.Write(b) }, "m", "corrallm")
	in := `data: {"id":"chatcmpl-1","choices":[{"delta":{"reasoning_content":"r"}}]}

data: {"choices":[{"delta":{"content":"hi"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t","function":{"name":"exec","arguments":"{\"a\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}

data: [DONE]

`
	for i := range len(in) { // split anywhere
		s.Feed([]byte{in[i]})
	}
	s.End()
	got := out.String()
	var order []string
	for _, line := range strings.Split(got, "\n") {
		if e, ok := strings.CutPrefix(line, "event: "); ok {
			order = append(order, e)
		}
	}
	want := "message_start content_block_start content_block_delta content_block_delta content_block_stop " +
		"content_block_start content_block_delta content_block_stop content_block_start content_block_delta " +
		"content_block_delta content_block_stop message_delta message_stop"
	if strings.Join(order, " ") != want {
		t.Errorf("events:\n%s", strings.Join(order, " "))
	}
	if !strings.Contains(got, `"stop_reason":"tool_use"`) || !strings.Contains(got, `"signature":"corrallm"`) {
		t.Errorf("stream:\n%s", got)
	}
}

func TestStripThinking(t *testing.T) {
	body := []byte(`{"thinking":{"type":"adaptive"},"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"corrallm"},{"type":"text","text":"a"}]}]}`)
	out := string(StripThinking(body, "corrallm"))
	if strings.Contains(out, "corrallm") || strings.Contains(out, `"thinking":{`) {
		t.Errorf("%s", out)
	}
	clean := []byte(`{"messages":[]}`)
	if string(StripThinking(clean, "corrallm")) != string(clean) {
		t.Error("a body with nothing to strip changed")
	}
}

func TestEstimate(t *testing.T) {
	if v, ok := Estimate(nil, "claude-opus-5-5", Usage{Input: 1_000_000, CacheWrite: 1_000_000, CacheWrite1h: 1_000_000}); !ok || !near(v, 12) {
		t.Errorf("%v %v", v, ok)
	}
	if _, ok := Estimate(nil, "gpt-5", Usage{Input: 1}); ok {
		t.Error("a model with no price was estimated")
	}
	if v, ok := Estimate(Prices{"acme": {Input: 2}}, "acme-1", Usage{Input: 1_000_000}); !ok || !near(v, 2) {
		t.Errorf("host price: %v %v", v, ok)
	}
}

func TestReverse_NotAChatCompletion(t *testing.T) {
	for _, body := range []string{`nope`, `{"error":"x"}`, `{}`} {
		if _, err := ReverseResponse([]byte(body), "m", "sig"); err == nil {
			t.Errorf("%s was translated", body)
		}
	}
	var out strings.Builder
	s := NewReverseStream(func(b []byte) { out.Write(b) }, "m", "sig")
	s.Feed([]byte("<html>bad gateway</html>\n"))
	if err := s.End(); err == nil || out.Len() != 0 {
		t.Errorf("err %v, sent %q", err, out.String())
	}
}

func TestReverse_TheCallersSignature(t *testing.T) {
	var out strings.Builder
	s := NewReverseStream(func(b []byte) { out.Write(b) }, "m", "dun-sig")
	s.Feed([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"r\"}}]}\n\n"))
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"signature":"dun-sig"`) {
		t.Errorf("%s", out.String())
	}
}

func TestReverseError(t *testing.T) {
	var e struct {
		Type  string
		Error struct{ Type, Message string }
	}
	if err := json.Unmarshal(ReverseError(429, "slow"), &e); err != nil || e.Type != "error" || e.Error.Type != ErrorType(429) || e.Error.Message != "slow" {
		t.Errorf("%+v %v", e, err)
	}
}

func TestEstimateTokens(t *testing.T) {
	if n := EstimateTokens([]byte(`{"system":"12345678","messages":[{"role":"user","content":"hi"}]}`)); n < 3 {
		t.Errorf("%d", n)
	}
}
