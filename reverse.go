package shim

// reverse.go — the other direction: a caller that speaks only the Messages API,
// Claude Code above all, in front of an OpenAI backend. ReverseRequest translates
// its request; ReverseResponse and ReverseStream translate the reply back.
//
// Moved from corrallm's internal/proxy/messages.go. What is around it stays there:
// choosing a backend, credentials, and error restating.
//
// Translated requests lose nothing a local model can use. They drop cache_control
// (llama.cpp caches the prefix by itself), thinking settings, metadata, server
// tools, and Claude Code's per-request billing line in the system prompt, which
// changes on every request and would stop a local backend reusing its prompt cache.
//
// Thinking a translated model streams back is sent as thinking blocks signed with
// the host's signature. Anthropic would refuse those, so a host that sends the
// conversation on to Anthropic runs it through StripThinking first.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
)

// A signature marks a thinking block made from a non-Anthropic model's reasoning.
// It is not a valid Anthropic signature, on purpose. The host supplies it, because
// a live conversation holds blocks signed with it: a host that changes its value
// stops StripThinking matching them, and Anthropic refuses the request.

// NewReverseStream returns a ReverseStream that writes Messages SSE through emit
// as bytes arrive, nothing held to the end.
func NewReverseStream(emit func([]byte), model, signature string) *ReverseStream {
	return &ReverseStream{emit: emit, model: model, signature: signature}
}

// --- request: Anthropic → OpenAI ---

type antInBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Source    *antInSource    `json:"source"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
}

type antInSource struct {
	Type      string `json:"type"` // base64 | url | text
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type antInRequest struct {
	Model      string            `json:"model"`
	MaxTokens  *int              `json:"max_tokens"`
	System     json.RawMessage   `json:"system"`
	Messages   []json.RawMessage `json:"messages"`
	Tools      []json.RawMessage `json:"tools"`
	ToolChoice *struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
	} `json:"tool_choice"`
	StopSequences []string `json:"stop_sequences"`
	Stream        bool     `json:"stream"`
	Temperature   *float64 `json:"temperature"`
	TopP          *float64 `json:"top_p"`
	TopK          *int     `json:"top_k"`
}

// blocksOf reads content that is a string or a list of blocks.
func blocksOf(raw json.RawMessage) ([]antInBlock, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []antInBlock{{Type: "text", Text: s}}, nil
	}
	var bs []antInBlock
	if err := json.Unmarshal(raw, &bs); err != nil {
		return nil, fmt.Errorf("content is neither a string nor a list of blocks")
	}
	return bs, nil
}

// billingLine is Claude Code's per-request attribution line in the system
// prompt ("x-anthropic-billing-header: cc_version=…; cch=…;"). Stripped here, on
// the translated direction only: it changes on every request, so it would stop a
// local backend reusing its prompt cache. A host passing a body through to
// Anthropic does not run this package on it.
var billingLine = regexp.MustCompile(`(?m)^x-anthropic-billing-header:[^\n]*\n?`)

// oaiPartOf turns a text or image block into an OpenAI content part.
func oaiPartOf(b antInBlock) (map[string]any, error) {
	switch b.Type {
	case "text":
		return map[string]any{"type": "text", "text": b.Text}, nil
	case "image":
		if b.Source == nil {
			return nil, fmt.Errorf("image block has no source")
		}
		u := b.Source.URL
		if b.Source.Type == "base64" {
			u = "data:" + b.Source.MediaType + ";base64," + b.Source.Data
		}
		if u == "" {
			return nil, fmt.Errorf("image source type %q is not supported", b.Source.Type)
		}
		return map[string]any{"type": "image_url", "image_url": map[string]string{"url": u}}, nil
	case "document":
		if b.Source != nil && b.Source.Type == "text" {
			return map[string]any{"type": "text", "text": b.Source.Data}, nil
		}
		return nil, fmt.Errorf("document blocks other than plain text are not supported for non-Anthropic models")
	}
	return nil, nil // thinking, redacted_thinking, server tool blocks: dropped
}

// userContent is OpenAI message content: a plain string when it is one text
// part (what every chat template handles best), else the part list.
func userContent(parts []map[string]any) any {
	if len(parts) == 1 && parts[0]["type"] == "text" {
		return parts[0]["text"]
	}
	return parts
}

// ReverseRequest translates an Anthropic Messages request into an OpenAI
// chat completions request.
func ReverseRequest(body []byte) ([]byte, error) {
	var in antInRequest
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, fmt.Errorf("body is not an Anthropic Messages request: %v", err)
	}
	out := map[string]any{"model": in.Model}
	var msgs []map[string]any

	sys, err := blocksOf(in.System)
	if err != nil {
		return nil, fmt.Errorf("system: %v", err)
	}
	var sysText []string
	for _, b := range sys {
		if t := strings.TrimSpace(billingLine.ReplaceAllString(b.Text, "")); b.Type == "text" && t != "" {
			sysText = append(sysText, t)
		}
	}
	if len(sysText) > 0 {
		msgs = append(msgs, map[string]any{"role": "system", "content": strings.Join(sysText, "\n\n")})
	}

	for i, raw := range in.Messages {
		var m struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, fmt.Errorf("messages[%d]: %v", i, err)
		}
		blocks, err := blocksOf(m.Content)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %v", i, err)
		}
		switch m.Role {
		case "assistant":
			var text []string
			var calls []map[string]any
			for _, b := range blocks {
				switch b.Type {
				case "text":
					text = append(text, b.Text)
				case "tool_use":
					args := string(b.Input)
					if args == "" || args == "null" {
						args = "{}"
					}
					calls = append(calls, map[string]any{"id": b.ID, "type": "function",
						"function": map[string]string{"name": b.Name, "arguments": args}})
				}
			}
			am := map[string]any{"role": "assistant", "content": strings.Join(text, "")}
			if len(calls) > 0 {
				am["tool_calls"] = calls
			}
			msgs = append(msgs, am)
		case "user":
			// Tool results first, each its own tool message; the rest of the
			// turn follows as one user message.
			var parts, toolImages []map[string]any
			for _, b := range blocks {
				if b.Type != "tool_result" {
					pt, err := oaiPartOf(b)
					if err != nil {
						return nil, fmt.Errorf("messages[%d]: %v", i, err)
					}
					if pt != nil {
						parts = append(parts, pt)
					}
					continue
				}
				inner, err := blocksOf(b.Content)
				if err != nil {
					return nil, fmt.Errorf("messages[%d] tool_result: %v", i, err)
				}
				var text []string
				for _, ib := range inner {
					switch ib.Type {
					case "text":
						text = append(text, ib.Text)
					case "image":
						if pt, err := oaiPartOf(ib); err == nil && pt != nil {
							toolImages = append(toolImages, pt)
						}
					}
				}
				content := strings.Join(text, "\n")
				if b.IsError && !strings.HasPrefix(content, "Error") {
					content = "Error: " + content
				}
				msgs = append(msgs, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": content})
			}
			// An image a tool returned rides in the user turn: a tool message
			// carries text only in most chat templates.
			parts = append(toolImages, parts...)
			if len(parts) > 0 {
				msgs = append(msgs, map[string]any{"role": "user", "content": userContent(parts)})
			}
		case "system":
			// Claude Code sends its environment block as a system message
			// AFTER the first user turn. Most chat templates refuse a system
			// message that is not first (Qwen's raises), so it goes as a user
			// message in the same place: the prefix before it stays the same,
			// which is what a prompt cache keys on.
			var text []string
			for _, b := range blocks {
				if b.Type == "text" {
					text = append(text, b.Text)
				}
			}
			msgs = append(msgs, map[string]any{"role": "user", "content": strings.Join(text, "\n\n")})
		default:
			return nil, fmt.Errorf("messages[%d]: role %q is not user, assistant or system", i, m.Role)
		}
	}
	out["messages"] = msgs

	var tools []map[string]any
	for _, raw := range in.Tools {
		var t struct {
			Type        string          `json:"type"`
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"input_schema"`
		}
		if json.Unmarshal(raw, &t) != nil || (t.Type != "" && t.Type != "custom") || len(t.InputSchema) == 0 {
			continue // a server tool (web_search_…) runs at Anthropic only
		}
		tools = append(tools, map[string]any{"type": "function",
			"function": map[string]any{"name": t.Name, "description": t.Description, "parameters": t.InputSchema}})
	}
	if len(tools) > 0 {
		out["tools"] = tools
		if tc := in.ToolChoice; tc != nil {
			switch tc.Type {
			case "auto":
				out["tool_choice"] = "auto"
			case "any":
				out["tool_choice"] = "required"
			case "none":
				out["tool_choice"] = "none"
			case "tool":
				out["tool_choice"] = map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
			}
			if tc.DisableParallelToolUse {
				out["parallel_tool_calls"] = false
			}
		}
	}
	if in.MaxTokens != nil {
		out["max_tokens"] = *in.MaxTokens
	}
	if len(in.StopSequences) > 0 {
		out["stop"] = in.StopSequences
	}
	if in.Temperature != nil {
		out["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		out["top_p"] = *in.TopP
	}
	if in.TopK != nil {
		out["top_k"] = *in.TopK
	}
	if in.Stream {
		out["stream"] = true
		out["stream_options"] = map[string]any{"include_usage": true}
	}
	return json.Marshal(out)
}

// StripThinking removes the thinking blocks a translated reply made (signed
// with signature) before a conversation goes back to Anthropic, which
// would refuse their signatures. When the LAST assistant turn loses one,
// thinking is switched off for this request: Anthropic requires that turn to
// keep its thinking while thinking is on, and the real one does not exist.
// The body is returned unchanged when there is nothing to strip, so the
// common case keeps its exact bytes.
func StripThinking(body []byte, signature string) []byte {
	if !bytes.Contains(body, []byte(`"`+signature+`"`)) {
		return body
	}
	var req map[string]json.RawMessage
	if json.Unmarshal(body, &req) != nil {
		return body
	}
	var msgs []map[string]json.RawMessage
	if json.Unmarshal(req["messages"], &msgs) != nil {
		return body
	}
	lastAssistant := -1
	for i, m := range msgs {
		if string(m["role"]) == `"assistant"` {
			lastAssistant = i
		}
	}
	changed, offLast := false, false
	for i, m := range msgs {
		var blocks []map[string]json.RawMessage
		if string(m["role"]) != `"assistant"` || json.Unmarshal(m["content"], &blocks) != nil {
			continue
		}
		kept := blocks[:0]
		for _, b := range blocks {
			if string(b["type"]) == `"thinking"` && string(b["signature"]) == `"`+signature+`"` {
				changed = true
				offLast = offLast || i == lastAssistant
				continue
			}
			kept = append(kept, b)
		}
		if len(kept) == 0 {
			kept = append(kept, map[string]json.RawMessage{"type": json.RawMessage(`"text"`), "text": json.RawMessage(`"…"`)})
		}
		m["content"], _ = json.Marshal(kept)
	}
	if !changed {
		return body
	}
	req["messages"], _ = json.Marshal(msgs)
	if offLast {
		delete(req, "thinking")
	}
	out, err := json.Marshal(req)
	if err != nil {
		return body
	}
	return out
}

// --- reply: OpenAI → Anthropic ---

type oaiToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaiReplyUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *oaiReplyUsage) anthropic() map[string]int {
	if u == nil {
		return map[string]int{"input_tokens": 0, "output_tokens": 0}
	}
	cached := u.PromptTokensDetails.CachedTokens
	return map[string]int{"input_tokens": max(u.PromptTokens-cached, 0), "output_tokens": u.CompletionTokens,
		"cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
}

// stopReason maps OpenAI's finish_reason to Anthropic's stop_reason.
func stopReason(finish string, sawTool bool) string {
	switch {
	case finish == "length":
		return "max_tokens"
	case finish == "tool_calls" || sawTool:
		return "tool_use"
	case finish == "content_filter":
		return "refusal"
	}
	return "end_turn"
}

var badToolID = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// toolID is an id Anthropic accepts: a translated model's ids can carry
// other characters, and the conversation may go back to Anthropic.
func toolID(id string) string {
	if id == "" {
		return "toolu_corrallm"
	}
	return badToolID.ReplaceAllString(id, "_")
}

func msgID(id string) string {
	if id == "" {
		return "msg_corrallm"
	}
	return "msg_" + badToolID.ReplaceAllString(strings.TrimPrefix(id, "chatcmpl-"), "_")
}

// ReverseResponse translates a non-streamed chat completion.
func ReverseResponse(body []byte, model, signature string) ([]byte, error) {
	var r struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content          string             `json:"content"`
				ReasoningContent string             `json:"reasoning_content"`
				ToolCalls        []oaiToolCallDelta `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *oaiReplyUsage `json:"usage"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("reply is not a chat completion: %w", err)
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("reply is not a chat completion: no choices")
	}
	content := []map[string]any{}
	finish := ""
	if len(r.Choices) > 0 {
		c := r.Choices[0]
		finish = c.FinishReason
		if c.Message.ReasoningContent != "" {
			content = append(content, map[string]any{"type": "thinking", "thinking": c.Message.ReasoningContent, "signature": signature})
		}
		if c.Message.Content != "" {
			content = append(content, map[string]any{"type": "text", "text": c.Message.Content})
		}
		for _, tc := range c.Message.ToolCalls {
			input := json.RawMessage(tc.Function.Arguments)
			if !json.Valid(input) {
				input = json.RawMessage("{}")
			}
			content = append(content, map[string]any{"type": "tool_use", "id": toolID(tc.ID), "name": tc.Function.Name, "input": input})
		}
	}
	sawTool := len(r.Choices) > 0 && len(r.Choices[0].Message.ToolCalls) > 0
	return json.Marshal(map[string]any{
		"id": msgID(r.ID), "type": "message", "role": "assistant", "model": model, "content": content,
		"stop_reason": stopReason(finish, sawTool), "stop_sequence": nil, "usage": r.Usage.anthropic(),
	})
}

// ReverseStream turns an OpenAI chat-completions SSE stream into Anthropic's
// event sequence: message_start, then per block content_block_start / _delta
// / _stop, then message_delta (stop_reason, usage) and message_stop.
type ReverseStream struct {
	emit      func([]byte)
	model     string
	signature string
	chunks    int // chat completion chunks seen
	pending   []byte
	started   bool
	done      bool
	open      string // type of the open block, "" if none
	index     int    // the open block's index; next block is index+1
	tool      int    // the open tool_use's OpenAI index
	sawTool   bool
	finish    string
	usage     *oaiReplyUsage
}

func (s *ReverseStream) send(event string, v any) {
	b, _ := json.Marshal(v)
	s.emit([]byte("event: " + event + "\ndata: " + string(b) + "\n\n"))
}

// Feed takes bytes of an OpenAI event stream, split anywhere.
func (s *ReverseStream) Feed(b []byte) {
	s.pending = append(s.pending, b...)
	for {
		i := bytes.IndexByte(s.pending, '\n')
		if i < 0 {
			return
		}
		line := bytes.TrimSpace(s.pending[:i])
		s.pending = s.pending[i+1:]
		if data, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			s.event(bytes.TrimSpace(data))
		}
	}
}

func (s *ReverseStream) start(id string) {
	if s.started {
		return
	}
	s.started = true
	s.index = -1
	s.send("message_start", map[string]any{"type": "message_start", "message": map[string]any{
		"id": msgID(id), "type": "message", "role": "assistant", "model": s.model, "content": []any{},
		"stop_reason": nil, "stop_sequence": nil, "usage": map[string]int{"input_tokens": 0, "output_tokens": 0},
	}})
}

func (s *ReverseStream) closeBlock() {
	if s.open == "" {
		return
	}
	if s.open == "thinking" {
		s.send("content_block_delta", map[string]any{"type": "content_block_delta", "index": s.index,
			"delta": map[string]string{"type": "signature_delta", "signature": s.signature}})
	}
	s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.index})
	s.open = ""
}

func (s *ReverseStream) openBlock(typ string, block map[string]any) {
	s.closeBlock()
	s.index++
	s.open = typ
	s.send("content_block_start", map[string]any{"type": "content_block_start", "index": s.index, "content_block": block})
}

func (s *ReverseStream) delta(d map[string]string) {
	s.send("content_block_delta", map[string]any{"type": "content_block_delta", "index": s.index, "delta": d})
}

func (s *ReverseStream) event(data []byte) {
	if s.done || len(data) == 0 {
		return
	}
	if bytes.Equal(data, []byte("[DONE]")) {
		s.End()
		return
	}
	var c struct {
		ID      string `json:"id"`
		Choices []struct {
			Delta struct {
				Content          string             `json:"content"`
				ReasoningContent string             `json:"reasoning_content"`
				ToolCalls        []oaiToolCallDelta `json:"tool_calls"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *oaiReplyUsage `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &c) != nil {
		return
	}
	s.chunks++
	s.start(c.ID)
	if c.Error != nil {
		s.send("error", map[string]any{"type": "error", "error": map[string]string{"type": "api_error", "message": c.Error.Message}})
		s.done = true
		return
	}
	if c.Usage != nil {
		s.usage = c.Usage
	}
	for _, ch := range c.Choices {
		d := ch.Delta
		if d.ReasoningContent != "" {
			if s.open != "thinking" {
				s.openBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})
			}
			s.delta(map[string]string{"type": "thinking_delta", "thinking": d.ReasoningContent})
		}
		if d.Content != "" {
			if s.open != "text" {
				s.openBlock("text", map[string]any{"type": "text", "text": ""})
			}
			s.delta(map[string]string{"type": "text_delta", "text": d.Content})
		}
		for _, tc := range d.ToolCalls {
			if s.open != "tool_use" || tc.Index != s.tool || (tc.ID != "" && tc.Function.Name != "") {
				s.openBlock("tool_use", map[string]any{"type": "tool_use", "id": toolID(tc.ID), "name": tc.Function.Name, "input": map[string]any{}})
				s.tool = tc.Index
				s.sawTool = true
			}
			if tc.Function.Arguments != "" {
				s.delta(map[string]string{"type": "input_json_delta", "partial_json": tc.Function.Arguments})
			}
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			s.finish = *ch.FinishReason
		}
	}
}

// End closes the message. Safe to call more than once, and on a stream that
// ended without [DONE]. It reports a stream that held no chat completion chunk at
// all, and sends nothing then, so the host can answer with an error of its own.
func (s *ReverseStream) End() error {
	if s.done {
		return nil
	}
	if s.chunks == 0 {
		s.done = true
		return fmt.Errorf("reply is not a chat completion stream")
	}
	s.start("")
	s.closeBlock()
	s.send("message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": stopReason(s.finish, s.sawTool), "stop_sequence": nil},
		"usage": s.usage.anthropic()})
	s.send("message_stop", map[string]string{"type": "message_stop"})
	s.done = true
	return nil
}

// ErrorType is the Messages API error type for an HTTP status.
func ErrorType(code int) string {
	switch code {
	case http.StatusBadRequest, http.StatusMethodNotAllowed, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusPaymentRequired:
		return "billing_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		return "overloaded_error"
	}
	return "api_error"
}

// ReverseError is a Messages API error body.
func ReverseError(code int, msg string) []byte {
	b, _ := json.Marshal(map[string]any{"type": "error",
		"error": map[string]string{"type": ErrorType(code), "message": msg}})
	return b
}

// EstimateTokens is a Messages request's tokens, about 4 characters each, for a
// count_tokens answer from a backend that has none. An estimate: say so.
func EstimateTokens(body []byte) int {
	var in antInRequest
	_ = json.Unmarshal(body, &in)
	chars := 0
	if sys, err := blocksOf(in.System); err == nil {
		for _, b := range sys {
			chars += len(b.Text)
		}
	}
	for _, raw := range in.Messages {
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		_ = json.Unmarshal(raw, &m)
		chars += len(m.Content) // JSON of the blocks: close enough for text, rough for tool calls
	}
	for _, t := range in.Tools {
		chars += len(t)
	}
	return (chars+3)/4 + 4*len(in.Messages)
}
