// Package shim translates between the OpenAI chat completions shape and the
// Claude Messages API, as bytes. It does no HTTP and holds no credentials: a host
// sends what Request returns to /v1/messages with its own transport, and reads
// the reply through Response, Stream and Error. So a client keeps its retries,
// its headers and its one code path for every other provider, and a proxy can
// put it in front of any OpenAI caller. Standard library only.
//
// It exists for prompt caching: Anthropic's OpenAI-compatible endpoint has no
// cache_control, so a caller that resends a long conversation pays full input
// price for all of it, every round. This package places the breakpoints.
//
// What does not cross:
//
//   - temperature, top_p, top_k, seed, the penalties, logit_bias, logprobs, n,
//     response_format, user, metadata and every llama.cpp extension. The body is
//     built from the fields named here, so the rest is dropped by construction.
//     Current models answer sampling settings with a 400.
//   - Thinking blocks: an OpenAI history carries no signature, and Anthropic only
//     accepts blocks it signed. reasoning_content on a replayed turn is dropped.
//   - Server tools and redacted thinking in a reply have no OpenAI form.
//
// Refused with an error rather than lost: content parts other than text and
// image_url, image URLs that are neither data: nor http(s), and roles other than
// system, developer, user, assistant and tool.
//
// Cache writes cost more than input (1.25×, 2× for a 1h TTL). The usage this
// package returns separates them, in cache_creation_input_tokens, for a meter
// that wants to price them.
package shim

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// defaultMaxTokens is the max_tokens sent when the caller set none and
// the options name no default. Anthropic REQUIRES the field; OpenAI callers
// routinely omit it. 16000 keeps a non-streaming reply inside the usual HTTP
// timeouts and fits every current Claude model's output cap; a stream has no
// such timeout, so it gets more room.
const (
	defaultMaxTokens       = 16000
	defaultStreamMaxTokens = 64000
)

// APIVersion is the anthropic-version header to send with a translated request.
const APIVersion = "2023-06-01"

// maxBreakpoints is Anthropic's per-request cap on cache_control
// markers. A request over it is a 400, so placement stops at it.
const maxBreakpoints = 4

// Options carries the per-backend settings the translation needs.
type Options struct {
	MaxTokens int    // the model's configured cap; the fallback when the caller sent none
	CacheTTL  string // "", "5m" or "1h" — the TTL on breakpoints this package places
}

// --- OpenAI request shapes (input) ---

type oaiChatRequest struct {
	Model               string          `json:"model"`
	Messages            []oaiMessage    `json:"messages"`
	Tools               []oaiTool       `json:"tools"`
	ToolChoice          json.RawMessage `json:"tool_choice"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	Stop                json.RawMessage `json:"stop"`
	ReasoningEffort     string          `json:"reasoning_effort"`
	Stream              bool            `json:"stream"`
}

type oaiMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []oaiToolCall   `json:"tool_calls"`
	ToolCallID string          `json:"tool_call_id"`
}

type oaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// oaiPart is one element of an OpenAI content array. cache_control is not
// OpenAI's, but clients written against OpenRouter put it on parts; it is
// honoured verbatim and counts against the breakpoint cap.
type oaiPart struct {
	Type         string          `json:"type"`
	Text         string          `json:"text"`
	Refusal      string          `json:"refusal"`
	ImageURL     json.RawMessage `json:"image_url"`
	CacheControl json.RawMessage `json:"cache_control"`
}

type oaiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
		Strict      *bool           `json:"strict"`
	} `json:"function"`
	CacheControl json.RawMessage `json:"cache_control"`
}

// --- Anthropic request shapes (output) ---

type antRequest struct {
	Model         string         `json:"model"`
	MaxTokens     int            `json:"max_tokens"`
	System        []antBlock     `json:"system,omitempty"`
	Messages      []antMessage   `json:"messages"`
	Tools         []antTool      `json:"tools,omitempty"`
	ToolChoice    *antToolChoice `json:"tool_choice,omitempty"`
	StopSequences []string       `json:"stop_sequences,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	OutputConfig  *antOutput     `json:"output_config,omitempty"`
}

type antOutput struct {
	Effort string `json:"effort,omitempty"`
}

type antMessage struct {
	Role    string     `json:"role"`
	Content []antBlock `json:"content"`
}

// antBlock is the union of the content blocks this package sends. Only the fields
// a given type uses are set; omitempty keeps the others off the wire.
type antBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	Source       *antSource      `json:"source,omitempty"`
	ID           string          `json:"id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        json.RawMessage `json:"input,omitempty"`
	ToolUseID    string          `json:"tool_use_id,omitempty"`
	IsError      bool            `json:"is_error,omitempty"`
	Content      []antBlock      `json:"content,omitempty"`
	CacheControl json.RawMessage `json:"cache_control,omitempty"`
}

type antSource struct {
	Type      string `json:"type"` // base64 | url
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type antTool struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema"`
	Strict       *bool           `json:"strict,omitempty"`
	CacheControl json.RawMessage `json:"cache_control,omitempty"`
	// EagerInputStreaming streams a tool's input as it is generated instead of
	// in one burst at the end; the caller checks the JSON it gets.
	EagerInputStreaming bool `json:"eager_input_streaming,omitempty"`
}

type antToolChoice struct {
	Type                   string `json:"type"` // auto | any | tool | none
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// Request translates an OpenAI chat.completions body into an
// Anthropic Messages body, with cache breakpoints placed. An error is the
// caller's request being untranslatable, and is answered as a 400.
func Request(body []byte, opts Options) ([]byte, error) {
	var in oaiChatRequest
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, fmt.Errorf("request is not a JSON chat completion: %w", err)
	}
	out := antRequest{Model: in.Model, Stream: in.Stream}
	if e := effort(in.Model, in.ReasoningEffort); e != "" {
		out.OutputConfig = &antOutput{Effort: e}
	}

	// max_completion_tokens is OpenAI's current name and wins over the
	// deprecated max_tokens.
	switch {
	case in.MaxCompletionTokens != nil && *in.MaxCompletionTokens > 0:
		out.MaxTokens = *in.MaxCompletionTokens
	case in.MaxTokens != nil && *in.MaxTokens > 0:
		out.MaxTokens = *in.MaxTokens
	case opts.MaxTokens > 0:
		out.MaxTokens = opts.MaxTokens
	case in.Stream:
		out.MaxTokens = defaultStreamMaxTokens
	default:
		out.MaxTokens = defaultMaxTokens
	}

	stops, err := stopSequences(in.Stop)
	if err != nil {
		return nil, err
	}
	out.StopSequences = stops

	for i, m := range in.Messages {
		switch m.Role {
		case "system", "developer":
			// A mid-conversation system message is folded into the top-level
			// system in order. Anthropic's own mid-conversation system role is
			// model-dependent, and the top-level field works on every model.
			parts, err := contentParts(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			for _, p := range parts {
				b, ok, err := partToBlock(p)
				if err != nil {
					return nil, fmt.Errorf("messages[%d]: %w", i, err)
				}
				if !ok {
					continue
				}
				if b.Type != "text" {
					return nil, fmt.Errorf("messages[%d]: a %s message may only carry text", i, m.Role)
				}
				out.System = append(out.System, b)
			}
		case "user":
			blocks, err := partsToBlocks(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			out.Messages = appendMessage(out.Messages, "user", blocks)
		case "assistant":
			blocks, err := partsToBlocks(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, antBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: toolInput(tc.Function.Arguments),
				})
			}
			// reasoning_content on a replayed assistant turn is dropped:
			// Anthropic only accepts thinking blocks it signed, and an OpenAI
			// history carries no signature.
			out.Messages = appendMessage(out.Messages, "assistant", blocks)
		case "tool":
			blocks, err := partsToBlocks(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			// A tool result is a user-turn block in Anthropic's model; the
			// consecutive results of one parallel call merge into one user
			// message via appendMessage, which is what Anthropic requires.
			out.Messages = appendMessage(out.Messages, "user", []antBlock{{
				Type:      "tool_result",
				ToolUseID: m.ToolCallID,
				Content:   blocks,
			}})
		default:
			return nil, fmt.Errorf("messages[%d]: role %q has no Anthropic equivalent", i, m.Role)
		}
	}
	out.Messages = repair(out.Messages)
	for i := range out.Messages {
		if out.Messages[i].Role == "user" {
			toolResultsFirst(out.Messages[i].Content)
		}
	}

	for _, t := range in.Tools {
		if t.Type != "" && t.Type != "function" {
			return nil, fmt.Errorf("tool type %q is not supported (only function)", t.Type)
		}
		schema := t.Function.Parameters
		if len(bytes.TrimSpace(schema)) == 0 || bytes.Equal(bytes.TrimSpace(schema), []byte("null")) {
			// Anthropic requires an object schema; OpenAI allows a tool with none.
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		at := antTool{
			Name:         t.Function.Name,
			Description:  t.Function.Description,
			InputSchema:  schema,
			CacheControl: t.CacheControl,

			EagerInputStreaming: in.Stream,
		}
		if t.Function.Strict != nil && *t.Function.Strict {
			at.Strict = t.Function.Strict
		}
		out.Tools = append(out.Tools, at)
	}
	tc, err := toolChoice(in.ToolChoice, in.ParallelToolCalls, len(out.Tools) > 0)
	if err != nil {
		return nil, err
	}
	out.ToolChoice, out.Tools = unforced(in.Model, tc, out.Tools)

	placeCacheBreakpoints(&out, opts.CacheTTL)
	return json.Marshal(out)
}

// contentParts normalises an OpenAI content field — absent, a string, or an
// array of parts — to parts.
func contentParts(raw json.RawMessage) ([]oaiPart, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return []oaiPart{{Type: "text", Text: s}}, nil
	}
	var parts []oaiPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("content is neither a string nor an array of parts")
	}
	return parts, nil
}

func partsToBlocks(raw json.RawMessage) ([]antBlock, error) {
	parts, err := contentParts(raw)
	if err != nil {
		return nil, err
	}
	var out []antBlock
	for _, p := range parts {
		b, ok, err := partToBlock(p)
		if err != nil {
			return nil, err
		}
		if ok {
			out = append(out, b)
		}
	}
	return out, nil
}

// partToBlock converts one content part. ok=false means the part carries
// nothing and is skipped: Anthropic rejects an empty text block, and an empty
// string is how OpenAI clients spell "no text" next to tool_calls.
func partToBlock(p oaiPart) (antBlock, bool, error) {
	switch p.Type {
	case "text", "input_text", "":
		if p.Text == "" {
			return antBlock{}, false, nil
		}
		return antBlock{Type: "text", Text: p.Text, CacheControl: p.CacheControl}, true, nil
	case "refusal":
		if p.Refusal == "" {
			return antBlock{}, false, nil
		}
		return antBlock{Type: "text", Text: p.Refusal, CacheControl: p.CacheControl}, true, nil
	case "image_url":
		src, err := imageSource(p.ImageURL)
		if err != nil {
			return antBlock{}, false, err
		}
		return antBlock{Type: "image", Source: src, CacheControl: p.CacheControl}, true, nil
	default:
		return antBlock{}, false, fmt.Errorf("content part type %q is not supported by an Anthropic backend", p.Type)
	}
}

// imageSource maps an OpenAI image_url ({url} or, from lax clients, a bare
// string) to an Anthropic image source: a data: URL becomes base64, an http(s)
// URL is passed for Anthropic to fetch.
func imageSource(raw json.RawMessage) (*antSource, error) {
	var u string
	var obj struct {
		URL string `json:"url"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.URL != "" {
		u = obj.URL
	} else if json.Unmarshal(raw, &u) != nil || u == "" {
		return nil, fmt.Errorf("image_url part has no url")
	}
	switch {
	case strings.HasPrefix(u, "data:"):
		meta, data, ok := strings.Cut(strings.TrimPrefix(u, "data:"), ",")
		media, enc, _ := strings.Cut(meta, ";")
		if !ok || enc != "base64" || media == "" {
			return nil, fmt.Errorf("image data: URL must be data:<media-type>;base64,<data>")
		}
		return &antSource{Type: "base64", MediaType: media, Data: data}, nil
	case strings.HasPrefix(u, "http://"), strings.HasPrefix(u, "https://"):
		return &antSource{Type: "url", URL: u}, nil
	default:
		return nil, fmt.Errorf("image url must be a data: or http(s) URL")
	}
}

// toolInput turns an OpenAI arguments string into an Anthropic input object.
//
// A non-object (empty, or JSON cut off by max_tokens) becomes {} rather than an
// error: it arrives on a REPLAYED turn, the tool_result that followed it has
// already told the model what happened, and refusing it would make the
// conversation unsendable for good.
func toolInput(args string) json.RawMessage {
	a := bytes.TrimSpace([]byte(args))
	if len(a) > 0 && a[0] == '{' && json.Valid(a) {
		return json.RawMessage(a)
	}
	return json.RawMessage(`{}`)
}

// appendMessage adds a turn, merging it into the previous one when the role
// repeats. Anthropic requires strictly alternating roles; OpenAI histories
// routinely carry adjacent user turns, and always carry adjacent tool results.
// A turn with no content is dropped — Anthropic refuses an empty one.
func appendMessage(msgs []antMessage, role string, blocks []antBlock) []antMessage {
	if len(blocks) == 0 {
		return msgs
	}
	if n := len(msgs); n > 0 && msgs[n-1].Role == role {
		msgs[n-1].Content = append(msgs[n-1].Content, blocks...)
		return msgs
	}
	return append(msgs, antMessage{Role: role, Content: blocks})
}

// toolResultsFirst moves tool_result blocks to the front of a user turn,
// keeping relative order. Anthropic requires them to lead the message that
// answers a tool_use; merging a tool reply with a following user message
// already produces that order, this covers a history that interleaved them.
func toolResultsFirst(blocks []antBlock) {
	var results, rest []antBlock
	for _, b := range blocks {
		if b.Type == "tool_result" {
			results = append(results, b)
		} else {
			rest = append(rest, b)
		}
	}
	copy(blocks, append(results, rest...))
}

// stopSequences reads OpenAI's stop (a string or an array of them).
// Whitespace-only sequences are dropped: Anthropic refuses them, and an OpenAI
// caller sending "\n" means a formatting preference, not a reason to 400.
func stopSequences(raw json.RawMessage) ([]string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	var list []string
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		list = []string{s}
	} else if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("stop must be a string or an array of strings")
	}
	out := list[:0]
	for _, s := range list {
		if strings.TrimSpace(s) != "" {
			out = append(out, s)
		}
	}
	return out, nil
}

// toolChoice maps OpenAI's tool_choice ("auto" | "none" | "required" |
// {type:function,function:{name}}) and parallel_tool_calls to Anthropic's.
// Without tools it is dropped: Anthropic rejects a tool_choice with nothing to
// choose. "required" and a named function become `any` and `tool`; unforced
// handles the models that refuse those.
func toolChoice(raw json.RawMessage, parallel *bool, haveTools bool) (*antToolChoice, error) {
	if !haveTools {
		return nil, nil
	}
	var tc *antToolChoice
	raw = bytes.TrimSpace(raw)
	switch {
	case len(raw) == 0 || bytes.Equal(raw, []byte("null")):
	case raw[0] == '"':
		var s string
		_ = json.Unmarshal(raw, &s)
		switch s {
		case "auto":
			tc = &antToolChoice{Type: "auto"}
		case "none":
			tc = &antToolChoice{Type: "none"}
		case "required":
			tc = &antToolChoice{Type: "any"}
		default:
			return nil, fmt.Errorf("tool_choice %q is not one of auto, none, required", s)
		}
	default:
		var obj struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if err := json.Unmarshal(raw, &obj); err != nil || obj.Function.Name == "" {
			return nil, fmt.Errorf("tool_choice object must name a function")
		}
		tc = &antToolChoice{Type: "tool", Name: obj.Function.Name}
	}
	if parallel != nil && !*parallel {
		if tc == nil {
			tc = &antToolChoice{Type: "auto"}
		}
		if tc.Type != "none" {
			tc.DisableParallelToolUse = true
		}
	}
	return tc, nil
}

// noForcedToolChoice lists the models that answer a forced tool_choice with a
// 400. By name: which models reject it moves by release, and a list can be read
// and fixed in one place.
var noForcedToolChoice = []string{"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"}

// unforced keeps a forced tool_choice from being a 400. On a model that refuses
// one it is auto; a named tool is then the only tool sent, so the model can call
// that or answer in text, never call another. `any` names no tool, so it stays
// auto over every tool.
//
// Tools lead the cached prefix, so narrowing them makes that one request miss the
// cache. A forced call is rare and has to mean what it says.
func unforced(model string, tc *antToolChoice, tools []antTool) (*antToolChoice, []antTool) {
	if tc == nil || (tc.Type != "any" && tc.Type != "tool") {
		return tc, tools
	}
	for _, m := range noForcedToolChoice {
		if !strings.HasPrefix(model, m) {
			continue
		}
		auto := &antToolChoice{Type: "auto", DisableParallelToolUse: tc.DisableParallelToolUse}
		for _, t := range tools {
			if t.Name == tc.Name {
				return auto, []antTool{t}
			}
		}
		return auto, tools
	}
	return tc, tools
}

// effort is the output_config.effort for an OpenAI reasoning_effort, "" for
// none. Thinking cannot be turned off on every model, so "none" is the lowest
// effort. The models before effort reject the field.
func effort(model, level string) string {
	for _, old := range []string{"claude-haiku-4-5", "claude-sonnet-4-5", "claude-3"} {
		if strings.HasPrefix(model, old) {
			return ""
		}
	}
	switch level {
	case "none", "minimal":
		return "low"
	case "low", "medium", "high", "xhigh", "max":
		return level
	}
	return ""
}

// repair makes a transcript sendable. A history that was folded or interrupted
// need not satisfy what the Messages API requires: every tool_use has a result in
// the next message, every result a tool_use, and the turns start and end with the
// user. Left alone, each of those is a 400 on every later request.
func repair(msgs []antMessage) []antMessage {
	var out []antMessage
	open := map[string]bool{} // tool_use ids waiting for a result
	var order []string
	closeOpen := func() []antBlock {
		var blocks []antBlock
		for _, id := range order {
			if open[id] {
				blocks = append(blocks, antBlock{Type: "tool_result", ToolUseID: id, IsError: true,
					Content: []antBlock{{Type: "text", Text: "(no result was recorded for this call)"}}})
			}
		}
		open, order = map[string]bool{}, nil
		return blocks
	}
	for _, m := range msgs {
		if m.Role == "assistant" {
			if pending := closeOpen(); len(pending) > 0 {
				out = appendMessage(out, "user", pending)
			}
			for _, b := range m.Content {
				if b.Type == "tool_use" {
					open[b.ID] = true
					order = append(order, b.ID)
				}
			}
			out = appendMessage(out, "assistant", m.Content)
			continue
		}
		var blocks []antBlock
		for _, b := range m.Content {
			switch {
			case b.Type != "tool_result":
				blocks = append(blocks, b)
			case open[b.ToolUseID]:
				delete(open, b.ToolUseID)
				blocks = append(blocks, b)
			default:
				// Its call was folded away. The text may still matter.
				blocks = append(blocks, b.Content...)
			}
		}
		blocks = append(closeOpen(), blocks...)
		out = appendMessage(out, "user", blocks)
	}
	if pending := closeOpen(); len(pending) > 0 {
		out = appendMessage(out, "user", pending)
	}
	if len(out) > 0 && out[0].Role != "user" {
		out = append([]antMessage{{Role: "user", Content: []antBlock{{Type: "text", Text: "(earlier conversation)"}}}}, out...)
	}
	if n := len(out); n > 0 && out[n-1].Role == "assistant" {
		out = append(out, antMessage{Role: "user", Content: []antBlock{{Type: "text", Text: "Continue."}}})
	}
	return out
}

// placeCacheBreakpoints marks the cacheable prefix, at most
// maxBreakpoints markers in all. Render order is tools → system →
// messages, so in priority order:
//
//  1. the last tool definition (the tool list rarely changes),
//  2. the last system block,
//  3. the last block of the FINAL user turn — writes this turn's prefix,
//  4. the last block of the user turn before it — the prefix the previous
//     request wrote, so this one reads it even when Anthropic's 20-block
//     lookback from (3) would not reach that far back.
//
// Markers the caller placed itself are kept verbatim and count against the
// cap first; a slot whose block already carries one is treated as covered.
// Prefixes shorter than the model's minimum (1024–4096 tokens) are simply not
// cached by Anthropic, so marking them costs nothing.
func placeCacheBreakpoints(req *antRequest, ttl string) {
	cc := json.RawMessage(`{"type":"ephemeral"}`)
	if ttl == "1h" {
		cc = json.RawMessage(`{"type":"ephemeral","ttl":"1h"}`)
	}
	budget := maxBreakpoints - countBreakpoints(req)
	mark := func(slot *json.RawMessage) {
		if len(*slot) > 0 || budget <= 0 {
			return
		}
		*slot = cc
		budget--
	}
	if n := len(req.Tools); n > 0 {
		mark(&req.Tools[n-1].CacheControl)
	}
	if n := len(req.System); n > 0 {
		mark(&req.System[n-1].CacheControl)
	}
	turns := 0
	for i := len(req.Messages) - 1; i >= 0 && turns < 2; i-- {
		m := &req.Messages[i]
		if m.Role != "user" || len(m.Content) == 0 {
			continue
		}
		mark(&m.Content[len(m.Content)-1].CacheControl)
		turns++
	}
}

// countBreakpoints counts the cache_control markers already in a request,
// including those nested in tool_result content.
func countBreakpoints(req *antRequest) int {
	n := 0
	for _, t := range req.Tools {
		if len(t.CacheControl) > 0 {
			n++
		}
	}
	var walk func([]antBlock)
	walk = func(bs []antBlock) {
		for _, b := range bs {
			if len(b.CacheControl) > 0 {
				n++
			}
			walk(b.Content)
		}
	}
	walk(req.System)
	for _, m := range req.Messages {
		walk(m.Content)
	}
	return n
}

// --- Anthropic response shapes (input) ---

type antUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

type antContent struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	Thinking string          `json:"thinking"`
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Input    json.RawMessage `json:"input"`
}

type antResponse struct {
	ID         string       `json:"id"`
	Model      string       `json:"model"`
	Content    []antContent `json:"content"`
	StopReason string       `json:"stop_reason"`
	Usage      antUsage     `json:"usage"`
}

// --- OpenAI response shapes (output) ---

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// Anthropic's two cache classes, for a meter that prices writes.
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	PromptTokensDetails      struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// usageOf restates Anthropic usage in OpenAI's terms. Anthropic's
// input_tokens EXCLUDES both cache classes, so the prompt total is the sum of
// all three; cached_tokens is the read share only.
func usageOf(u antUsage) oaiUsage {
	var o oaiUsage
	o.PromptTokens = u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	o.CompletionTokens = u.OutputTokens
	o.TotalTokens = o.PromptTokens + o.CompletionTokens
	o.CacheReadInputTokens = u.CacheReadInputTokens
	o.CacheCreationInputTokens = u.CacheCreationInputTokens
	o.PromptTokensDetails.CachedTokens = u.CacheReadInputTokens
	return o
}

// finishOf maps Anthropic's stop_reason to OpenAI's finish_reason.
func finishOf(stop string) string {
	switch stop {
	case "max_tokens", "model_context_window_exceeded":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default: // end_turn, stop_sequence, pause_turn, unknown
		return "stop"
	}
}

type oaiFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

type oaiToolCallOut struct {
	Index    *int            `json:"index,omitempty"` // streaming deltas only
	ID       string          `json:"id,omitempty"`
	Type     string          `json:"type,omitempty"`
	Function oaiFunctionCall `json:"function"`
}

type oaiMessageOut struct {
	Role             string           `json:"role"`
	Content          *string          `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCallOut `json:"tool_calls,omitempty"`
}

type oaiChoice struct {
	Index        int           `json:"index"`
	Message      oaiMessageOut `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type oaiCompletion struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   oaiUsage    `json:"usage"`
}

// Response translates a non-streaming Messages reply into a
// chat.completion. Text blocks concatenate into content, thinking blocks into
// reasoning_content, tool_use blocks become tool_calls. Server-tool and
// redacted-thinking blocks have no OpenAI form and are dropped.
func Response(body []byte) ([]byte, error) {
	var in antResponse
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, err
	}
	var out oaiCompletion
	out.ID, out.Object, out.Created, out.Model = in.ID, "chat.completion", time.Now().Unix(), in.Model
	out.Choices = make([]oaiChoice, 1)
	ch := &out.Choices[0]
	ch.Message.Role = "assistant"
	var text, reasoning strings.Builder
	for _, b := range in.Content {
		switch b.Type {
		case "text":
			text.WriteString(b.Text)
		case "thinking":
			reasoning.WriteString(b.Thinking)
		case "tool_use":
			args := string(b.Input)
			if len(bytes.TrimSpace(b.Input)) == 0 {
				args = "{}"
			}
			ch.Message.ToolCalls = append(ch.Message.ToolCalls, oaiToolCallOut{
				ID: b.ID, Type: "function", Function: oaiFunctionCall{Name: b.Name, Arguments: args},
			})
		}
	}
	// OpenAI sends content:null beside tool_calls with no text.
	if s := text.String(); s != "" || len(ch.Message.ToolCalls) == 0 {
		ch.Message.Content = &s
	}
	ch.Message.ReasoningContent = reasoning.String()
	ch.FinishReason = finishOf(in.StopReason)
	out.Usage = usageOf(in.Usage)
	return json.Marshal(out)
}

// Error restates an Anthropic error body ({type:error, error:{type,
// message}}) as OpenAI's envelope. A body that is not Anthropic's shape is
// returned unchanged — it is still the only account of what went wrong.
func Error(body []byte) []byte {
	var in struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &in) != nil || in.Error.Message == "" {
		return body
	}
	out, err := json.Marshal(map[string]any{"error": map[string]any{
		"message": in.Error.Message, "type": in.Error.Type, "code": nil,
	}})
	if err != nil {
		return body
	}
	return out
}

// Transcode reads a Messages event stream and yields OpenAI SSE. Closing the
// result does not close r; a host that cuts a stream closes that itself.
func Transcode(r io.Reader, opts StreamOptions) io.Reader {
	pr, pw := io.Pipe()
	go func() {
		s := NewStream(func(b []byte) { _, _ = pw.Write(b) }, opts)
		buf := make([]byte, 32<<10)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				s.Feed(buf[:n])
			}
			if err != nil {
				s.Finish()
				if err == io.EOF {
					err = nil
				}
				pw.CloseWithError(err)
				return
			}
		}
	}()
	return pr
}

// StreamOptions are the choices a stream's reader makes.
type StreamOptions struct {
	// RefusalAsError reports a refusal as an {"error":…} event, with its category
	// and explanation, instead of finish_reason content_filter: for a caller that
	// would take the empty reply for an answer.
	RefusalAsError bool
}

// NewStream returns a Stream that writes OpenAI SSE through emit.
func NewStream(emit func([]byte), opts StreamOptions) *Stream {
	return &Stream{emit: emit, opts: opts}
}

// Stream translates Anthropic's SSE events into OpenAI
// chat.completion.chunk events:
//
//	message_start            → first chunk, delta {role: assistant}
//	content_block_start      → tool_use opens a tool_calls entry (id, name)
//	content_block_delta      → text_delta → content, thinking_delta →
//	                           reasoning_content, input_json_delta → arguments
//	message_delta            → stop_reason and final usage, held
//	message_stop             → finish chunk, usage chunk, [DONE]
//	error                    → {"error":…} chunk, [DONE]
//	content_block_stop       → "{}" for a call that streamed no arguments
//	ping, signature_delta    → nothing
//
// The usage chunk is sent whether or not the caller asked for
// stream_options.include_usage: it is what extractUsage meters from, and a
// chunk with empty choices is part of the OpenAI stream contract.
type Stream struct {
	emit    func([]byte)
	opts    StreamOptions
	refused string
	pending []byte
	id      string
	model   string
	created int64
	usage   antUsage
	stop    string
	tools   map[int]int  // Anthropic content-block index → OpenAI tool_calls index
	noArgs  map[int]bool // OpenAI tool_calls index → no arguments have streamed
	started bool
	done    bool
}

// Feed takes bytes of a Messages event stream, split anywhere.
func (s *Stream) Feed(b []byte) {
	if s.done {
		return
	}
	// JSON escapes CR inside strings, so a raw CR is only ever a line ending.
	s.pending = append(s.pending, bytes.ReplaceAll(b, []byte("\r"), nil)...)
	for {
		i := bytes.Index(s.pending, []byte("\n\n"))
		if i < 0 {
			return
		}
		ev := s.pending[:i]
		s.pending = s.pending[i+2:]
		s.event(ev)
		if s.done {
			return
		}
	}
}

// event handles one SSE event. The JSON payload's own `type` is authoritative;
// the `event:` line repeats it and is ignored.
func (s *Stream) event(raw []byte) {
	var data []byte
	for _, line := range bytes.Split(raw, []byte("\n")) {
		if d, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			if len(data) > 0 {
				data = append(data, '\n')
			}
			data = append(data, bytes.TrimSpace(d)...)
		}
	}
	if len(data) == 0 {
		return
	}
	var ev struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message struct {
			ID    string   `json:"id"`
			Model string   `json:"model"`
			Usage antUsage `json:"usage"`
		} `json:"message"`
		ContentBlock antContent `json:"content_block"`
		Delta        struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
			StopDetails struct {
				Category    string `json:"category"`
				Explanation string `json:"explanation"`
			} `json:"stop_details"`
		} `json:"delta"`
		Usage antUsage        `json:"usage"`
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &ev) != nil {
		return
	}
	switch ev.Type {
	case "message_start":
		s.id, s.model, s.usage = ev.Message.ID, ev.Message.Model, ev.Message.Usage
		s.created = time.Now().Unix()
		s.started = true
		empty := ""
		s.chunk(oaiDelta{Role: "assistant", Content: &empty}, nil)
	case "content_block_start":
		switch ev.ContentBlock.Type {
		case "text":
			if t := ev.ContentBlock.Text; t != "" {
				s.chunk(oaiDelta{Content: &t}, nil)
			}
		case "thinking":
			if t := ev.ContentBlock.Thinking; t != "" {
				s.chunk(oaiDelta{ReasoningContent: t}, nil)
			}
		case "tool_use":
			if s.tools == nil {
				s.tools = map[int]int{}
				s.noArgs = map[int]bool{}
			}
			idx := len(s.tools)
			s.tools[ev.Index] = idx
			s.noArgs[idx] = true
			s.chunk(oaiDelta{ToolCalls: []oaiToolCallOut{{
				Index: &idx, ID: ev.ContentBlock.ID, Type: "function",
				Function: oaiFunctionCall{Name: ev.ContentBlock.Name},
			}}}, nil)
		}
	case "content_block_delta":
		switch ev.Delta.Type {
		case "text_delta":
			if t := ev.Delta.Text; t != "" {
				s.chunk(oaiDelta{Content: &t}, nil)
			}
		case "thinking_delta":
			if t := ev.Delta.Thinking; t != "" {
				s.chunk(oaiDelta{ReasoningContent: t}, nil)
			}
		case "input_json_delta":
			idx, ok := s.tools[ev.Index]
			if ok && ev.Delta.PartialJSON != "" {
				s.noArgs[idx] = false
				s.chunk(oaiDelta{ToolCalls: []oaiToolCallOut{{
					Index: &idx, Function: oaiFunctionCall{Arguments: ev.Delta.PartialJSON},
				}}}, nil)
			}
		}
	case "content_block_stop":
		// A call with no arguments streams none, and "" is not JSON.
		if idx, ok := s.tools[ev.Index]; ok && s.noArgs[idx] {
			s.noArgs[idx] = false
			s.chunk(oaiDelta{ToolCalls: []oaiToolCallOut{{Index: &idx, Function: oaiFunctionCall{Arguments: "{}"}}}}, nil)
		}
	case "message_delta":
		if ev.Delta.StopReason != "" {
			s.stop = ev.Delta.StopReason
		}
		if s.stop == "refusal" {
			s.refused = strings.TrimSpace("the model refused the request: " + ev.Delta.StopDetails.Category + " " + ev.Delta.StopDetails.Explanation)
		}
		// output_tokens is cumulative here; the input classes are repeated by
		// newer API versions and absent (zero) from older ones.
		s.usage.OutputTokens = ev.Usage.OutputTokens
		if ev.Usage.InputTokens > 0 {
			s.usage.InputTokens = ev.Usage.InputTokens
		}
		if ev.Usage.CacheReadInputTokens > 0 {
			s.usage.CacheReadInputTokens = ev.Usage.CacheReadInputTokens
		}
		if ev.Usage.CacheCreationInputTokens > 0 {
			s.usage.CacheCreationInputTokens = ev.Usage.CacheCreationInputTokens
		}
	case "message_stop":
		if s.opts.RefusalAsError && s.refused != "" {
			b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": s.refused, "type": "refusal"}})
			s.emit(append(append([]byte("data: "), b...), '\n', '\n'))
			s.emit([]byte("data: [DONE]\n\n"))
			s.done = true
			return
		}
		fr := finishOf(s.stop)
		s.chunk(oaiDelta{}, &fr)
		s.usageChunk()
		s.emit([]byte("data: [DONE]\n\n"))
		s.done = true
	case "error":
		if b := Error(data); b != nil {
			s.emit(append(append([]byte("data: "), b...), '\n', '\n'))
		}
		s.emit([]byte("data: [DONE]\n\n"))
		s.done = true
	}
}

// Finish closes a stream that ended without message_stop — the upstream
// dropped, or the request was cancelled. The usage seen so far is sent so the
// input the caller was billed for is metered, but no finish_reason and no
// [DONE]: the caller must still be able to tell the answer was cut off.
func (s *Stream) Finish() {
	if s.started && !s.done {
		s.usageChunk()
		s.done = true
	}
}

type oaiDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          *string          `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCallOut `json:"tool_calls,omitempty"`
}

type oaiChunkChoice struct {
	Index        int      `json:"index"`
	Delta        oaiDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type oaiChunk struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []oaiChunkChoice `json:"choices"`
	Usage   *oaiUsage        `json:"usage,omitempty"`
}

func (s *Stream) chunk(d oaiDelta, finish *string) {
	s.write(oaiChunk{ID: s.id, Object: "chat.completion.chunk", Created: s.created, Model: s.model,
		Choices: []oaiChunkChoice{{Delta: d, FinishReason: finish}}})
}

func (s *Stream) usageChunk() {
	u := usageOf(s.usage)
	s.write(oaiChunk{ID: s.id, Object: "chat.completion.chunk", Created: s.created, Model: s.model,
		Choices: []oaiChunkChoice{}, Usage: &u})
}

func (s *Stream) write(c oaiChunk) {
	b, err := json.Marshal(c)
	if err != nil {
		return
	}
	s.emit(append(append([]byte("data: "), b...), '\n', '\n'))
}
