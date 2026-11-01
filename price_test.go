package shim

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func usd(t *testing.T, model string, u antUsage) float64 {
	t.Helper()
	c := cost(nil, model, u)
	if c == nil {
		t.Fatalf("%s has no price", model)
	}
	return *c
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestCost(t *testing.T) {
	u := antUsage{InputTokens: 1_000_000, OutputTokens: 1_000_000, CacheReadInputTokens: 1_000_000, CacheCreationInputTokens: 1_000_000}
	for model, want := range map[string]float64{
		"claude-opus-5-5":           4 + 20 + 0.20 + 5,
		"claude-opus-5":             5 + 25 + 0.50 + 6.25, // not taken for 5-5, nor 5-5 for 5
		"claude-sonnet-5-5":         2 + 10 + 0.10 + 2.50,
		"claude-fable-5-1":          10 + 50 + 0.25 + 12.50,
		"claude-haiku-4-5-20251001": 1 + 5 + 0.10 + 1.25, // a dated id
	} {
		if got := usd(t, model, u); !near(got, want) {
			t.Errorf("%s: %v, want %v", model, got, want)
		}
	}
	if cost(nil, "gpt-5", u) != nil {
		t.Error("a model with no price got a cost")
	}
}

func TestCost_OneHourWrites(t *testing.T) {
	u := antUsage{CacheCreationInputTokens: 1_000_000}
	u.CacheCreation.Ephemeral1h = 400_000
	if got, want := usd(t, "claude-opus-5-5", u), 0.6*5+0.4*8; !near(got, want) {
		t.Errorf("%v, want %v", got, want)
	}
}

// Over the line every token pays the higher rate, and cached tokens count.
func TestCost_HaikuLongPrompt(t *testing.T) {
	short := antUsage{InputTokens: 100_000}
	if got := usd(t, "claude-haiku-5-5", short); !near(got, 0.01) {
		t.Errorf("at the line: %v", got)
	}
	long := antUsage{InputTokens: 1, CacheReadInputTokens: 100_000, OutputTokens: 1_000}
	want := (0.50*1 + 0.05*100_000 + 2.50*1_000) / 1e6
	if got := usd(t, "claude-haiku-5-5", long); !near(got, want) {
		t.Errorf("over it: %v, want %v", got, want)
	}
}

func TestCost_ReachesBothReplies(t *testing.T) {
	out, err := Response([]byte(`{"id":"m","model":"claude-opus-5-5","content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":1000000,"output_tokens":0}}`), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var v struct {
		Usage struct {
			Cost      *float64 `json:"cost"`
			Estimated bool     `json:"cost_estimated"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &v); err != nil || v.Usage.Cost == nil || !near(*v.Usage.Cost, 4) || !v.Usage.Estimated {
		t.Errorf("reply %s", out)
	}
	s := transcoded(t, sse(`{"type":"message_start","message":{"id":"m","model":"claude-opus-5-5","usage":{"input_tokens":1000000}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`, `{"type":"message_stop"}`), StreamOptions{})
	if !strings.Contains(s, `"cost":4`) || !strings.Contains(s, `"cost_estimated":true`) {
		t.Errorf("stream %s", s)
	}
}

// A host's rates win, and add a model the table lacks.
func TestCost_HostPrices(t *testing.T) {
	own := Prices{"claude-opus": {Input: 1}, "acme": {Input: 2}}
	u := antUsage{InputTokens: 1_000_000}
	for model, want := range map[string]float64{"claude-opus-5-5": 1, "acme-1": 2, "claude-sonnet-5-5": 2} {
		c := cost(own, model, u)
		if c == nil || !near(*c, want) {
			t.Errorf("%s: %v, want %v", model, c, want)
		}
	}
}

func TestCost_NoPriceSaysNothing(t *testing.T) {
	out, err := Response([]byte(`{"id":"m","model":"acme","content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":1}}`), StreamOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "cost") {
		t.Errorf("reply %s", out)
	}
}
