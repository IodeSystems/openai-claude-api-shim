package shim

import "testing"

const listing = `{"data":[
 {"id":"claude-opus-5","line":"opus","created_at":"2026-07-24T00:00:00Z"},
 {"id":"claude-opus-5-5","line":"opus","created_at":"2026-09-21T16:24:00Z"},
 {"id":"claude-sonnet-5-5","line":"sonnet","created_at":"2026-09-28T00:00:00Z"}]}`

func TestLatestIn(t *testing.T) {
	for alias, want := range map[string]string{"opus-latest": "claude-opus-5-5", "sonnet-latest": "claude-sonnet-5-5"} {
		if got, ok := LatestIn([]byte(listing), alias); !ok || got != want {
			t.Errorf("%s: %q %v", alias, got, ok)
		}
	}
	if _, ok := LatestIn([]byte(listing), "fable-latest"); ok {
		t.Error("a line with no model resolved")
	}
	if _, ok := LatestIn([]byte(`not json`), "opus-latest"); ok {
		t.Error("junk resolved")
	}
}

func TestIsAlias(t *testing.T) {
	if !IsAlias("opus-latest") || IsAlias("claude-opus-5-5") || IsAlias("claude-3-5-sonnet-latest") {
		t.Error("IsAlias")
	}
}
