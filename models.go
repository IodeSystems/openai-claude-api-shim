package shim

import (
	"encoding/json"
	"strings"
)

// A model alias is a line and -latest: opus-latest is the newest Opus, whatever it
// is called this month. /v1/models says each model's line and when it was made, so
// an alias needs no table here to go stale.
const aliasSuffix = "-latest"

// IsAlias reports whether model is an alias, not a model id.
func IsAlias(model string) bool {
	return strings.HasSuffix(model, aliasSuffix) && !strings.HasPrefix(model, "claude-")
}

// Model is what an alias needs from one /v1/models entry.
type Model struct {
	ID        string `json:"id"`
	Line      string `json:"line"`
	CreatedAt string `json:"created_at"` // RFC 3339, so it sorts as text in UTC
}

// Latest is the newest of models in alias's line.
func Latest(models []Model, alias string) (string, bool) {
	line := strings.TrimSuffix(alias, aliasSuffix)
	best, created := "", ""
	for _, m := range models {
		if m.Line == line && m.CreatedAt > created {
			best, created = m.ID, m.CreatedAt
		}
	}
	return best, best != ""
}

// LatestIn is Latest over a /v1/models body.
func LatestIn(body []byte, alias string) (string, bool) {
	var r struct {
		Data []Model `json:"data"`
	}
	if json.Unmarshal(body, &r) != nil {
		return "", false
	}
	return Latest(r.Data, alias)
}
