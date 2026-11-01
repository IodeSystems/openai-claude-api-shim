package shim

import "strings"

// price is a model's rates in USD per million tokens.
type price struct {
	input, write5m, write1h, read, output float64
}

// PricesAsOf is when prices was copied from
// https://platform.claude.com/docs/en/about-claude/pricing. Anthropic reports
// tokens and no money, so a cost has to be worked out here; a model missing from
// the table gets no cost, not a guess. Standard rates only: fast mode, batch and
// inference_geo are not sent by this package.
const PricesAsOf = "2026-11-01"

// prices is matched by longest prefix, so a dated id finds its model and
// claude-opus-5-5 is not taken for claude-opus-5.
var prices = map[string]price{
	"claude-fable-5-1":  {10, 12.50, 20, 0.25, 50},
	"claude-mythos-5-1": {10, 12.50, 20, 0.25, 50},
	"claude-fable-5":    {10, 12.50, 20, 1, 50},
	"claude-mythos-5":   {10, 12.50, 20, 1, 50},
	"claude-opus-5-5":   {4, 5, 8, 0.20, 20},
	"claude-opus-5":     {5, 6.25, 10, 0.50, 25},
	"claude-opus-4-8":   {5, 6.25, 10, 0.50, 25},
	"claude-opus-4-7":   {5, 6.25, 10, 0.50, 25},
	"claude-opus-4-6":   {5, 6.25, 10, 0.50, 25},
	"claude-opus-4-5":   {5, 6.25, 10, 0.50, 25},
	"claude-sonnet-5-5": {2, 2.50, 4, 0.10, 10},
	"claude-sonnet-5":   {2, 2.50, 4, 0.20, 10},
	"claude-sonnet-4-6": {3, 3.75, 6, 0.30, 15},
	"claude-sonnet-4-5": {3, 3.75, 6, 0.30, 15},
	"claude-haiku-5-5":  {0.10, 0.125, 0.20, 0.01, 0.50},
	"claude-haiku-4-5":  {1, 1.25, 2, 0.10, 5},
}

// longPrompt is the second tier: a Haiku 5.5 request whose prompt, cache reads
// and writes included, is over 100,000 tokens pays these for all of it.
var longPrompt = map[string]price{
	"claude-haiku-5-5": {0.50, 0.625, 1, 0.05, 2.50},
}

const longPromptTokens = 100_000

func lookup(table map[string]price, model string) (price, bool) {
	var best string
	for name := range table {
		if strings.HasPrefix(model, name) && len(name) > len(best) {
			best = name
		}
	}
	p, ok := table[best]
	return p, ok
}

// cost is what u cost in USD, nil when model has no price.
func cost(model string, u antUsage) *float64 {
	p, ok := lookup(prices, model)
	if !ok {
		return nil
	}
	if u.InputTokens+u.CacheReadInputTokens+u.CacheCreationInputTokens > longPromptTokens {
		if long, ok := lookup(longPrompt, model); ok {
			p = long
		}
	}
	// Writes not broken down by TTL are at the 5-minute rate, the default.
	long := u.CacheCreation.Ephemeral1h
	short := u.CacheCreationInputTokens - long
	usd := (float64(u.InputTokens)*p.input + float64(short)*p.write5m + float64(long)*p.write1h +
		float64(u.CacheReadInputTokens)*p.read + float64(u.OutputTokens)*p.output) / 1e6
	return &usd
}
