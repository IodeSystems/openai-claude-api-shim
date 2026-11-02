package shim

import "strings"

// Price is a model's rates in USD per million tokens. Long, when set, replaces
// them for a request whose prompt is over LongOver tokens, cache reads and writes
// included, and applies to all of that request.
type Price struct {
	Input    float64 `json:"input"`
	Write5m  float64 `json:"write_5m"`
	Write1h  float64 `json:"write_1h"`
	Read     float64 `json:"read"`
	Output   float64 `json:"output"`
	Long     *Price  `json:"long,omitempty"`
	LongOver int     `json:"long_over,omitempty"`
}

// Prices maps a model id, or a prefix of one, to its rates. A host passes its own
// through StreamOptions: they win over the built-in table and add models to it.
type Prices map[string]Price

// PricesAsOf is when the built-in table was copied from
// https://platform.claude.com/docs/en/about-claude/pricing. Anthropic reports
// tokens and no money, so a cost has to be worked out here, and it is an estimate:
// usage marks it cost_estimated, and the provider's invoice is the real figure. A
// model missing from the table gets no cost, not a guess. Standard rates only:
// fast mode, batch and inference_geo are not sent by this package.
const PricesAsOf = "2026-11-01"

// builtin is matched by longest prefix, so a dated id finds its model and
// claude-opus-5-5 is not taken for claude-opus-5.
var builtin = Prices{
	"claude-fable-5-1":  {Input: 10, Write5m: 12.50, Write1h: 20, Read: 0.25, Output: 50},
	"claude-mythos-5-1": {Input: 10, Write5m: 12.50, Write1h: 20, Read: 0.25, Output: 50},
	"claude-fable-5":    {Input: 10, Write5m: 12.50, Write1h: 20, Read: 1, Output: 50},
	"claude-mythos-5":   {Input: 10, Write5m: 12.50, Write1h: 20, Read: 1, Output: 50},
	"claude-opus-5-5":   {Input: 4, Write5m: 5, Write1h: 8, Read: 0.20, Output: 20},
	"claude-opus-5":     {Input: 5, Write5m: 6.25, Write1h: 10, Read: 0.50, Output: 25},
	"claude-opus-4-8":   {Input: 5, Write5m: 6.25, Write1h: 10, Read: 0.50, Output: 25},
	"claude-opus-4-7":   {Input: 5, Write5m: 6.25, Write1h: 10, Read: 0.50, Output: 25},
	"claude-opus-4-6":   {Input: 5, Write5m: 6.25, Write1h: 10, Read: 0.50, Output: 25},
	"claude-opus-4-5":   {Input: 5, Write5m: 6.25, Write1h: 10, Read: 0.50, Output: 25},
	"claude-sonnet-5-5": {Input: 2, Write5m: 2.50, Write1h: 4, Read: 0.10, Output: 10},
	"claude-sonnet-5":   {Input: 2, Write5m: 2.50, Write1h: 4, Read: 0.20, Output: 10},
	"claude-sonnet-4-6": {Input: 3, Write5m: 3.75, Write1h: 6, Read: 0.30, Output: 15},
	"claude-sonnet-4-5": {Input: 3, Write5m: 3.75, Write1h: 6, Read: 0.30, Output: 15},
	"claude-haiku-5-5": {Input: 0.10, Write5m: 0.125, Write1h: 0.20, Read: 0.01, Output: 0.50, LongOver: 100_000,
		Long: &Price{Input: 0.50, Write5m: 0.625, Write1h: 1, Read: 0.05, Output: 2.50}},
	"claude-haiku-4-5": {Input: 1, Write5m: 1.25, Write1h: 2, Read: 0.10, Output: 5},
}

// lookup finds model's rates: the longest matching prefix in the host's table,
// else in the built-in one. A host's entry wins even over a longer built-in one,
// because it is the one somebody set.
func lookup(own Prices, model string) (Price, bool) {
	for _, table := range []Prices{own, builtin} {
		best, found := "", false
		for name := range table {
			if strings.HasPrefix(model, name) && len(name) >= len(best) {
				best, found = name, true
			}
		}
		if found {
			return table[best], true
		}
	}
	return Price{}, false
}

// PriceOf is model's rates, by own then the built-in table.
func PriceOf(own Prices, model string) (Price, bool) { return lookup(own, model) }

// cost is what u cost in USD, by own then the built-in table; nil with no price.
func cost(own Prices, model string, u antUsage) *float64 {
	p, ok := lookup(own, model)
	if !ok {
		return nil
	}
	if p.Long != nil && p.LongOver > 0 && u.InputTokens+u.CacheReadInputTokens+u.CacheCreationInputTokens > p.LongOver {
		p = *p.Long
	}
	// Writes not broken down by TTL are at the 5-minute rate, the default.
	long := u.CacheCreation.Ephemeral1h
	short := u.CacheCreationInputTokens - long
	usd := (float64(u.InputTokens)*p.Input + float64(short)*p.Write5m + float64(long)*p.Write1h +
		float64(u.CacheReadInputTokens)*p.Read + float64(u.OutputTokens)*p.Output) / 1e6
	return &usd
}

// Usage is a request's tokens, for Estimate. Input is what was neither read from
// nor written to the cache; CacheWrite1h is the part of CacheWrite at the 1-hour
// rate.
type Usage struct {
	Input, Output, CacheRead, CacheWrite, CacheWrite1h int
}

// Estimate is what u is worth in USD for model, by own then the built-in table,
// and false when neither has a price. For a host that sees usage this package did
// not translate, a passthrough to Anthropic for one. It is an estimate: say so.
func Estimate(own Prices, model string, u Usage) (float64, bool) {
	var a antUsage
	a.InputTokens, a.OutputTokens = u.Input, u.Output
	a.CacheReadInputTokens, a.CacheCreationInputTokens = u.CacheRead, u.CacheWrite
	a.CacheCreation.Ephemeral1h = u.CacheWrite1h
	c := cost(own, model, a)
	if c == nil {
		return 0, false
	}
	return *c, true
}
