# openai-claude-api-shim

Translates between the OpenAI chat completions shape and the
[Claude Messages API](https://platform.claude.com/docs), as bytes. No HTTP, no
credentials, standard library only.

A host keeps its transport and its one code path for every other provider. It
translates the request, sends it to `/v1/messages` itself, and translates the reply.

```go
body, err := shim.Request(openAIBody, shim.Options{})
// POST body to https://api.anthropic.com/v1/messages with x-api-key and
// anthropic-version: shim.APIVersion

reply, err := shim.Response(messagesBody, shim.StreamOptions{}) // JSON
stream := shim.Transcode(resp.Body, shim.StreamOptions{}) // SSE in, OpenAI SSE out
oaiErr := shim.Error(errorBody)                   // error envelope
```

The other direction, for a caller that speaks only the Messages API, such as Claude Code,
in front of an OpenAI backend:

```go
body, err := shim.ReverseRequest(messagesBody)
reply, err := shim.ReverseResponse(openAIBody, model, signature)
stream := shim.NewReverseStream(emit, model, signature) // stream.Feed(b), err := stream.End()
```

`signature` is the host's mark on thinking blocks made from another model's reasoning.
Keep it stable: live conversations hold blocks signed with it. A body passed straight
through to Anthropic should not go through this package.

`shim.Estimate` prices usage this package did not translate, and `shim.StripThinking`
removes those blocks before a conversation goes back to Anthropic.

Aliases: `IsAlias`, `Latest` and `LatestIn` resolve `opus-latest`, `sonnet-latest` and so on
to the newest model in that line, from a `/v1/models` listing. `PriceOf` is a model's rates.

## Why

Anthropic's OpenAI-compatible endpoint has no prompt caching, so a caller that resends
a long conversation pays full input price for all of it, every round. `/v1/messages`
caches, and this package places the breakpoints. Measured through an agent session on
Opus 5.5: 92% of the prompt tokens read from cache.

## What it does

- Up to 4 cache breakpoints: last tool, last system block, the last two user turns.
  Caller-placed `cache_control` markers are kept and count against the 4. `CacheTTL`
  sets `5m` or `1h`.
- `reasoning_effort` becomes `output_config.effort`. `none` is `low`, because thinking
  cannot be turned off on every model.
- A forced `tool_choice` is `any` or `tool`, except on the models that answer those
  with a 400 (Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1). There it is `auto`, and a
  named tool is the only tool sent. The model can then still answer in text.
- Repairs a transcript that was folded or interrupted: a call with no result gets one,
  a result with no call becomes text, and the turns start and end with the user.
- Usage reports `cached_tokens`, plus `cache_read_input_tokens` and
  `cache_creation_input_tokens`.
- Usage also carries an **estimated** `cost` in USD, marked `cost_estimated: true`.
  Anthropic reports tokens and no money, so it comes from the table in `price.go` (see
  `PricesAsOf`), with cache writes at their own rate and Haiku 5.5's long-prompt tier.
  A model not in the table gets no cost, not a guess. A host passes its own rates in
  `StreamOptions.Prices`; they win over the table.

## What does not cross

- `temperature`, `top_p`, `top_k`, `seed`, the penalties, `logit_bias`, `logprobs`,
  `n`, `response_format`, `user`, `metadata` and llama.cpp's extensions. Current models
  answer sampling settings with a 400.
- Thinking blocks. An OpenAI history has no signature, so they are not replayed.
- Server tools and redacted thinking in a reply.

A `file` part (or flat `input_file`) whose `file_data` is a data: URI crosses as a
`document` block: a PDF as a base64 source, text (`text/*`, JSON, XML, YAML) as a
plain-text source, the filename as its title. Citations are not enabled.

Audio (`input_audio`, `audio_url`) and video (`video_url`, `input_video`) parts, a file
by `file_id` or by URL, a file that is neither PDF nor text, any other part type, and
unknown roles are an error, not dropped. Every content refusal wraps
`ErrUnsupportedContent`, so a caller can fall back with `errors.Is`.

## License

MIT
