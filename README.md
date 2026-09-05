# reasoning-carry

Carry opaque reasoning blocks across LLM turns without tripping provider 400s.

Every provider now stamps **opaque reasoning blobs** that must round-trip
byte-identical on the next call — Gemini 3 `thoughtSignature`, Claude
`thinking.signature`, OpenAI Responses `reasoning.encrypted_content`,
DeepSeek `reasoning_content`. Drop one (rebuild history, strip unknown
fields, `model_dump()` nulls, openai-compat proxies) and the next call 400s.

## The rule

**Never rebuild history. Append the provider's response object unchanged.**

```ts
import { append, fromResponse, assertReplaySafe } from "reasoning-carry";

// after each API call, persist the response verbatim:
history = append(history, response, "openai");

// or capture + combine manually:
const turns = fromResponse(response, "anthropic");
history = [...history, ...turns];

// optional guard before sending: throws CarryError instead of a 400
history = assertReplaySafe(history, "gemini");
```

`fromResponse` understands the documented shapes (`candidates[0].content`,
`response.content`, `response.output`, `choices[0].message`) and clones —
your live response object is never mutated. `assertReplaySafe` is the
optional linter, not the product: it cannot restore a blob that was already
dropped upstream.

## Options

```ts
// store:true / previous_response_id conversations may replay id-only
// reasoning items (no encrypted_content). Default is stateless (strict).
assertReplaySafe(history, "openai", { store: true });

// thinking is on but the history alone can't prove it (fully stripped
// blobs leave no evidence): force the thinking/tool rules. false disables
// them (tools on, thinking off — avoids false positives). Default infers
// from history evidence.
assertReplaySafe(history, "anthropic", { thinking: true });
assertReplaySafe(history, "deepseek", { thinking: false });
```

## Limits (honest)

- Rules are validated against documented API behavior and synthetic
  fixtures, not captured live 200/400 traffic — real round-trip bodies are
  the next milestone before any 1.0.
- Gemini validates the current turn only (older compacted turns pass);
  image-workflow signature rules are not encoded.
- A `type: "message"` that lost its preceding reasoning after message-only
  filtering is indistinguishable from a valid no-reasoning reply, so the
  guard preserves order and lets the API judge it.

Zero runtime dependencies. ESM + CJS + types.
