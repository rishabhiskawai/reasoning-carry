# reasoning-carry

Every provider now ships **opaque reasoning blobs** that must round-trip
unmodified on the next turn — or the API 400s:

| provider | blob | breaks when |
|---|---|---|
| gemini 3+ | `thoughtSignature` / `thought_signature` on text/functionCall parts | dropped by JSON clone, openai-compat proxies |
| claude (thinking) | `thinking` / `redacted_thinking` + `signature` | replayed incomplete, fields stripped |
| openai responses | `reasoning` items incl. `encrypted_content` | omitted with `store:false`, reordered |
| deepseek | `reasoning_content` on messages | stripped by "drop unknown fields" |

## The job (v2: attach-at-persist, not validate-after-damage)

A post-hoc validator cannot restore a blob that is already gone. So the
product is capture + append, with the guard optional:

```ts
import { append, fromResponse, assertReplaySafe } from "reasoning-carry";

// after each API call, persist the response verbatim (THE product):
history = append(history, response, "openai");

// or capture + combine manually:
const turns = fromResponse(response, "anthropic"); // understands Responses output[], Chat Completions choices[0].message, candidates[0].content, response.content
history = [...history, ...turns];

// optional guard before sending: throws CarryError instead of a 400
history = assertReplaySafe(history, "gemini");
```

`carry(history, provider)` remains as a deprecated alias of
`assertReplaySafe`.

## Options

```ts
// store:true / previous_response_id: allow id-only reasoning (no blob)
assertReplaySafe(history, "openai", { store: true });

// thinking is on but the history alone can't prove it (fully stripped):
// force the thinking/tool rules instead of evidence-heuristic
assertReplaySafe(history, "anthropic", { thinking: true });
assertReplaySafe(history, "deepseek", { thinking: true }); // default: infer from reasoning_content evidence
```

## Contract

- Pure functions, JSON-in/JSON-out. Never drop, reorder, or mutate opaque
  fields; preserve part/message identity; refuse (throw `CarryError`) on
  shapes they cannot prove safe instead of silently mangling them.
- Per-provider codecs in `src/<provider>.ts`, each exporting
  `carry<Provider>(turns)` + a `supports()` guard.
- Zero runtime dependencies. Node 18+ (structuredClone).
- Tests colocated as `src/*.test.ts` (vitest) + `test/fixtures/` corpus
  (standalone shape records, run through the real dispatcher in
  `src/carry.test.ts`).

## Known limits (not bugs — documented non-checks)

- Fixtures are synthetic payloads modeled on documented failures, NOT
  captured live 200/400 bodies. Do not tag 1.0 until happy-paths are
  replaced with redacted real traffic.
- OpenClaw-style "drop all reasoning, keep messages" passes by design: a
  lone message is indistinguishable client-side from a valid no-reasoning
  reply. The fix is capturing with fromResponse/append, not a smarter
  linter — pass `{ thinking: true }` when you know the config.
- Fully-stripped thinking (no evidence left anywhere) passes without the
  flag, for the same reason.
- Gemini validates the current turn only; image-workflow signature rules
  are not encoded (text without a signature passes).

## Module map

- `src/types.ts` — `Provider`, `CarryError`
- `src/gemini.ts` — thoughtSignature/thought_signature + first-call-per-step
- `src/anthropic.ts` — thinking/redacted_thinking + prefix + tool_use pairing
- `src/openai.ts` — Responses reasoning/encrypted_content/store + DeepSeek reasoning_content
- `src/index.ts` — `assertReplaySafe`/`fromResponse`/`append` + barrel exports
