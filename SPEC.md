# reasoning-carry

Every provider now ships **opaque reasoning blobs** that must round-trip
unmodified on the next turn — or the API 400s:

| provider | blob | breaks when |
|---|---|---|
| gemini 3+ | `thought_signature` on text/functionCall parts | dropped by JSON clone, openai-compat proxies |
| claude (thinking) | `thinking` / `redacted_thinking` + `signature` | replayed incomplete, fields stripped |
| openai responses | `reasoning` items incl. `encrypted_content` | omitted with `store:false`, reordered |
| deepseek | `reasoning_content` on messages | stripped by "drop unknown fields" |

```ts
import { carry } from "reasoning-carry";

// history = your stored conversation turns (provider-native shape)
const safe = carry(history, "gemini"); // blobs preserved + ordered, nothing else touched
```

## contract (v1)

- `carry(history, provider)` — pure function, JSON-in/JSON-out. never drops,
  reorders, or mutates the opaque fields above; preserves part/message
  identity; refuses (throws `CarryError`) on shapes it cannot prove safe
  instead of silently mangling them.
- per-provider codecs live in `src/<provider>.ts`, each exporting
  `carry<Provider>(turns: unknown[]): unknown[]` + a `supports()` guard.
- zero runtime dependencies. node 18+ (uses structuredClone).
- tests colocated as `src/*.test.ts` (vitest). golden 400-fixtures:
  real failing payload shapes (redacted) that must survive the round trip.

## module map (team build)

- `src/types.ts` — shared types (`Provider`, `CarryError`) — reviewer-owned
- `src/gemini.ts` — thought_signature + functionCall identity
- `src/anthropic.ts` — thinking/redacted_thinking + signature replay
- `src/openai.ts` — reasoning items, encrypted_content, deepseek reasoning_content
- `src/index.ts` — `carry()` dispatcher + barrel exports — reviewer-owned
