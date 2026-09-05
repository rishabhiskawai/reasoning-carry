# reasoning-carry

Carry opaque reasoning blocks across LLM turns without tripping provider 400s.

Every provider now stamps **opaque reasoning blobs** that must round-trip
byte-identical on the next turn:

- **Gemini 3+** — `thought_signature` on text/functionCall parts (missing → 400)
- **Claude thinking** — `thinking`/`redacted_thinking` + `signature` (incomplete replay → 400)
- **OpenAI Responses** — `reasoning` items with `encrypted_content` (omitted with `store:false` → pairing error)
- **DeepSeek** — `reasoning_content` on messages (stripped by "drop unknown fields" passes)

Normal code — `JSON.parse(JSON.stringify())`, proxies, sanitizers — silently
strips these. Everything looks fine, then the next call 400s. This package is
the one-line guard:

```ts
import { carry } from "reasoning-carry";

const safe = await callModel(carry(history, "gemini"));
```

`carry(history, provider)` is pure (JSON-in/JSON-out, never mutates input):
blobs preserved byte-identical and in order, everything else untouched.
Shapes it cannot prove safe throw `CarryError` instead of shipping a turn
that will 400.

Zero runtime dependencies. Node 18+.
