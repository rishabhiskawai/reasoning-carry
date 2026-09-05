---
name: Live traffic report
about: Paste a redacted real 200 or 400 body so we can replace synthetic fixtures
title: "[traffic] <provider>: <one-line shape>"
labels: live-traffic
---

## Provider
<!-- gemini / anthropic / openai / deepseek -->

## Result
<!-- 200 (replayed fine) or 400 (rejected — paste the error message too) -->

## How it was produced
<!-- e.g. Responses API store:false, two-turn tool loop; Claude thinking + tool_use; Gemini REST parallel function calls -->

## Redacted body
<!-- Paste the request history you sent. REPLACE all blob values with XXX
     but keep every key, order, and shape intact. Strip tokens, prompts,
     personal data. Example: "thoughtSignature": "XXX" -->
```json
[]
```

## API error (400s only)
```
```

## Checklist
- [ ] All blob values redacted to `XXX` (keys/order kept)
- [ ] No API keys, tokens, emails, or personal data in the body
- [ ] I state the exact API + options used above
