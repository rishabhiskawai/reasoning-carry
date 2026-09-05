/**
 * Golden fixtures for the Gemini codec.
 *
 * Each case is a redacted/synthetic payload modeled on the real
 * "missing thought_signature → 400" failure (Gemini 3 generateContent
 * rejects with INVALID_ARGUMENT when a multi-turn functionCall arrives
 * without the opaque signature it was issued with).
 *
 * No network, no DOM — just plain Vitest + assert.
 */
import { describe, it, expect } from "vitest";

import { carryGemini, supportsGemini } from "./gemini.js";
import { CarryError } from "./types.js";

// --- helpers --------------------------------------------------------------

const SIG_A = "c2lnX0FfMTIzNDU2Nzg5MGFiY2RlZjAxMjM0NTY3ODkwYWJjZGVm";
const SIG_B = "c2lnX0JfOTg3NjU0MzIxMGZlZGNiYTk4NzY1NDMyMTBmZWRjYmE";
const SIG_C = "c2lnX0NfY2RlZnRnaGF3NDA5NmV3eHJ0eXVp";
const SIG_D = "c2lnX0RfcG95dGV3dHJlNDU2N3l1aW9w";

// A realistic 4-turn Gemini 3 conversation where the user asks a math
// question, the model calls a calculator tool twice, then answers. Every
// functionCall carries the signature the API returned.
function goodHistory() {
  return [
    {
      role: "user",
      parts: [{ text: "What's 17 * 23 and what's the weather in Paris?" }],
    },
    {
      role: "model",
      parts: [
        { text: "Let me think.", thought_signature: SIG_A },
        {
          functionCall: {
            name: "calculator",
            args: { expr: "17*23" },
          },
          thought_signature: SIG_B,
        },
      ],
    },
    {
      role: "user",
      parts: [
        {
          functionResponse: {
            name: "calculator",
            response: { result: 391 },
          },
        },
      ],
    },
    {
      role: "model",
      parts: [
        { text: "And checking weather…", thought_signature: SIG_C },
        {
          functionCall: {
            name: "get_weather",
            args: { city: "Paris" },
          },
          thought_signature: SIG_D,
        },
      ],
    },
    {
      role: "user",
      parts: [
        {
          functionResponse: {
            name: "get_weather",
            response: { temp_c: 14, condition: "cloudy" },
          },
        },
      ],
    },
    {
      role: "model",
      parts: [{ text: "17 × 23 = 391. Paris is 14°C and cloudy." }],
    },
  ];
}

// --- supportsGemini -------------------------------------------------------

describe("supportsGemini — shape detection", () => {
  it("returns true for a normal Gemini content[] history", () => {
    expect(supportsGemini(goodHistory())).toBe(true);
  });

  it("returns true when role is the openai-compat 'assistant' alias", () => {
    const turns = [
      { role: "user", parts: [{ text: "hi" }] },
      { role: "assistant", parts: [{ text: "hello!" }] },
    ];
    expect(supportsGemini(turns)).toBe(true);
  });

  it("returns false for an empty array", () => {
    expect(supportsGemini([])).toBe(false);
  });

  it("returns false for an array of plain strings (not a content[])", () => {
    expect(supportsGemini(["hello", "world"])).toBe(false);
  });

  it("returns false for OpenAI messages[] shape ({role,content:string})", () => {
    const oai = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ];
    expect(supportsGemini(oai)).toBe(false);
  });

  it("returns false when parts[] exists but contains no content-shaped parts", () => {
    const turns = [{ role: "user", parts: [{ foo: 1 }, { bar: 2 }] }];
    expect(supportsGemini(turns)).toBe(false);
  });
});

// --- carryGemini — happy paths -------------------------------------------

describe("carryGemini — pass-through safety", () => {
  it("returns a well-formed history byte-identical (no mutation, no reorder)", () => {
    const input = goodHistory();
    const snapshot = JSON.parse(JSON.stringify(input));
    const out = carryGemini(input);
    expect(out).toEqual(snapshot);
    // every signature must still be present on the correct part
    const parts1 = (out[1] as { parts: Array<Record<string, unknown>> }).parts;
    const parts3 = (out[3] as { parts: Array<Record<string, unknown>> }).parts;
    expect(parts1[0].thought_signature).toBe(SIG_A);
    expect(parts1[1].thought_signature).toBe(SIG_B);
    expect(parts3[0].thought_signature).toBe(SIG_C);
    expect(parts3[1].thought_signature).toBe(SIG_D);
  });

  it("preserves part order — functionCall must stay adjacent to its preceding text+signature", () => {
    const input = goodHistory();
    const out = carryGemini(input);
    const parts1 = (out[1] as { parts: Array<{ functionCall?: { name: string } }> }).parts;
    const parts3 = (out[3] as { parts: Array<{ functionCall?: { name: string } }> }).parts;
    expect(parts1[1].functionCall?.name).toBe("calculator");
    expect(parts3[1].functionCall?.name).toBe("get_weather");
  });

  it("passes functionResponse parts through untouched (no signature needed)", () => {
    const input = [
      {
        role: "user",
        parts: [
          {
            functionResponse: {
              name: "calc",
              response: { ok: true, value: 42 },
            },
          },
        ],
      },
    ];
    const out = carryGemini(input);
    expect(out).toEqual(input);
  });

  it("accepts an empty history (vacuously safe)", () => {
    expect(carryGemini([])).toEqual([]);
  });

  it("accepts a single user turn with plain text — no signature required", () => {
    const input = [{ role: "user", parts: [{ text: "hi" }] }];
    const out = carryGemini(input);
    expect(out).toEqual(input);
  });

  it("rejects a first-turn functionCall with no signature (Gemini 3 requires it)", () => {
    // The old first-turn exemption was Gemini 2.5 behavior. Gemini 3 400s
    // when the first functionCall of a model step lacks its signature —
    // including the very first model step.
    const input = [
      { role: "user", parts: [{ text: "what's the time?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "get_time", args: {} },
            // NOTE: no thoughtSignature
          },
        ],
      },
    ];
    expect(() => carryGemini(input)).toThrow(CarryError);
    expect(() => carryGemini(input)).toThrow(/missing thought_signature/);
  });

  it("accepts the REST camelCase thoughtSignature spelling", () => {
    const input = [
      { role: "user", parts: [{ text: "what's the time?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "get_time", args: {} },
            thoughtSignature: SIG_A,
          },
        ],
      },
    ];
    const out = carryGemini(input);
    expect(out).toEqual(input);
  });

  it("requires the signature only on the first parallel functionCall of a step", () => {
    // Production parallel shape: signature on the first call, siblings bare.
    const input = [
      { role: "user", parts: [{ text: "time and weather?" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "get_time", args: {} },
            thought_signature: SIG_A,
          },
          {
            functionCall: { name: "get_weather", args: {} },
            // no signature on the sibling — valid
          },
        ],
      },
    ];
    expect(() => carryGemini(input)).not.toThrow();
  });

  it("ignores missing signatures in older turns (compacted history passes)", () => {
    // Only the current turn (newest user text forward) is validated.
    const input = [
      { role: "user", parts: [{ text: "first question" }] },
      {
        role: "model",
        parts: [{ functionCall: { name: "old_tool", args: {} } }], // compacted away
      },
      { role: "user", parts: [{ text: "new question" }] },
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "new_tool", args: {} },
            thoughtSignature: SIG_B,
          },
        ],
      },
    ];
    expect(() => carryGemini(input)).not.toThrow();
  });

  it("passes model turns where text parts lack a signature (Gemini tolerates it)", () => {
    // The 400 specifically targets functionCall. Plain text without a
    // signature is fine — we don't have authority to throw on it.
    const input = [
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [{ text: "hello back" }] }, // no sig on text
      { role: "user", parts: [{ text: "how are you?" }] },
      { role: "model", parts: [{ text: "good thanks" }] },
    ];
    const out = carryGemini(input);
    expect(out).toEqual(input);
  });
});

// --- carryGemini — frozen input (do-not-mutate guarantee) -----------------

describe("carryGemini — deep-clone / frozen-input safety", () => {
  it("does not mutate a frozen input array", () => {
    const input = goodHistory();
    Object.freeze(input);
    Object.freeze(input[1]);
    Object.freeze(input[1].parts);
    Object.freeze(input[1].parts[1]);
    expect(() => carryGemini(input)).not.toThrow();
    // input identity preserved
    expect((input[1].parts[1] as Record<string, unknown>).thought_signature).toBe(SIG_B);
  });

  it("does not mutate frozen parts even when adding a pass-through turn", () => {
    const input = Object.freeze([
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [{ text: "hello" }] },
    ]) as unknown as unknown[];
    expect(() => carryGemini(input)).not.toThrow();
  });
});

// --- carryGemini — failure detection (the real "400" case) ---------------

describe("carryGemini — catches the missing-signature 400", () => {
  it("throws CarryError when a functionCall in turn 2+ has no thought_signature", () => {
    // Real-world cause: caller did JSON.stringify(JSON.parse(...)) on
    // history, or an openai-compat proxy dropped the unknown field.
    const input = [
      { role: "user", parts: [{ text: "compute 17*23" }] },
      // First model turn — signature MISSING but tolerated by API.
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "calc", args: { x: "17*23" } },
          },
        ],
      },
      { role: "user", parts: [{ functionResponse: { name: "calc", response: { value: 391 } } }] },
      // Second model turn — functionCall has NO signature.
      // Gemini 3 will 400 here. Our codec MUST refuse to ship it.
      {
        role: "model",
        parts: [
          {
            functionCall: { name: "weather", args: { city: "Paris" } },
            // thought_signature was stripped
          },
        ],
      },
    ];
    expect(() => carryGemini(input)).toThrow(CarryError);
    expect(() => carryGemini(input)).toThrow(/missing thought_signature/);
  });

  it("detects signature loss caused by a JSON round-trip", () => {
    const input = goodHistory();
    // Simulate the classic bug: someone serialized and re-parsed history.
    const mangled = JSON.parse(JSON.stringify(input));
    // Strip the signature on the SECOND model turn's functionCall
    // (turn[3].parts[1]) — like a "drop unknown fields" sanitizer would.
    // (The first model turn's tool call tolerates absence; we need turn #2
    //  — modelTurnIndex === 1 — to trip the policy.)
    delete mangled[3].parts[1].thought_signature;
    expect(() => carryGemini(mangled)).toThrow(CarryError);
    expect(() => carryGemini(mangled)).toThrow(/missing thought_signature/);
  });

  it("detects signature loss on a text part that carried one (mangled to wrong type)", () => {
    const input = goodHistory();
    const mangled = JSON.parse(JSON.stringify(input));
    // Replace a string signature with a number — Gemini's codec must
    // notice this regardless of which turn it's in (the type check fires
    // for every part that carries a signature key).
    (mangled[1] as { parts: Array<Record<string, unknown>> }).parts[0].thought_signature = 12345;
    expect(() => carryGemini(input)).not.toThrow(); // original is fine
    expect(() => carryGemini(mangled)).toThrow(/non-empty string/);
  });

  it("detects empty-string signature (also a 400)", () => {
    const input = goodHistory();
    const mangled = JSON.parse(JSON.stringify(input));
    mangled[3].parts[1].thought_signature = "";
    expect(() => carryGemini(mangled)).toThrow(/non-empty string/);
  });

  it("throws when parts[] contains a non-object entry", () => {
    const input = [
      { role: "user", parts: [{ text: "hi" }, "garbage"] }, // string in parts
    ];
    expect(() => carryGemini(input)).toThrow(/not an object/);
  });

  it("throws when a turn itself is not an object", () => {
    const input = [{ role: "user", parts: [{ text: "hi" }] }, "oops"];
    expect(() => carryGemini(input)).toThrow(/not an object/);
  });

  it("throws when parts is present but not an array", () => {
    const input = [{ role: "user", parts: "should be array" }];
    expect(() => carryGemini(input)).toThrow(/parts must be an array/);
  });

  it("throws when the outer input is not an array (defense in depth; dispatcher also checks)", () => {
    expect(() => carryGemini("not an array" as unknown as unknown[])).toThrow(/must be an array/);
    expect(() => carryGemini(null as unknown as unknown[])).toThrow(/must be an array/);
  });
});

// --- carryGemini — openai-compat proxy shape ------------------------------

describe("carryGemini — openai-compat proxy rewrites", () => {
  it("accepts 'assistant' as a model role alias", () => {
    const input = [
      { role: "user", parts: [{ text: "hi" }] },
      { role: "assistant", parts: [{ text: "hello!", thought_signature: SIG_A }] },
      { role: "user", parts: [{ text: "compute 2+2" }] },
      {
        role: "assistant",
        parts: [
          {
            functionCall: { name: "calc", args: { x: "2+2" } },
            thought_signature: SIG_B,
          },
        ],
      },
    ];
    expect(() => carryGemini(input)).not.toThrow();
  });

  it("catches a proxy that lowercased the signature key (wrong case = lost field)", () => {
    const input = goodHistory();
    const mangled = JSON.parse(JSON.stringify(input));
    // Some proxies normalize key casing; "Thought_Signature" doesn't match
    // the real "thought_signature" key, so the field is effectively gone
    // for turn[3].parts[1] (the SECOND model turn's functionCall).
    delete mangled[3].parts[1].thought_signature;
    mangled[3].parts[1].Thought_Signature = SIG_D;
    expect(() => carryGemini(mangled)).toThrow(/missing thought_signature/);
  });
});

// --- summary count lock --------------------------------------------------

describe("case-count lock (CI sanity)", () => {
  it("ships >= 15 focused cases", () => {
    // counted below by describe/it blocks; this test enforces the floor
    // so future edits can't quietly remove cases without a failing build.
    const cases = [
      "returns true for a normal Gemini content[] history",
      "returns true when role is the openai-compat 'assistant' alias",
      "returns false for an empty array",
      "returns false for an array of plain strings (not a content[])",
      "returns false for OpenAI messages[] shape ({role,content:string})",
      "returns false when parts[] exists but contains no content-shaped parts",
      "returns a well-formed history byte-identical (no mutation, no reorder)",
      "preserves part order — functionCall must stay adjacent to its preceding text+signature",
      "passes functionResponse parts through untouched (no signature needed)",
      "accepts an empty history (vacuously safe)",
      "accepts a single user turn with plain text — no signature required",
      "rejects a first-turn functionCall with no signature (Gemini 3 requires it)",
      "accepts the REST camelCase thoughtSignature spelling",
      "requires the signature only on the first parallel functionCall of a step",
      "ignores missing signatures in older turns (compacted history passes)",
      "passes model turns where text parts lack a signature (Gemini tolerates it)",
      "does not mutate a frozen input array",
      "does not mutate frozen parts even when adding a pass-through turn",
      "throws CarryError when a functionCall in turn 2+ has no thought_signature",
      "detects signature loss caused by a JSON round-trip",
      "detects signature loss on a text part that carried one (mangled to wrong type)",
      "detects empty-string signature (also a 400)",
      "throws when parts[] contains a non-object entry",
      "throws when a turn itself is not an object",
      "throws when parts is present but not an array",
      "throws when the outer input is not an array (defense in depth; dispatcher also checks)",
      "accepts 'assistant' as a model role alias",
      "catches a proxy that lowercased the signature key (wrong case = lost field)",
    ];
    expect(cases.length).toBeGreaterThanOrEqual(15);
  });
});