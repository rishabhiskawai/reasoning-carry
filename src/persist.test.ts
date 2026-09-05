import { describe, expect, it } from "vitest";
import { append, assertReplaySafe, fromResponse } from "./index.js";
import { CarryError } from "./types.js";

const geminiRes = () => ({
  candidates: [{
    content: {
      role: "model",
      parts: [
        { text: "Checking.", thought_signature: "U0lHTkI=" },
        { functionCall: { name: "weather", args: { city: "Paris" } }, thought_signature: "U0lHTkMy" },
      ],
    },
  }],
});

const claudeRes = () => ({
  id: "msg_1",
  content: [
    { type: "thinking", thinking: "Let me check.", signature: "c2ln" },
    { type: "text", text: "Sunny." },
  ],
});

const openaiRes = () => ({
  id: "resp_1",
  output: [
    { type: "reasoning", id: "rs_1", summary: [], encrypted_content: "RU5D" },
    { type: "message", id: "m_1", role: "assistant", status: "completed", content: [] },
  ],
});

const deepseekRes = () => ({
  choices: [{
    message: { role: "assistant", content: "Sunny.", reasoning_content: "Clouds point sun." },
    finish_reason: "stop",
  }],
});

describe("fromResponse", () => {
  it("extracts the gemini content verbatim with signatures", () => {
    const res = geminiRes();
    const out = fromResponse(res, "gemini");
    expect(out).toEqual([res.candidates[0].content]);
    expect(out).not.toBe(res.candidates);
  });

  it("wraps claude content blocks in an assistant message", () => {
    const res = claudeRes();
    expect(fromResponse(res, "anthropic")).toEqual([{ role: "assistant", content: res.content }]);
  });

  it("extracts openai output items verbatim", () => {
    const res = openaiRes();
    expect(fromResponse(res, "openai")).toEqual(res.output);
  });

  it("extracts the deepseek message verbatim", () => {
    const res = deepseekRes();
    expect(fromResponse(res, "deepseek")).toEqual([res.choices[0].message]);
  });

  it("never mutates the response object", () => {
    for (const [res, p] of [
      [geminiRes(), "gemini"],
      [claudeRes(), "anthropic"],
      [openaiRes(), "openai"],
      [deepseekRes(), "deepseek"],
    ] as const) {
      const before = structuredClone(res);
      fromResponse(res, p);
      expect(res).toEqual(before);
    }
  });

  it("throws CarryError on unrecognized shapes", () => {
    expect(() => fromResponse({}, "gemini")).toThrow(CarryError);
    expect(() => fromResponse({ content: "nope" }, "anthropic")).toThrow(CarryError);
    expect(() => fromResponse({ choices: [] }, "openai")).toThrow(CarryError);
    expect(() => fromResponse({}, "deepseek")).toThrow(CarryError);
    expect(() => fromResponse(null, "openai")).toThrow(CarryError);
  });

  it("throws on unknown provider", () => {
    expect(() => fromResponse({}, "cohere" as never)).toThrow(CarryError);
  });
});

describe("append", () => {
  it("appends fresh turns and validates the combined history", () => {
    const history = [{ role: "user", parts: [{ text: "Weather in Paris?" }] }];
    const out = append(history, geminiRes(), "gemini");
    expect(out.length).toBe(2);
    expect(out[1]).toEqual(geminiRes().candidates[0].content);
    // originals untouched
    expect(history.length).toBe(1);
  });

  it("openai tool flow appends across turns", () => {
    const history = [{ role: "user", content: "Weather?" }];
    const out = append(history, openaiRes(), "openai");
    expect(out.length).toBe(3); // user turn + reasoning + message
  });

  it("throws when the combined history is unsafe", () => {
    const bad = { output: [{ type: "reasoning", id: "rs_x", summary: [] }] }; // no encrypted_content
    expect(() => append([], bad, "openai")).toThrow(CarryError);
  });

  it("assertReplaySafe is the guard; carry stays as alias", async () => {
    const mod = await import("./index.js");
    expect(mod.carry).toBe(mod.assertReplaySafe);
    expect(assertReplaySafe([], "openai")).toEqual([]);
  });
});
