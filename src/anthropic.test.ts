import { describe, expect, it } from "vitest";
import { carryAnthropic, supportsAnthropic } from "./anthropic.js";
import { CarryError } from "./types.js";

const SIG_A = "EqoBCkgIARABGAIqSYNTHETIC_SIGNATURE_aaa=";
const SIG_B = "EqoBCkgIARABGAIqSYNTHETIC_SIGNATURE_bbb=";
const REDACTED = "EmwKAhgBEgySYNTHETIC_REDACTED_THINKING_BLOB";

function thinking(text: string, signature: string) {
  return { type: "thinking" as const, thinking: text, signature };
}

function redacted(data: string) {
  return { type: "redacted_thinking" as const, data };
}

function text(t: string) {
  return { type: "text" as const, text: t };
}

function toolUse(id: string, name: string, input: Record<string, unknown>) {
  return { type: "tool_use" as const, id, name, input };
}

function toolResult(tool_use_id: string, content: unknown) {
  return { type: "tool_result" as const, tool_use_id, content };
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object") {
    Object.freeze(value);
    if (Array.isArray(value)) {
      for (const item of value) deepFreeze(item);
    } else {
      for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
    }
  }
  return value;
}

function expectCarryError(fn: () => unknown) {
  expect(fn).toThrow(CarryError);
}

describe("supportsAnthropic", () => {
  it("returns false for empty history (no Claude evidence)", () => {
    expect(supportsAnthropic([])).toBe(false);
  });

  it("returns true for Claude role/content-block messages", () => {
    const turns = [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [thinking("plan", SIG_A), text("hello")],
      },
    ];
    expect(supportsAnthropic(turns)).toBe(true);
  });

  it("returns true for string-content user/assistant history", () => {
    expect(
      supportsAnthropic([
        { role: "user", content: "q" },
        { role: "assistant", content: "a" },
      ]),
    ).toBe(true);
  });

  it("returns false for Gemini parts[] shape", () => {
    expect(
      supportsAnthropic([
        { role: "user", parts: [{ text: "hi" }] },
        { role: "model", parts: [{ text: "yo" }] },
      ]),
    ).toBe(false);
  });

  it("returns false for OpenAI role:tool / tool_calls shape", () => {
    expect(
      supportsAnthropic([
        { role: "assistant", content: null, tool_calls: [{ id: "c1" }] },
        { role: "tool", tool_call_id: "c1", content: "ok" },
      ]),
    ).toBe(false);
  });
});

describe("carryAnthropic", () => {
  it("returns a new empty array for empty history", () => {
    const input: unknown[] = [];
    const out = carryAnthropic(input);
    expect(out).toEqual([]);
    expect(out).not.toBe(input);
  });

  it("passes string-content turns through without dropping fields", () => {
    const turns = [
      { role: "user", content: "what is 2+2?", meta: 1 },
      { role: "assistant", content: "4" },
    ];
    expect(carryAnthropic(turns)).toEqual(turns);
  });

  it("replays complete thinking blocks including signature (golden)", () => {
    const turns = [
      { role: "user", content: "explain" },
      {
        role: "assistant",
        content: [thinking("I should add 2 and 2.", SIG_A), text("4")],
      },
    ];
    const out = carryAnthropic(turns) as typeof turns;
    const block = (out[1] as { content: unknown[] }).content[0] as {
      type: string;
      thinking: string;
      signature: string;
    };
    expect(block.type).toBe("thinking");
    expect(block.thinking).toBe("I should add 2 and 2.");
    expect(block.signature).toBe(SIG_A);
    expect(out).toEqual(turns);
  });

  it("throws on signature-stripped thinking replay (never silent pass)", () => {
    const stripped = [
      { role: "user", content: "explain" },
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "I should add 2 and 2." }, text("4")],
      },
    ];
    expectCarryError(() => carryAnthropic(stripped));
  });

  it("throws on empty-string signature (stripped-by-proxy shape)", () => {
    const turns = [
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "secret plan", signature: "" }, text("ok")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("throws on whitespace-only signature", () => {
    const turns = [
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "plan", signature: "   " }, text("ok")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("preserves intact redacted_thinking data blobs", () => {
    const turns = [
      {
        role: "assistant",
        content: [redacted(REDACTED), thinking("visible", SIG_A), text("done")],
      },
    ];
    const out = carryAnthropic(turns);
    expect(out).toEqual(turns);
    const first = (out as typeof turns)[0].content[0] as { type: string; data: string };
    expect(first.type).toBe("redacted_thinking");
    expect(first.data).toBe(REDACTED);
  });

  it("throws when redacted_thinking is not intact (missing data)", () => {
    const turns = [
      {
        role: "assistant",
        content: [{ type: "redacted_thinking" }, text("hi")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("throws when redacted_thinking data is emptied (synthetic 400 fixture)", () => {
    const turns = [
      {
        role: "assistant",
        content: [{ type: "redacted_thinking", data: "" }, text("hi")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("survives JSON.parse(JSON.stringify) round-trip without dropping signatures", () => {
    const turns = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [thinking("step", SIG_A), redacted(REDACTED), text("ok")],
      },
    ];
    const cloned = JSON.parse(JSON.stringify(turns)) as unknown[];
    const out = carryAnthropic(cloned);
    expect(out).toEqual(turns);
    const blocks = (out as typeof turns)[1].content;
    expect(blocks[0]).toMatchObject({ type: "thinking", signature: SIG_A });
    expect(blocks[1]).toMatchObject({ type: "redacted_thinking", data: REDACTED });
  });

  it("does not mutate frozen input", () => {
    const turns = deepFreeze([
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [thinking("step", SIG_A), text("ok")],
      },
    ]);
    const snapshot = JSON.stringify(turns);
    const out = carryAnthropic(turns);
    expect(JSON.stringify(turns)).toBe(snapshot);
    expect(out).toEqual(turns);
    expect(out).not.toBe(turns);
    (out[1] as { content: unknown[] }).content.push({ type: "text", text: "mut" });
    expect((turns[1] as { content: unknown[] }).content).toHaveLength(2);
  });

  it("preserves mixed thinking/text/tool_use block order (does not reorder)", () => {
    const turns = [
      { role: "user", content: "weather in SF?" },
      {
        role: "assistant",
        content: [
          thinking("need a tool", SIG_A),
          text("checking"),
          toolUse("toolu_01", "get_weather", { city: "SF" }),
        ],
      },
      {
        role: "user",
        content: [toolResult("toolu_01", "72F and sunny")],
      },
    ];
    const out = carryAnthropic(turns) as typeof turns;
    const types = (out[1].content as { type: string }[]).map((b) => b.type);
    expect(types).toEqual(["thinking", "text", "tool_use"]);
    expect(out).toEqual(turns);
  });

  it("throws if thinking appears after non-thinking (replay would 400)", () => {
    const turns = [
      {
        role: "assistant",
        content: [text("oops"), thinking("late", SIG_A)],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("passes tool_result content arrays through untouched (no nested walk)", () => {
    const nested = [
      { type: "thinking", thinking: "should not be validated" },
      { type: "text", text: "screenshot said 72" },
    ];
    const turns = [
      {
        role: "assistant",
        content: [thinking("use tool", SIG_A), toolUse("toolu_99", "look", {})],
      },
      {
        role: "user",
        content: [toolResult("toolu_99", nested)],
      },
    ];
    const out = carryAnthropic(turns) as typeof turns;
    const result = out[1].content[0] as { content: unknown };
    expect(result.content).toEqual(nested);
    expect(result.content).not.toBe(nested);
  });

  it("throws when tool_result has no matching tool_use (broken pairing)", () => {
    const turns = [
      { role: "user", content: [toolResult("toolu_orphan", "nope")] },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("throws when tool_result pairing order consumes an unknown id", () => {
    const turns = [
      {
        role: "assistant",
        content: [thinking("t", SIG_A), toolUse("toolu_a", "x", {})],
      },
      {
        role: "user",
        content: [toolResult("toolu_b", "mismatch")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("allows unmatched trailing tool_use (last assistant turn)", () => {
    const turns = [
      {
        role: "assistant",
        content: [thinking("call", SIG_A), toolUse("toolu_live", "x", { n: 1 })],
      },
    ];
    expect(carryAnthropic(turns)).toEqual(turns);
  });

  it("keeps two tool_use ids paired to later results without reordering", () => {
    const turns = [
      {
        role: "assistant",
        content: [
          thinking("both", SIG_A),
          toolUse("toolu_1", "a", {}),
          toolUse("toolu_2", "b", {}),
        ],
      },
      {
        role: "user",
        content: [toolResult("toolu_1", "A"), toolResult("toolu_2", "B")],
      },
      {
        role: "assistant",
        content: [thinking("done", SIG_B), text("ok")],
      },
    ];
    const out = carryAnthropic(turns) as typeof turns;
    const ids = (out[1].content as { tool_use_id: string }[]).map((b) => b.tool_use_id);
    expect(ids).toEqual(["toolu_1", "toolu_2"]);
    expect(out).toEqual(turns);
  });

  it("throws on non-array history", () => {
    expectCarryError(() => carryAnthropic({} as unknown as unknown[]));
  });

  it("throws on thinking block missing thinking text", () => {
    const turns = [
      {
        role: "assistant",
        content: [{ type: "thinking", signature: SIG_A }, text("x")],
      },
    ];
    expectCarryError(() => carryAnthropic(turns));
  });

  it("preserves extra fields on thinking blocks (cache_control, etc.)", () => {
    const turns = [
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: "plan",
            signature: SIG_A,
            cache_control: { type: "ephemeral" },
          },
          text("ok"),
        ],
      },
    ];
    expect(carryAnthropic(turns)).toEqual(turns);
  });
});
