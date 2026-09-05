import { describe, expect, it } from "vitest";
import { carryOpenAI, supportsOpenAI } from "./openai.js";
import { CarryError } from "./types.js";

const responsePair = () => [
  {
    id: "rs_redacted_001",
    type: "reasoning",
    summary: [{ type: "summary_text", text: "redacted" }],
    encrypted_content: "gAAAAA_synthetic_ciphertext",
  },
  {
    id: "msg_redacted_001",
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text: "Synthetic answer." }],
  },
];

function freezePair(): unknown[] {
  const pair = responsePair();
  Object.freeze((pair[0] as { summary: unknown[] }).summary[0]);
  Object.freeze((pair[0] as { summary: unknown[] }).summary);
  Object.freeze(pair[0]);
  Object.freeze((pair[1] as { content: unknown[] }).content[0]);
  Object.freeze((pair[1] as { content: unknown[] }).content);
  Object.freeze(pair[1]);
  return Object.freeze(pair) as unknown as unknown[];
}

describe("carryOpenAI Responses API", () => {
  it("preserves a frozen reasoning/message pair including encrypted_content", () => {
    const turns = freezePair();

    const carried = carryOpenAI(turns, "openai");

    expect(carried).toEqual(turns);
    expect(carried).not.toBe(turns);
    expect((carried[0] as Record<string, unknown>).encrypted_content).toBe(
      "gAAAAA_synthetic_ciphertext",
    );
    expect(Object.isFrozen(turns[0])).toBe(true);
  });

  it("preserves the order of multiple reasoning/message pairs", () => {
    const first = responsePair();
    const second = responsePair();
    (second[0] as { id: string }).id = "rs_redacted_002";
    (second[0] as { encrypted_content: string }).encrypted_content = "ciphertext_002";
    (second[1] as { id: string }).id = "msg_redacted_002";
    const turns = [...first, ...second];

    const carried = carryOpenAI(turns, "openai") as Array<{ id: string }>;

    expect(carried.map((item) => item.id)).toEqual([
      "rs_redacted_001",
      "msg_redacted_001",
      "rs_redacted_002",
      "msg_redacted_002",
    ]);
  });

  it("passes through unknown typed output items without reordering fields", () => {
    const turns = [
      { type: "web_search_call", id: "ws_1", status: "completed", vendor_blob: { z: 1 } },
    ];

    expect(carryOpenAI(turns, "openai")).toEqual(turns);
  });

  it("rejects a reasoning item with omitted encrypted_content", () => {
    const turns = responsePair();
    delete (turns[0] as { encrypted_content?: string }).encrypted_content;

    expect(() => carryOpenAI(turns, "openai")).toThrowError(CarryError);
    expect(() => carryOpenAI(turns, "openai")).toThrow(/encrypted_content/);
  });

  it("rejects null encrypted_content", () => {
    const turns = responsePair();
    (turns[0] as { encrypted_content: unknown }).encrypted_content = null;

    expect(() => carryOpenAI(turns, "openai")).toThrow(/encrypted_content/);
  });

  it("rejects empty encrypted_content", () => {
    const turns = responsePair();
    (turns[0] as { encrypted_content: string }).encrypted_content = "";

    expect(() => carryOpenAI(turns, "openai")).toThrow(/encrypted_content/);
  });

  it("accepts a trailing reasoning item with no message yet (in-flight tool turn)", () => {
    // Turn 2 of every agent loop: reasoning emitted, assistant message not
    // yet produced. The old pairing rule rejected this valid state — the
    // uninstall-grade false positive.
    const [reasoning] = responsePair();
    expect(() => carryOpenAI([reasoning], "openai")).not.toThrow();
    expect(carryOpenAI([reasoning], "openai")).toEqual([reasoning]);
  });

  it("accepts an in-flight reasoning + function_call turn without any message", () => {
    const [reasoning] = responsePair();
    const turns = [
      reasoning,
      { type: "function_call", id: "fc_1", call_id: "call_1", name: "lookup", arguments: "{}" },
      { type: "function_call_output", call_id: "call_1", output: "{}" },
    ];

    expect(() => carryOpenAI(turns, "openai")).not.toThrow();
  });

  it("preserves tool-flow items between a reasoning item and its message", () => {
    // Canonical completed replay: reasoning -> function_call -> output -> message.
    const [reasoning, message] = responsePair();
    const turns = [reasoning, { type: "function_call", id: "fc_1", name: "lookup" }, message];

    expect(() => carryOpenAI(turns, "openai")).not.toThrow();
  });

  it("accepts message-before-reasoning order (next turn already started)", () => {
    const [reasoning, message] = responsePair();

    expect(() => carryOpenAI([message, reasoning], "openai")).not.toThrow();
  });

  it("accepts user-role input messages mixed with prior output items", () => {
    // Canonical stateless replay: fresh user input + prior assistant output.
    const turns = responsePair();
    (turns[1] as { role: string }).role = "user";

    expect(() => carryOpenAI(turns, "openai")).not.toThrow();
  });

  it("rejects a response message with a role outside the Responses set", () => {
    const turns = responsePair();
    (turns[1] as { role: string }).role = "tool";

    expect(() => carryOpenAI(turns, "openai")).toThrow(/unsupported role/);
  });

  it("rejects null id/status fields from model_dump artifacts", () => {
    const withNullId = responsePair();
    (withNullId[0] as { id: unknown }).id = null;
    expect(() => carryOpenAI(withNullId, "openai")).toThrow(/non-string id/);

    const withNullStatus = responsePair();
    (withNullStatus[1] as { status: unknown }).status = null;
    expect(() => carryOpenAI(withNullStatus, "openai")).toThrow(/non-string status/);
  });

  it("allows id-only reasoning replay with { store: true }", () => {
    const turns = responsePair();
    delete (turns[0] as { encrypted_content?: string }).encrypted_content;

    expect(() => carryOpenAI(turns, "openai")).toThrow(/encrypted_content/);
    expect(carryOpenAI(turns, "openai", { store: true })).toEqual(turns);
  });

  it("still rejects id-only reasoning with { store: true } when no id exists", () => {
    const turns = responsePair();
    delete (turns[0] as { encrypted_content?: string }).encrypted_content;
    delete (turns[0] as { id?: string }).id;

    expect(() => carryOpenAI(turns, "openai", { store: true })).toThrow(/encrypted_content/);
  });

  it("returns a deep clone rather than aliases nested input", () => {
    const turns = responsePair();
    const carried = carryOpenAI(turns, "openai") as Array<Record<string, unknown>>;
    ((carried[0].summary as Array<Record<string, unknown>>)[0]).text = "changed";

    expect(((turns[0] as { summary: Array<{ text: string }> }).summary[0]).text).toBe("redacted");
  });
});

describe("carryOpenAI chat and DeepSeek messages", () => {
  it("preserves OpenAI chat reasoning_content when present", () => {
    const turns = [
      { role: "user", content: "Synthetic prompt" },
      { role: "assistant", reasoning_content: "private chain redacted", content: "Answer" },
    ];

    expect(carryOpenAI(turns, "openai")).toEqual(turns);
  });

  it("keeps DeepSeek reasoning_content that a drop-unknown-fields pass would strip", () => {
    const failingFixture = [
      { role: "user", content: "2 + 2?" },
      {
        role: "assistant",
        content: "4",
        reasoning_content: "Synthetic hidden reasoning, not a real trace.",
      },
    ];

    const carried = carryOpenAI(failingFixture, "deepseek") as Array<Record<string, unknown>>;

    expect(carried[1]).toHaveProperty(
      "reasoning_content",
      "Synthetic hidden reasoning, not a real trace.",
    );
  });

  it("preserves an explicitly empty reasoning_content string", () => {
    const turns = [{ role: "assistant", content: "Done", reasoning_content: "" }];

    expect(carryOpenAI(turns, "deepseek")).toEqual(turns);
    expect(carryOpenAI(turns, "deepseek")[0]).toHaveProperty("reasoning_content", "");
  });

  it("preserves unknown message fields recursively", () => {
    const turns = [
      {
        role: "assistant",
        content: "Answer",
        reasoning_content: "redacted",
        provider_extension: { trace: ["a", "b"], enabled: true },
      },
    ];

    expect(carryOpenAI(turns, "deepseek")).toEqual(turns);
  });

  it("passes ordinary chat history without reasoning fields", () => {
    const turns = [
      { role: "system", content: "Be concise" },
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
      { role: "tool", tool_call_id: "call_1", content: "ok" },
    ];

    expect(carryOpenAI(turns, "deepseek")).toEqual(turns);
  });

  it("accepts frozen DeepSeek messages without mutating them", () => {
    const assistant = Object.freeze({
      role: "assistant",
      content: "Answer",
      reasoning_content: "redacted",
    });
    const turns = Object.freeze([assistant]) as unknown as unknown[];

    const carried = carryOpenAI(turns, "deepseek");

    expect(carried).toEqual(turns);
    expect(carried).not.toBe(turns);
    expect(Object.isFrozen(turns[0])).toBe(true);
  });

  it("rejects non-string reasoning_content", () => {
    const turns = [{ role: "assistant", content: "Answer", reasoning_content: { lost: true } }];

    expect(() => carryOpenAI(turns, "deepseek")).toThrow(/reasoning_content.*string/);
  });

  it("rejects reasoning_content attached to a non-assistant message", () => {
    const turns = [{ role: "user", content: "Prompt", reasoning_content: "misattached" }];

    expect(() => carryOpenAI(turns, "deepseek")).toThrow(/reasoning_content.*assistant/);
  });

  it("rejects DeepSeek tool_calls without reasoning_content (the documented 400)", () => {
    const turns = [
      {
        role: "assistant",
        content: "Checking.",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "weather", arguments: "{}" } },
        ],
      },
    ];

    expect(() => carryOpenAI(turns, "deepseek")).toThrow(/reasoning_content/);
  });

  it("accepts DeepSeek tool_calls with reasoning_content preserved", () => {
    const turns = [
      {
        role: "assistant",
        content: "Checking.",
        reasoning_content: "Synthetic hidden reasoning.",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "weather", arguments: "{}" } },
        ],
      },
    ];

    expect(carryOpenAI(turns, "deepseek")).toEqual(turns);
  });

  it("rejects DeepSeek tool_calls with empty reasoning_content", () => {
    const turns = [
      {
        role: "assistant",
        content: "Checking.",
        reasoning_content: "",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "weather", arguments: "{}" } },
        ],
      },
    ];

    expect(() => carryOpenAI(turns, "deepseek")).toThrow(/reasoning_content/);
  });

  it("rejects Responses output-item arrays for the DeepSeek provider", () => {
    expect(() => carryOpenAI(responsePair(), "deepseek")).toThrow(/DeepSeek.*chat/i);
  });

  it("accepts the canonical mixed replay: fresh chat messages plus prior output items", () => {
    const turns = [
      { role: "user", content: "Prompt" },
      { type: "reasoning", id: "rs_1", encrypted_content: "QUJD", summary: [] },
      { type: "message", id: "m_1", role: "assistant", status: "completed", content: [] },
    ];

    expect(() => carryOpenAI(turns, "openai")).not.toThrow();
    expect(carryOpenAI(structuredClone(turns), "openai")).toEqual(turns);
  });

  it("accepts mixed arrays whose reasoning has no later message yet (in-flight)", () => {
    const turns = [
      { role: "user", content: "Prompt" },
      { type: "reasoning", id: "rs_1", encrypted_content: "QUJD", summary: [] },
    ];

    expect(() => carryOpenAI(turns, "openai")).not.toThrow();
    expect(carryOpenAI(structuredClone(turns), "openai")).toEqual(turns);
  });
});

describe("JSON safety and empty histories", () => {
  it("accepts an empty history for either routed provider", () => {
    expect(carryOpenAI([], "openai")).toEqual([]);
    expect(carryOpenAI([], "deepseek")).toEqual([]);
  });

  it("wraps non-JSON function fields in CarryError", () => {
    const turns = [{ role: "assistant", content: "Answer", extension: () => "unsafe" }];

    expect(() => carryOpenAI(turns, "openai")).toThrowError(CarryError);
    expect(() => carryOpenAI(turns, "openai")).toThrow(/JSON-safe/);
  });

  it("rejects non-enumerable array fields that JSON would silently drop", () => {
    const content = ["Answer"];
    Object.defineProperty(content, "hidden", { value: "opaque", enumerable: false });
    const turns = [{ role: "assistant", content }];

    expect(() => carryOpenAI(turns, "openai")).toThrow(/non-enumerable/);
  });

  it("rejects accessor array slots without invoking their getter", () => {
    let invoked = false;
    const content: unknown[] = [];
    Object.defineProperty(content, "0", {
      enumerable: true,
      configurable: true,
      get() {
        invoked = true;
        return "Answer";
      },
    });
    content.length = 1;
    const turns = [{ role: "assistant", content }];

    expect(() => carryOpenAI(turns, "openai")).toThrow(/accessor/);
    expect(invoked).toBe(false);
  });

  it("rejects cyclic histories with CarryError", () => {
    const message: Record<string, unknown> = { role: "assistant", content: "Answer" };
    message.self = message;

    expect(() => carryOpenAI([message], "openai")).toThrowError(CarryError);
    expect(() => carryOpenAI([message], "openai")).toThrow(/cyclic/);
  });
});

describe("supportsOpenAI", () => {
  it("recognizes Responses output-item arrays", () => {
    expect(supportsOpenAI(responsePair())).toBe(true);
  });

  it("recognizes a Responses shape even when safety validation will reject its missing blob", () => {
    const turns = responsePair();
    delete (turns[0] as { encrypted_content?: string }).encrypted_content;

    expect(supportsOpenAI(turns)).toBe(true);
  });

  it("recognizes DeepSeek chat messages with reasoning_content", () => {
    expect(
      supportsOpenAI([{ role: "assistant", content: "Answer", reasoning_content: "redacted" }]),
    ).toBe(true);
  });

  it("recognizes ordinary chat-message arrays", () => {
    expect(supportsOpenAI([{ role: "user", content: "Hello" }])).toBe(true);
  });

  it("treats empty history as a supported identity case", () => {
    expect(supportsOpenAI([])).toBe(true);
  });

  it("supports mixed replay arrays but rejects malformed ones", () => {
    expect(supportsOpenAI([{ role: "user" }, { type: "function_call" }])).toBe(true);
    expect(supportsOpenAI([null])).toBe(false);
  });
});
