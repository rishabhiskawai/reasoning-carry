import { CarryError } from "./types.js";

// Messages API roles are user|assistant only — a `system` ROLE inside
// messages[] 400s (system prompts belong in the separate `system` param).
const CLAUDE_ROLES = new Set(["user", "assistant"]);
const THINKING_BLOCK_TYPES = new Set(["thinking", "redacted_thinking"]);
const OPENAI_BLOCK_TYPES = new Set(["image_url", "function", "file", "input_text", "output_text"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function fail(message: string): never {
  throw new CarryError(message);
}

function looksLikeClaudeMessage(turn: unknown): boolean {
  if (!isRecord(turn)) return false;
  if (typeof turn.role !== "string" || !CLAUDE_ROLES.has(turn.role)) return false;
  if ("parts" in turn) return false;
  if ("tool_calls" in turn) return false;
  if (!("content" in turn)) return false;
  const content = turn.content;
  if (typeof content === "string") return true;
  if (!Array.isArray(content)) return false;
  for (const block of content) {
    if (!isRecord(block) || typeof block.type !== "string") return false;
    if (OPENAI_BLOCK_TYPES.has(block.type)) return false;
  }
  return true;
}

/**
 * True when `turns` looks like a Claude Messages API `messages[]`:
 * objects with role user|assistant and string or typed content blocks.
 * Empty / Gemini parts[] / OpenAI tool_calls|role:tool|role:system → false.
 */
export function supportsAnthropic(turns: unknown[]): boolean {
  if (!Array.isArray(turns) || turns.length === 0) return false;
  return turns.every(looksLikeClaudeMessage);
}

function cloneTurns(turns: unknown[]): unknown[] {
  try {
    return structuredClone(turns);
  } catch {
    fail("history is not structured-cloneable; cannot prove replay-safe");
  }
}

function assertThinkingPrefix(role: unknown, type: string, seenNonThinking: boolean, at: string) {
  if (role !== "assistant") return;
  if (THINKING_BLOCK_TYPES.has(type) && seenNonThinking) {
    fail(
      `${at}: thinking/redacted_thinking must be a prefix of assistant content (Anthropic replay 400s if they follow text/tool_use)`,
    );
  }
}

/**
 * Safety pass for Claude thinking replay.
 *
 * THROW when we cannot prove the next request will keep complete thinking
 * blobs (signature-stripped thinking, hollow redacted_thinking, thinking after
 * non-thinking, broken tool_use/tool_result ids, non-message turns, system
 * role inside messages[]).
 *
 * Thinking + tool_use: with thinking enabled the API 400s
 * ("expected thinking or redacted_thinking, but found tool_use") when an
 * assistant message carries tool_use without a thinking prefix. The request
 * carries no thinking flag, so the codec uses conversation evidence: once
 * ANY thinking block appears in the history, every assistant message
 * containing tool_use must open with thinking/redacted_thinking. Histories
 * with no thinking anywhere (thinking off) are unaffected.
 *
 * PASS (structuredClone, no drop/reorder/mutate) when every thinking block has
 * a non-empty signature, every redacted_thinking has non-empty `data`, and
 * tool_result ids resolve to earlier tool_use ids. tool_result.content is not
 * inspected. Trailing unmatched tool_use is allowed (in-flight tool turn —
 * with a thinking prefix when thinking is on).
 */
export function carryAnthropic(turns: unknown[]): unknown[] {
  if (!Array.isArray(turns)) fail("anthropic history must be an array");

  const cloned = cloneTurns(turns);

  // Conversation-level evidence: is thinking enabled for this history?
  let thinkingEnabled = false;
  for (const msg of cloned) {
    if (!isRecord(msg) || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (isRecord(block) && typeof block.type === "string" && THINKING_BLOCK_TYPES.has(block.type)) {
        thinkingEnabled = true;
        break;
      }
    }
    if (thinkingEnabled) break;
  }

  const pendingToolUseIds = new Set<string>();

  for (let i = 0; i < cloned.length; i++) {
    const msg = cloned[i];
    if (!isRecord(msg)) fail(`turn ${i} is not an object`);
    if (typeof msg.role !== "string" || !CLAUDE_ROLES.has(msg.role)) {
      fail(`turn ${i} has non-Claude role ${String(msg.role)}`);
    }
    if ("parts" in msg) fail(`turn ${i} looks like Gemini parts[], not Claude content`);

    const content = msg.content;
    if (typeof content === "string") continue;
    if (content == null) fail(`turn ${i} missing content (cannot prove safe)`);
    if (!Array.isArray(content)) fail(`turn ${i} content is neither string nor array`);

    if (
      thinkingEnabled &&
      msg.role === "assistant" &&
      (content as unknown[]).some((b) => isRecord(b) && b.type === "tool_use")
    ) {
      const first = (content as unknown[])[0];
      if (!isRecord(first) || typeof first.type !== "string" || !THINKING_BLOCK_TYPES.has(first.type)) {
        fail(
          `turn ${i} carries tool_use without a thinking/redacted_thinking prefix while thinking is enabled (Anthropic replay 400s: expected thinking, found tool_use)`,
        );
      }
    }

    let seenNonThinking = false;
    for (let j = 0; j < content.length; j++) {
      const block = content[j];
      const at = `turn ${i} block ${j}`;
      if (!isRecord(block)) fail(`${at} is not a content block object`);
      if (typeof block.type !== "string" || block.type.length === 0) {
        fail(`${at} missing type`);
      }
      const type = block.type;

      assertThinkingPrefix(msg.role, type, seenNonThinking, at);

      if (type === "thinking") {
        if (msg.role !== "assistant") fail(`${at}: thinking is only valid on assistant turns`);
        if (typeof block.thinking !== "string") {
          fail(`${at}: thinking block missing thinking text`);
        }
        if (typeof block.signature !== "string" || block.signature.trim().length === 0) {
          fail(
            `${at}: thinking block missing signature — cannot prove replay-safe (signature-stripped replays 400)`,
          );
        }
      } else if (type === "redacted_thinking") {
        if (msg.role !== "assistant") fail(`${at}: redacted_thinking is only valid on assistant turns`);
        if (typeof block.data !== "string" || block.data.length === 0) {
          fail(`${at}: redacted_thinking is not intact (missing data)`);
        }
      } else {
        seenNonThinking = true;
      }

      if (type === "tool_use") {
        if (typeof block.id !== "string" || block.id.length === 0) {
          fail(`${at}: tool_use missing id`);
        }
        if (pendingToolUseIds.has(block.id)) fail(`${at}: duplicate tool_use id ${block.id}`);
        pendingToolUseIds.add(block.id);
      }

      if (type === "tool_result") {
        // Do not walk tool_result.content — pass arrays through untouched.
        if (typeof block.tool_use_id !== "string" || block.tool_use_id.length === 0) {
          fail(`${at}: tool_result missing tool_use_id`);
        }
        if (!pendingToolUseIds.has(block.tool_use_id)) {
          fail(`${at}: tool_result references unknown tool_use_id ${block.tool_use_id}`);
        }
        pendingToolUseIds.delete(block.tool_use_id);
      }
    }
  }

  return cloned;
}
