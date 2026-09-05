import { CarryError } from "./types.js";

type JsonObject = Record<string, unknown>;
type HistoryShape = "empty" | "responses" | "chat" | "mixed";

/** One history element with its original position (error messages use it). */
type Indexed = { item: JsonObject; index: number };

const CHAT_ROLES = new Set([
  "system",
  "developer",
  "user",
  "assistant",
  "tool",
  "function",
]);

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOwn(value: JsonObject, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

/**
 * Shape recognition is intentionally shallow: supportsOpenAI answers which
 * codec owns a history, while carryOpenAI performs the safety validation.
 */
/**
 * Per-element recognition. A Responses output item always carries a string
 * `type` (a Responses `message` item has both `type` and `role`, and `type`
 * wins so it validates as an output item); a plain chat message carries
 * `role` and no `type`.
 */
function elementKind(value: unknown): "responses-item" | "chat-message" | null {
  if (!isObject(value)) return null;
  if (typeof value.type === "string" && value.type.length > 0) return "responses-item";
  if (typeof value.role === "string" && value.role.length > 0) return "chat-message";
  return null;
}

function classify(turns: unknown[]): HistoryShape | null {
  if (turns.length === 0) return "empty";

  const kinds = new Set<string>();
  for (const turn of turns) {
    const kind = elementKind(turn);
    if (kind === null) return null;
    kinds.add(kind);
  }
  if (kinds.size === 2) return "mixed";
  return kinds.has("responses-item") ? "responses" : "chat";
}

function jsonError(path: string, detail: string): never {
  throw new CarryError(`history must be JSON-safe: ${detail} at ${path}`);
}

/**
 * Refuse values that structuredClone can copy but JSON cannot faithfully
 * represent. Accessors and non-enumerable/symbol properties are rejected too,
 * because a JSON round trip would silently omit or evaluate them.
 */
function assertJsonSafe(
  value: unknown,
  path: string,
  ancestors: Set<object>,
): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) jsonError(path, "non-finite number");
    return;
  }
  if (typeof value !== "object") jsonError(path, `unsupported ${typeof value}`);

  const object = value as object;
  if (ancestors.has(object)) {
    throw new CarryError(`history must be JSON-safe: cyclic reference at ${path}`);
  }
  if (Object.getOwnPropertySymbols(object).length > 0) {
    jsonError(path, "symbol-keyed property");
  }

  ancestors.add(object);
  try {
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) {
        jsonError(path, "non-standard array prototype");
      }
      const descriptors = Object.getOwnPropertyDescriptors(value);
      for (const [key, descriptor] of Object.entries(descriptors)) {
        if (key === "length") continue;
        const childPath = `${path}.${key}`;
        if (!descriptor.enumerable) jsonError(childPath, "non-enumerable property");
        if (!("value" in descriptor)) jsonError(childPath, "accessor property");
        if (!/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length) {
          jsonError(childPath, "non-index array property");
        }
      }
      for (let index = 0; index < value.length; index += 1) {
        const descriptor = descriptors[String(index)];
        if (descriptor === undefined) {
          jsonError(`${path}[${index}]`, "sparse array slot");
        }
        assertJsonSafe(descriptor.value, `${path}[${index}]`, ancestors);
      }
      return;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      jsonError(path, "non-plain object");
    }

    for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(value))) {
      const childPath = `${path}.${key}`;
      if (!descriptor.enumerable) jsonError(childPath, "non-enumerable property");
      if (!("value" in descriptor)) jsonError(childPath, "accessor property");
      assertJsonSafe(descriptor.value, childPath, ancestors);
    }
  } finally {
    ancestors.delete(object);
  }
}

function validateResponses(entries: Indexed[]): void {
  for (let pos = 0; pos < entries.length; pos += 1) {
    const { item, index } = entries[pos];

    if (item.type === "message") {
      if (item.role !== "assistant") {
        throw new CarryError(
          `OpenAI Responses message at index ${index} must have role assistant`,
        );
      }
      if (!Array.isArray(item.content)) {
        throw new CarryError(
          `OpenAI Responses message at index ${index} must have an array content field`,
        );
      }
    }

    if (item.type !== "reasoning") continue;

    // store:false replay cannot recover a missing server-side reasoning item.
    if (
      !hasOwn(item, "encrypted_content") ||
      typeof item.encrypted_content !== "string" ||
      item.encrypted_content.length === 0
    ) {
      throw new CarryError(
        `OpenAI reasoning item at index ${index} requires non-empty encrypted_content for store:false replay`,
      );
    }

    // Pairing is order within the output-item sequence, not adjacency: a
    // reasoning item must have its message LATER in the sequence. Tool flows
    // (reasoning -> function_call -> output -> message) are the canonical
    // shape; only an orphaned reasoning item with no later message breaks
    // store:false replay. A dangling function_call without its output is
    // preserved verbatim and left to the API — it does not 400.
    const hasLaterMessage = entries
      .slice(pos + 1)
      .some((entry) => entry.item.type === "message");
    if (!hasLaterMessage) {
      throw new CarryError(
        `OpenAI reasoning item at index ${index} must be followed by its message in the output-item sequence`,
      );
    }
  }
}

function validateChat(entries: Indexed[]): void {
  for (const { item: message, index } of entries) {
    const role = message.role as string;
    if (!CHAT_ROLES.has(role)) {
      throw new CarryError(`chat message at index ${index} has unsupported role: ${role}`);
    }

    if (!hasOwn(message, "reasoning_content")) continue;
    if (role !== "assistant") {
      throw new CarryError(
        `reasoning_content at chat message index ${index} is only safe on an assistant message`,
      );
    }
    if (typeof message.reasoning_content !== "string") {
      throw new CarryError(
        `reasoning_content at chat message index ${index} must be a string`,
      );
    }
  }
}

/**
 * Safety rules:
 * - each history element must be a Responses output item or a chat message;
 *   the canonical stateless replay interleaves fresh chat messages with prior
 *   output items, so mixed arrays validate each element by its own kind
 *   (reasoning/message pairing is checked within the output-item sequence);
 * - for OpenAI Responses, require each reasoning item to carry a non-empty
 *   encrypted_content blob and be followed by its message in that sequence;
 * - DeepSeek accepts chat messages only;
 * - preserve every JSON field and array position by structured-cloning only
 *   after validation; otherwise throw CarryError rather than sanitize.
 */
export function carryOpenAI(
  turns: unknown[],
  provider: "openai" | "deepseek",
): unknown[] {
  if (!Array.isArray(turns)) throw new CarryError("history must be an array");
  if (provider !== "openai" && provider !== "deepseek") {
    throw new CarryError(`unsupported OpenAI-family provider: ${String(provider)}`);
  }

  const shape = classify(turns);
  if (shape === null) {
    throw new CarryError(
      "each history element must be a Responses output item or a chat message",
    );
  }

  assertJsonSafe(turns, "history", new Set());

  if (shape === "empty") {
    try {
      return structuredClone(turns);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new CarryError(`history could not be cloned safely: ${detail}`);
    }
  }

  const entries: Indexed[] = (turns as JsonObject[]).map((item, index) => ({ item, index }));
  const responses = entries.filter(({ item }) => elementKind(item) === "responses-item");
  const chat = entries.filter(({ item }) => elementKind(item) === "chat-message");

  if (shape === "responses") {
    if (provider === "deepseek") {
      throw new CarryError("DeepSeek history must use chat messages, not Responses output items");
    }
    validateResponses(responses);
  } else if (shape === "chat") {
    validateChat(chat);
  } else {
    if (provider === "deepseek") {
      throw new CarryError("DeepSeek history must use chat messages, not Responses output items");
    }
    validateResponses(responses);
    validateChat(chat);
  }

  try {
    return structuredClone(turns);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new CarryError(`history could not be cloned safely: ${detail}`);
  }
}

/** True for recognizable Responses output-item arrays or chat-message arrays. */
export function supportsOpenAI(turns: unknown[]): boolean {
  return Array.isArray(turns) && classify(turns) !== null;
}
