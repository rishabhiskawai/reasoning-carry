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

/**
 * Roles a Responses `type: "message"` item may carry. The canonical
 * stateless replay mixes fresh user/system/developer input messages with
 * prior assistant output items — restricting to assistant rejects valid
 * production payloads.
 */
const MESSAGE_ROLES = new Set(["assistant", "user", "system", "developer"]);

/** Responses item types whose id/status fields we type-check (nulls 400). */
const TYPED_ITEMS = new Set(["reasoning", "message", "function_call", "function_call_output"]);

export interface OpenAIOptions {
  /**
   * Set true when replaying against a server-persisted conversation
   * (`store: true` / previous_response_id): reasoning items may then be
   * id-only, without encrypted_content. Default false — stateless replay
   * must preserve the blob or the next call 400s.
   */
  store?: boolean;
  /**
   * DeepSeek thinking switch. `true` forces the tool_calls ⇒
   * reasoning_content rule; `false` disables it (tools on, thinking off —
   * otherwise a false positive). Default undefined: infer from history —
   * enforced only when some message already carries reasoning_content.
   */
  thinking?: boolean;
}

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

/** Null SDK fields (e.g. `status: null` from model_dump) are JSON-safe but the API 400s on them. */
function assertStringField(item: JsonObject, key: string, index: number, what: string): void {
  if (hasOwn(item, key) && typeof item[key] !== "string") {
    throw new CarryError(
      `OpenAI ${what} at index ${index} has non-string ${key} (null SDK fields 400 — drop or restore the value)`,
    );
  }
}

function validateResponses(entries: Indexed[], options?: OpenAIOptions): void {
  for (let pos = 0; pos < entries.length; pos += 1) {
    const { item, index } = entries[pos];

    if (typeof item.type === "string" && TYPED_ITEMS.has(item.type)) {
      assertStringField(item, "id", index, item.type);
      assertStringField(item, "status", index, item.type);
    }

    if (item.type === "message") {
      // Responses input mixes fresh user/system/developer messages with
      // prior assistant output items — all four roles are valid here.
      if (typeof item.role !== "string" || !MESSAGE_ROLES.has(item.role)) {
        throw new CarryError(
          `OpenAI Responses message at index ${index} has unsupported role: ${String(item.role)}`,
        );
      }
      if (!Array.isArray(item.content)) {
        throw new CarryError(
          `OpenAI Responses message at index ${index} must have an array content field`,
        );
      }
    }

    if (item.type !== "reasoning") continue;

    // Stateless replay cannot recover a missing server-side reasoning blob.
    // store:true conversations may replay id-only reasoning items instead.
    if (hasOwn(item, "encrypted_content")) {
      if (typeof item.encrypted_content !== "string" || item.encrypted_content.length === 0) {
        throw new CarryError(
          `OpenAI reasoning item at index ${index} requires non-empty encrypted_content for store:false replay`,
        );
      }
      continue;
    }
    if (!(options?.store === true && typeof item.id === "string" && item.id.length > 0)) {
      throw new CarryError(
        `OpenAI reasoning item at index ${index} requires non-empty encrypted_content for store:false replay (pass { store: true } for server-persisted id-only replay)`,
      );
    }

    // NOTE (deliberate non-check): an in-flight tool turn — reasoning +
    // function_call with no trailing message yet — is VALID agent-loop
    // state, so no later message is required. The inverse failure (a message
    // that lost its preceding reasoning after message-only filtering) is
    // indistinguishable client-side from a valid no-reasoning reply, so we
    // do not guess: order is preserved verbatim and the API judges the rest.
  }
}

function hasToolCalls(message: JsonObject): boolean {
  return Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
}

function validateChat(
  entries: Indexed[],
  provider: "openai" | "deepseek",
  options?: OpenAIOptions,
): void {
  // DeepSeek thinking evidence: the tool rule is enforced when the caller
  // declares thinking on, or (default) when the history itself shows
  // thinking was used. Without either, tools-without-reasoning passes —
  // thinking may simply be off.
  let thinkingEvidence = options?.thinking;
  if (thinkingEvidence === undefined && provider === "deepseek") {
    thinkingEvidence = entries.some(
      ({ item }) =>
        hasOwn(item, "reasoning_content") && typeof item.reasoning_content === "string" && item.reasoning_content.length > 0,
    );
  }
  const enforceToolReasoning = provider === "deepseek" && thinkingEvidence === true;

  for (const { item: message, index } of entries) {
    const role = message.role as string;
    if (!CHAT_ROLES.has(role)) {
      throw new CarryError(`chat message at index ${index} has unsupported role: ${role}`);
    }

    if (!hasOwn(message, "reasoning_content")) {
      // DeepSeek thinking + tools: an assistant message that issued a tool
      // call must send reasoning_content back on every later request. The
      // production failure is ABSENCE — a rebuilt message with only role,
      // content and tool_calls — so absence with tool_calls throws here
      // whenever thinking is on (declared or evidenced).
      if (enforceToolReasoning && role === "assistant" && hasToolCalls(message)) {
        throw new CarryError(
          `DeepSeek assistant message at index ${index} issued tool_calls without reasoning_content (replay 400s — preserve choices[0].message verbatim)`,
        );
      }
      continue;
    }
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
    if (enforceToolReasoning && hasToolCalls(message) && message.reasoning_content.length === 0) {
      throw new CarryError(
        `DeepSeek assistant message at index ${index} issued tool_calls with empty reasoning_content (replay 400s)`,
      );
    }
  }
}

/**
 * Safety rules:
 * - each history element must be a Responses output item or a chat message;
 *   the canonical stateless replay interleaves fresh chat messages with prior
 *   output items, so mixed arrays validate each element by its own kind;
 * - for OpenAI Responses, each reasoning item must carry a non-empty
 *   encrypted_content blob (stateless default) or — with { store: true } —
 *   a server-persisted id. In-flight tool turns (reasoning + function_call,
 *   no trailing message yet) are valid and pass;
 * - DeepSeek accepts chat messages only; an assistant message with
 *   tool_calls must carry non-empty reasoning_content;
 * - null id/status fields (model_dump artifacts) throw even though they are
 *   JSON-safe, because the API 400s on them;
 * - preserve every JSON field and array position by structured-cloning only
 *   after validation; otherwise throw CarryError rather than sanitize.
 */
export function carryOpenAI(
  turns: unknown[],
  provider: "openai" | "deepseek",
  options?: OpenAIOptions,
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
    validateResponses(responses, options);
  } else if (shape === "chat") {
    validateChat(chat, provider, options);
  } else {
    if (provider === "deepseek") {
      throw new CarryError("DeepSeek history must use chat messages, not Responses output items");
    }
    validateResponses(responses, options);
    validateChat(chat, provider, options);
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
