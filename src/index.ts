import type { Provider } from "./types.js";
import { CarryError } from "./types.js";
import { carryGemini } from "./gemini.js";
import { carryAnthropic, type AnthropicOptions } from "./anthropic.js";
import { carryOpenAI, type OpenAIOptions } from "./openai.js";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Clone with a consistent error surface: structuredClone throws raw
 * DataCloneError on uncloneable input — the public API only ever throws
 * CarryError so agent loops can catch one type.
 */
function safeClone<T>(value: T, what: string): T {
  try {
    return structuredClone(value);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new CarryError(`${what} could not be cloned safely: ${detail}`);
  }
}

export interface CarryOptions {
  /**
   * OpenAI-family only: set true when replaying against a server-persisted
   * conversation (`store: true`). Allows id-only reasoning items without
   * encrypted_content. Ignored by other providers.
   */
  store?: boolean;
  /**
   * Thinking switch (anthropic + deepseek): `true` forces the
   * thinking/tool rules with no history evidence needed (catches fully
   * stripped histories); `false` disables them (thinking off — avoids
   * false positives). Default undefined: infer from history evidence.
   */
  thinking?: boolean;
}

/**
 * Guard: run a stored conversation through one safety pass for `provider`.
 * Opaque reasoning blobs must be byte-identical and ordered; everything else
 * passes through untouched (structuredClone — input is never mutated).
 * Throws CarryError on shapes that cannot be proven safe instead of shipping
 * a turn that will 400. This validates; it cannot restore a blob that was
 * already dropped upstream — capture with fromResponse() at persist time.
 */
export function assertReplaySafe(
  history: unknown[],
  provider: Provider,
  options?: CarryOptions,
): unknown[] {
  if (!Array.isArray(history)) throw new CarryError("history must be an array");
  const turns = safeClone(history, "history");
  switch (provider) {
    case "gemini":
      return carryGemini(turns);
    case "anthropic":
      return carryAnthropic(turns, options as AnthropicOptions | undefined);
    case "openai":
    case "deepseek":
      return carryOpenAI(turns, provider, options as OpenAIOptions | undefined);
    default:
      throw new CarryError(`unknown provider: ${String(provider)}`);
  }
}

/** @deprecated Renamed to assertReplaySafe — same behavior, honest name. */
export const carry = assertReplaySafe;

/**
 * Extract storable turns from a fresh provider response, blobs intact.
 * This is the persist-time half: capture here and you never need the guard
 * to resurrect anything. Returns a NEW array (structuredClone); the response
 * object is never mutated. Throws CarryError on unrecognized shapes.
 *
 * Expected shapes (verbatim sub-objects, per provider docs):
 * - gemini: `{ candidates: [{ content: { role, parts } }] }`
 * - anthropic: `{ content: [...] }` (response content blocks)
 * - openai: `{ output: [...] }` (Responses API) or
 *   `{ choices: [{ message: {...} }] }` (Chat Completions API)
 * - deepseek: `{ choices: [{ message: {...} }] }`
 */
export function fromResponse(response: unknown, provider: Provider): unknown[] {
  if (!isObject(response)) throw new CarryError("response must be an object");
  switch (provider) {
    case "gemini": {
      const content = (response.candidates as unknown[])?.[0] as Record<string, unknown> | undefined;
      const inner = content?.["content"];
      if (!isObject(inner) || !Array.isArray(inner["parts"])) {
        throw new CarryError("gemini response must contain candidates[0].content with parts[]");
      }
      return safeClone([inner], "gemini response content");
    }
    case "anthropic": {
      if (!Array.isArray(response.content)) {
        throw new CarryError("anthropic response must contain content[]");
      }
      return safeClone([{ role: "assistant", content: response.content }], "anthropic response content");
    }
    case "openai": {
      if (Array.isArray(response.output)) {
        return safeClone(response.output, "openai response output");
      }
      const message = (response.choices as unknown[])?.[0] as Record<string, unknown> | undefined;
      const inner = message?.["message"];
      if (isObject(inner)) {
        return safeClone([inner], "openai chat-completions message");
      }
      throw new CarryError(
        "openai response must contain output[] (Responses API) or choices[0].message (Chat Completions API)",
      );
    }
    case "deepseek": {
      const message = (response.choices as unknown[])?.[0] as Record<string, unknown> | undefined;
      const inner = message?.["message"];
      if (!isObject(inner)) {
        throw new CarryError("deepseek response must contain choices[0].message");
      }
      return safeClone([inner], "deepseek response message");
    }
    default:
      throw new CarryError(`unknown provider: ${String(provider)}`);
  }
}

/**
 * Append a fresh provider response to history and validate the COMBINED
 * conversation in one step: `append(history, res, provider)` ===
 * `assertReplaySafe([...history, ...fromResponse(res, provider)], provider)`.
 * Returns a NEW array; neither argument is mutated. Throws CarryError when
 * the combined history cannot be proven safe.
 */
export function append(
  history: unknown[],
  response: unknown,
  provider: Provider,
  options?: CarryOptions,
): unknown[] {
  if (!Array.isArray(history)) throw new CarryError("history must be an array");
  return assertReplaySafe([...history, ...fromResponse(response, provider)], provider, options);
}

export type { Provider } from "./types.js";
export { CarryError } from "./types.js";
export type { OpenAIOptions } from "./openai.js";
export type { AnthropicOptions } from "./anthropic.js";
export { carryGemini } from "./gemini.js";
export { carryAnthropic } from "./anthropic.js";
export { carryOpenAI } from "./openai.js";
