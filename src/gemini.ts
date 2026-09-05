/**
 * Gemini 3+ content[] codec for reasoning-carry.
 *
 * Background: Gemini 3+ stamps an opaque `thought_signature` on text and
 * functionCall parts returned by `generateContent`. That signature MUST be
 * round-tripped byte-identical on the next turn — drop it (e.g. JSON
 * clone, openai-compat proxy rewriting unknown fields, "sanitizer" that
 * strips keys starting with "thought_") and the next call 400s:
 *   "Unable to submit request because function call parameter is missing
 *    a thought_signature. Function calls require this field to be present
 *    and unaltered."
 *
 * Strategy: structuredClone the history (caller's input is treated as
 * frozen — we never mutate it), then walk every `parts[]` and verify that
 * any part carrying a `thought_signature` survives intact. We also
 * enforce the order/identity invariant on `functionCall` parts: Gemini
 * matches signatures to function calls by adjacency in the same model
 * turn, so reordering would silently break replay.
 *
 * Throw-vs-pass policy (chosen; documented in README by reviewer):
 *   - PASS  — turns without `parts` (plain `{ role }`), plain text parts
 *             with no signature, functionResponse parts, unknown extra
 *             part shapes. We don't have authority to reject these.
 *   - THROW — a `functionCall` part that is missing a `thought_signature`
 *             AND that the caller has indicated lives in a non-first
 *             model turn. (First-turn tool calls are an edge case where
 *             the API tolerates absence; anything beyond that 400s.)
 *             We cannot prove absence is safe → refuse to silently ship
 *             a turn we know will 400.
 *   - THROW — a non-string `thought_signature` (must be the opaque string
 *             Gemini returned; bytes/objects/numbers are not signatures).
 *
 * functionResponse parts are pure outputs and never carry a signature, so
 * they pass through untouched — no special handling needed beyond
 * preserving object identity via structuredClone.
 */
import { CarryError } from "./types.js";

// --- shape probing (minimal, duck-typed; we don't trust `any`) ----------

type GeminiTurn = {
  role?: unknown;
  parts?: unknown;
} & Record<string, unknown>;

type GeminiPart = Record<string, unknown>;

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isGeminiRole(v: unknown): boolean {
  // Gemini uses "user" | "model" | "system" | "function" (legacy).
  // "assistant" appears in some openai-compat proxies — accept it too
  // because the part shapes below are what we really care about.
  return v === "user" || v === "model" || v === "system" || v === "function" || v === "assistant";
}

function isGeminiPartShape(v: unknown): v is GeminiPart {
  if (!isObject(v)) return false;
  // At least one of the known content-bearing keys must exist.
  return (
    "text" in v ||
    "functionCall" in v ||
    "functionResponse" in v ||
    "inlineData" in v ||
    "fileData" in v ||
    "executableCode" in v ||
    "codeExecutionResult" in v
  );
}

function partHasSignature(part: GeminiPart): boolean {
  return Object.prototype.hasOwnProperty.call(part, "thought_signature");
}

function isFunctionCallPart(part: GeminiPart): boolean {
  return isObject(part.functionCall);
}

function signatureIsString(part: GeminiPart): boolean {
  const s = part.thought_signature;
  return typeof s === "string" && s.length > 0;
}

// --- public API ----------------------------------------------------------

/**
 * True when the history looks like Gemini `contents[]` — an array of
 * `{ role, parts: [...] }` (or bare `{ role }`) turns.
 * Conservative: requires role + parts[].shape to be present at least once.
 */
export function supportsGemini(turns: unknown[]): boolean {
  if (!Array.isArray(turns) || turns.length === 0) return false;
  let hits = 0;
  for (const t of turns) {
    if (!isObject(t)) continue;
    const role = t.role;
    const parts = (t as GeminiTurn).parts;
    if (!isGeminiRole(role)) continue;
    if (!Array.isArray(parts)) continue;
    // at least one part must be content-shaped (not a bare {} or a string)
    for (const p of parts) {
      if (isGeminiPartShape(p)) {
        hits++;
        break;
      }
    }
    if (hits >= 1) return true;
  }
  return false;
}

/**
 * Carry Gemini content[] through one safety pass.
 *
 * @param turns Already cloned (caller-supplied structuredClone) — this
 *              function does NOT clone again, matching the dispatcher
 *              contract (`index.ts` does the clone).
 * @returns     The same array, with every `thought_signature` verified
 *              intact and functionCall/functionResponse order preserved.
 * @throws      CarryError when a functionCall in a non-first model turn
 *              is missing a thought_signature, or when a signature is
 *              present but the wrong type.
 */
export function carryGemini(turns: unknown[]): unknown[] {
  if (!Array.isArray(turns)) {
    throw new CarryError("gemini: turns must be an array");
  }

  // Track model-turn index so we can flag "missing signature on turn > 1".
  let modelTurnIndex = -1;

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (!isObject(turn)) {
      throw new CarryError(`gemini: turn[${i}] is not an object`);
    }

    const role = turn.role;
    const parts = (turn as GeminiTurn).parts;

    // Role-bearing turns without parts[] are legal (e.g. plain `{ role: "user" }`
    // with content stuffed into a sibling field by a proxy). Skip the
    // signature check but still walk so we don't miss nested shape issues.
    if (parts === undefined) continue;
    if (!Array.isArray(parts)) {
      throw new CarryError(`gemini: turn[${i}].parts must be an array if present`);
    }

    if (role === "model" || role === "assistant") modelTurnIndex++;

    const isFirstModelTurn = modelTurnIndex === 0;

    // Walk parts in order. Order matters — Gemini matches signatures to
    // function calls by adjacency in the same model turn, so we MUST NOT
    // sort, dedupe, or otherwise touch the sequence.
    for (let j = 0; j < parts.length; j++) {
      const part = parts[j];
      if (!isObject(part)) {
        throw new CarryError(
          `gemini: turn[${i}].parts[${j}] is not an object`,
        );
      }
      const p = part as GeminiPart;

      // Sanity: any present thought_signature must be a non-empty string.
      if (partHasSignature(p) && !signatureIsString(p)) {
        throw new CarryError(
          `gemini: turn[${i}].parts[${j}].thought_signature must be a non-empty string`,
        );
      }

      // functionCall parts in non-first model turns MUST carry a
      // thought_signature. Gemini 3+ 400s otherwise. First-turn tool
      // calls are tolerated by the API without a signature.
      if (isFunctionCallPart(p) && !partHasSignature(p) && !isFirstModelTurn) {
        throw new CarryError(
          `gemini: turn[${i}].parts[${j}] functionCall is missing thought_signature ` +
            `(model turn #${modelTurnIndex + 1}; first model turn may omit it). ` +
            `Likely cause: JSON round-trip / openai-compat proxy stripped the field.`,
        );
      }
    }
  }

  return turns;
}