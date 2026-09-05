/**
 * Gemini 3+ content[] codec for reasoning-carry.
 *
 * Background: Gemini 3+ stamps an opaque `thoughtSignature` (REST/camelCase;
 * some SDKs and proxies surface it as `thought_signature`) on text and
 * functionCall parts returned by `generateContent`. That signature MUST be
 * round-tripped byte-identical on the next turn — drop it (e.g. JSON
 * clone, openai-compat proxy rewriting unknown fields, "sanitizer" that
 * strips keys starting with "thought_") and the next call 400s:
 *   "Unable to submit request because function call parameter is missing
 *    a thoughtSignature. Function calls require this field to be present
 *    and unaltered."
 *
 * Production rules encoded here (Gemini 3 spec):
 * - BOTH `thoughtSignature` and `thought_signature` are accepted; whichever
 *   is present must be a non-empty string.
 * - Only the CURRENT turn is validated: everything from the newest user
 *   TEXT turn forward. Older turns are skipped — compacted histories that
 *   dropped old signatures are valid and must pass.
 * - Within the current turn, the FIRST functionCall part of each model step
 *   must carry a signature (including the very first model step — the
 *   first-turn exemption was Gemini 2.5 behavior). Later parallel
 *   functionCalls in the same step may or may not repeat it.
 * - Plain text parts without a signature pass: we lack the signal to prove
 *   an image-edit workflow needed one, and guessing would false-positive.
 *
 * Strategy: verify in place on the dispatcher-cloned array (caller's input
 * is treated as frozen — we never mutate it). Part order is never touched:
 * Gemini matches signatures to calls by adjacency in the same model turn.
 */
import { CarryError } from "./types.js";

// --- shape probing (minimal, duck-typed; we don't trust `any`) ----------

type GeminiTurn = {
  role?: unknown;
  parts?: unknown;
} & Record<string, unknown>;

type GeminiPart = Record<string, unknown>;

/** Both spellings seen in the wild: REST camelCase and snake_case. */
const SIGNATURE_KEYS = ["thoughtSignature", "thought_signature"] as const;

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

function signatureKeyOf(part: GeminiPart): string | undefined {
  for (const key of SIGNATURE_KEYS) {
    if (Object.prototype.hasOwnProperty.call(part, key)) return key;
  }
  return undefined;
}

function partHasSignature(part: GeminiPart): boolean {
  return signatureKeyOf(part) !== undefined;
}

function isFunctionCallPart(part: GeminiPart): boolean {
  return isObject(part.functionCall);
}

function isModelRole(role: unknown): boolean {
  return role === "model" || role === "assistant";
}

/** A user turn carrying typed text starts a new turn; functionResponse-only turns continue the tool loop. */
function isUserTextTurn(turn: GeminiTurn): boolean {
  if (turn.role !== "user" || !Array.isArray(turn.parts)) return false;
  return (turn.parts as unknown[]).some((p) => isObject(p) && typeof p.text === "string");
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
 * @returns     The same array, with every current-turn first-functionCall
 *              signature verified intact.
 * @throws      CarryError when a current-turn model step's first
 *              functionCall lacks a thoughtSignature/thought_signature, or
 *              when a present signature is not a non-empty string.
 */
export function carryGemini(turns: unknown[]): unknown[] {
  if (!Array.isArray(turns)) {
    throw new CarryError("gemini: turns must be an array");
  }

  // Scope: only the current turn is validated — from the newest user TEXT
  // turn forward. functionResponse-only user turns are tool-loop
  // continuations, not new turns. Older turns are skipped so compacted
  // histories (old signatures dropped) pass.
  let scopeStart = 0;
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i];
    if (!isObject(turn)) continue;
    if (isUserTextTurn(turn as GeminiTurn)) {
      scopeStart = i;
      break;
    }
  }

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

    // Walk parts in order. Order matters — Gemini matches signatures to
    // function calls by adjacency in the same model turn, so we MUST NOT
    // sort, dedupe, or otherwise touch the sequence.
    let firstFunctionCallIndex = -1;
    for (let j = 0; j < parts.length; j++) {
      const part = parts[j];
      if (!isObject(part)) {
        throw new CarryError(
          `gemini: turn[${i}].parts[${j}] is not an object`,
        );
      }
      const p = part as GeminiPart;

      // Sanity (all turns, any age): a present signature under EITHER
      // spelling must be a non-empty string. Presence with the wrong type
      // is never valid — bytes/objects/numbers are not signatures.
      const key = signatureKeyOf(p);
      if (key !== undefined && (typeof p[key] !== "string" || (p[key] as string).length === 0)) {
        throw new CarryError(
          `gemini: turn[${i}].parts[${j}].${key} must be a non-empty string`,
        );
      }

      if (isFunctionCallPart(p) && firstFunctionCallIndex === -1) {
        firstFunctionCallIndex = j;
      }
    }

    // Pairing rule (current turn only): the FIRST functionCall of each
    // model step must carry a signature — including the first model step
    // (Gemini 3; the old first-turn exemption was 2.5 behavior). Later
    // parallel calls in the same step may omit it.
    if (
      i >= scopeStart &&
      isModelRole(role) &&
      firstFunctionCallIndex !== -1 &&
      !partHasSignature(parts[firstFunctionCallIndex] as GeminiPart)
    ) {
      throw new CarryError(
        `gemini: turn[${i}].parts[${firstFunctionCallIndex}] functionCall is missing thought_signature/thoughtSignature ` +
          `(first call of this model step must carry it). ` +
          `Likely cause: JSON round-trip / openai-compat proxy stripped the field.`,
      );
    }
  }

  return turns;
}
