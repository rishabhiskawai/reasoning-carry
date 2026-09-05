import type { Provider } from "./types.js";
import { CarryError } from "./types.js";
import { carryGemini } from "./gemini.js";
import { carryAnthropic } from "./anthropic.js";
import { carryOpenAI } from "./openai.js";

/**
 * Carry a stored conversation through one safety pass for `provider`:
 * opaque reasoning blobs are preserved byte-identical and in order;
 * everything else passes through untouched (structuredClone).
 * Throws CarryError on shapes that cannot be proven safe.
 */
export function carry(history: unknown[], provider: Provider): unknown[] {
  if (!Array.isArray(history)) throw new CarryError("history must be an array");
  const turns = structuredClone(history);
  switch (provider) {
    case "gemini":
      return carryGemini(turns);
    case "anthropic":
      return carryAnthropic(turns);
    case "openai":
    case "deepseek":
      return carryOpenAI(turns, provider);
    default:
      throw new CarryError(`unknown provider: ${String(provider)}`);
  }
}

export type { Provider } from "./types.js";
export { CarryError } from "./types.js";
export { carryGemini } from "./gemini.js";
export { carryAnthropic } from "./anthropic.js";
export { carryOpenAI } from "./openai.js";
