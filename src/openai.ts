import { CarryError } from "./types.js";
export function carryOpenAI(_turns: unknown[], _provider: "openai" | "deepseek"): unknown[] { throw new CarryError("stub"); }
export function supportsOpenAI(_turns: unknown[]): boolean { return false; }
