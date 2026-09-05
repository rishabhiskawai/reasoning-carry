/** Shared types for reasoning-carry. Reviewer-owned: do not edit without asking. */
export type Provider = "gemini" | "anthropic" | "openai" | "deepseek";

export class CarryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CarryError";
  }
}
