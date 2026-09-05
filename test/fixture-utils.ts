import { readdirSync, readFileSync } from "node:fs";
import type { Provider } from "../src/types.js";

export const providers: Provider[] = ["gemini", "anthropic", "openai", "deepseek"];
export const requiredCases = [
  "happy-path", "signature-stripped", "json-clone-mangled", "frozen-input",
  "empty-history", "garbage-input", "text-only",
];
export interface Fixture {
  input: unknown[];
  mustPreserve: string[];
  mustThrow?: boolean;
}
export interface FixtureFile {
  provider: Provider;
  name: string;
  raw: unknown;
}
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Deliberately narrow JSONPath subset: $[index].field[index].field.
// Fail unresolved paths, rather than accidentally comparing undefined === undefined.
export function atPath(value: unknown, path: string): unknown {
  if (!/^\$(?:\[\d+\]|\.[A-Za-z_][A-Za-z_0-9]*)+$/.test(path)) {
    throw new Error(`Unsupported preservation path: ${path}`);
  }
  for (const token of path.slice(1).match(/\[\d+\]|\.[A-Za-z_][A-Za-z_0-9]*/g) ?? []) {
    const key = token.startsWith("[") ? token.slice(1, -1) : token.slice(1);
    if (value === null || typeof value !== "object" || !Object.hasOwn(value, key)) {
      throw new Error(`Unresolved preservation path: ${path}`);
    }
    value = (value as Record<string, unknown>)[key];
  }
  return value;
}

export function readCorpus(): FixtureFile[] {
  const root = new URL("./fixtures/", import.meta.url);
  return readdirSync(root).sort().flatMap((directory) => {
    if (!providers.includes(directory as Provider)) throw new Error(`Unknown fixture provider: ${directory}`);
    return readdirSync(new URL(`${directory}/`, root)).sort().map((file) => {
      if (!file.endsWith(".json")) throw new Error(`Unexpected fixture file: ${directory}/${file}`);
      return {
        provider: directory as Provider,
        name: file.slice(0, -5),
        raw: JSON.parse(readFileSync(new URL(`${directory}/${file}`, root), "utf8")) as unknown,
      };
    });
  });
}

export function asFixture(raw: unknown): Fixture {
  if (!isRecord(raw) || !Array.isArray(raw.input) || !Array.isArray(raw.mustPreserve)
    || !raw.mustPreserve.every((path) => typeof path === "string")
    || (raw.mustThrow !== undefined && typeof raw.mustThrow !== "boolean")) {
    throw new Error("Fixture requires input array, mustPreserve strings, and optional boolean mustThrow");
  }
  return raw as unknown as Fixture;
}

// Detect replayable opaque fields, not just their names. An unsigned Claude
// thinking block is NOT preservable, even if its visible thinking text remains.
export function blobPaths(value: unknown, provider: Provider, path = "$"): string[] {
  if (Array.isArray(value)) return value.flatMap((v, i) => blobPaths(v, provider, `${path}[${i}]`));
  if (!isRecord(value)) return [];
  const found: string[] = [];
  const nonempty = (key: string) => typeof value[key] === "string" && value[key].length > 0;
  if (provider === "gemini" && nonempty("thought_signature")) found.push(`${path}.thought_signature`);
  if (provider === "gemini" && nonempty("thoughtSignature")) found.push(`${path}.thoughtSignature`);
  if (provider === "anthropic") {
    if (value.type === "thinking" && typeof value.thinking === "string" && nonempty("signature")) {
      found.push(`${path}.thinking`, `${path}.signature`);
    }
    if (value.type === "redacted_thinking" && nonempty("data")) found.push(`${path}.data`);
  }
  if (provider === "openai" && value.type === "reasoning" && nonempty("encrypted_content")) {
    found.push(`${path}.encrypted_content`);
  }
  if (provider === "deepseek" && typeof value.reasoning_content === "string") {
    found.push(`${path}.reasoning_content`);
  }
  return found.concat(Object.entries(value).flatMap(([key, v]) => blobPaths(v, provider, `${path}.${key}`)));
}

export function deepFreeze(value: unknown): void {
  if (value !== null && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
}
