import { describe, expect, it } from "vitest";
import { carry } from "./index.js";
import { CarryError, type Provider } from "./types.js";
import { asFixture, atPath, deepFreeze, providers, readCorpus } from "../test/fixture-utils.js";

const corpus = readCorpus();
describe("carry: real-dispatcher fixture integration", () => {
  it("discovers fixtures for every provider (never vacuously passes)", () => {
    for (const provider of providers) {
      expect(corpus.filter((file) => file.provider === provider).length).toBeGreaterThanOrEqual(6);
    }
  });
  for (const { provider, name, raw } of corpus) {
    it(`${provider}/${name}`, () => {
      const fixture = asFixture(raw);
      const input = structuredClone(fixture.input);
      const before = structuredClone(input);
      if (name === "frozen-input") deepFreeze(input);
      if (fixture.mustThrow) {
        expect(() => carry(input, provider)).toThrow(CarryError);
      } else {
        const output = carry(input, provider);
        // Strict structural equality locks message/part order and semantic identity,
        // including IDs, tool associations, summaries and unrelated extension fields.
        // Reference identity is not required: the public dispatcher clones history.
        expect(output).toStrictEqual(before);
        for (const path of fixture.mustPreserve) {
          const original = atPath(before, path);
          const preserved = atPath(output, path);
          expect(typeof original).toBe("string");
          expect(preserved).toBe(original);
          expect(Buffer.from(preserved as string, "utf8")).toEqual(Buffer.from(original as string, "utf8"));
        }
        expect(carry(output, provider)).toStrictEqual(output);
      }
      expect(input).toStrictEqual(before);
    });
  }
});

describe("carry dispatcher guards", () => {
  for (const provider of providers) {
    it(`${provider}: rejects non-array history with CarryError`, () => {
      for (const history of [null, undefined, {}, "history", 7, true]) {
        expect(() => carry(history as unknown as unknown[], provider)).toThrow(CarryError);
      }
    });
  }
  it("rejects unknown providers with CarryError", () => {
    for (const provider of ["unknown", "Gemini", "", null, undefined]) {
      expect(() => carry([], provider as Provider)).toThrow(CarryError);
    }
  });
});
