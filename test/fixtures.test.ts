import { describe, expect, it } from "vitest";
import { asFixture, atPath, blobPaths, providers, readCorpus, requiredCases } from "./fixture-utils.js";

const corpus = readCorpus();
describe("standalone fixture corpus (no codecs)", () => {
  for (const provider of providers) {
    it(`${provider}: all required cases are present`, () => {
      const names = corpus.filter((file) => file.provider === provider).map((file) => file.name);
      expect(names.length).toBeGreaterThanOrEqual(6);
      expect(new Set(names).size).toBe(names.length);
      expect(names).toEqual(expect.arrayContaining(requiredCases));
    });
  }
  for (const { provider, name, raw } of corpus) {
    it(`${provider}/${name}: parses and declares every preservable blob`, () => {
      const fixture = asFixture(raw);
      expect(new Set(fixture.mustPreserve).size).toBe(fixture.mustPreserve.length);
      const blobs = blobPaths(fixture.input, provider).sort();
      expect([...fixture.mustPreserve].sort()).toEqual(blobs);
      for (const path of fixture.mustPreserve) expect(typeof atPath(fixture.input, path)).toBe("string");
      if (fixture.mustThrow) {
        expect(fixture.mustPreserve).toEqual([]);
        expect(blobs).toEqual([]);
        expect(fixture.input.length).toBeGreaterThan(0);
      }
      if (["signature-stripped", "json-clone-mangled", "garbage-input"].includes(name)) {
        expect(fixture.mustThrow).toBe(true);
      } else {
        expect(fixture.mustThrow ?? false).toBe(false);
      }
      if (["happy-path", "frozen-input"].includes(name)) expect(blobs.length).toBeGreaterThan(0);
      if (name === "empty-history") expect(fixture.input).toEqual([]);
      if (name === "frozen-input") {
        const happy = corpus.find((file) => file.provider === provider && file.name === "happy-path");
        expect(fixture).toEqual(asFixture(happy?.raw));
      }
    });
  }
  it("path resolution rejects missing paths and unsupported syntax", () => {
    expect(atPath([{ parts: [{ thought_signature: "synthetic==" }] }], "$[0].parts[0].thought_signature"))
      .toBe("synthetic==");
    expect(() => atPath([], "$[0].signature")).toThrow("Unresolved");
    expect(() => atPath([], "$[*].signature")).toThrow("Unsupported");
  });
  it("schema rejects malformed corpus records", () => {
    for (const raw of [null, {}, { input: null, mustPreserve: [] },
      { input: [], mustPreserve: [42] }, { input: [], mustPreserve: [], mustThrow: "yes" }]) {
      expect(() => asFixture(raw)).toThrow();
    }
  });
  it("unsigned and mangled fields are not counted as replayable blobs", () => {
    expect(blobPaths([{ type: "thinking", thinking: "synthetic unsigned text" }], "anthropic")).toEqual([]);
    expect(blobPaths([{ thought_signature: null }], "gemini")).toEqual([]);
    expect(blobPaths([{ type: "reasoning", encrypted_content: {} }], "openai")).toEqual([]);
    expect(blobPaths([{ reasoning_content: [] }], "deepseek")).toEqual([]);
  });
});
