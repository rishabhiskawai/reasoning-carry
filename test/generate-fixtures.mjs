// Deterministic synthetic corpus. Run: node test/generate-fixtures.mjs
// No real provider data or cryptographic validity claims. JSON-clone-mangled
// cases model an adapter replacing a string with bytes, then JSON cloning it;
// JSON cloning an ordinary string does NOT itself corrupt that string.
import { mkdirSync, writeFileSync, readdirSync, readFileSync } from "node:fs";

const root = new URL("./fixtures/", import.meta.url);
const clone = (value) => JSON.parse(JSON.stringify(value));
const fixture = (input, mustPreserve = [], mustThrow = false) =>
  mustThrow ? { input, mustPreserve, mustThrow: true } : { input, mustPreserve };
const mangledBytes = () => clone(new Uint8Array([83, 89, 78, 84, 72]));
const corpora = {
  gemini: {
    "happy-path": fixture([
      { role: "user", parts: [{ text: "Synthetic: check the weather in Testville." }] },
      { role: "model", parts: [
        { thought: true, text: "Synthetic internal note.", thought_signature: "U1lOVEhfR0VNX1RFWFQ9PQ==" },
        { functionCall: { id: "call_synthetic_g1", name: "weather", args: { city: "Testville" } }, thought_signature: "U1lOVEhfR0VNX0NBTExfMQ==" },
        { functionCall: { id: "call_synthetic_g2", name: "clock", args: { zone: "UTC" } }, thought_signature: "U1lOVEhfR0VNX0NBTExfMg==" },
      ] },
      { role: "user", parts: [
        { functionResponse: { id: "call_synthetic_g1", name: "weather", response: { temperature: 21 } } },
        { functionResponse: { id: "call_synthetic_g2", name: "clock", response: { time: "12:00" } } },
      ] },
      { role: "model", parts: [{ text: "Synthetic: 21 degrees at noon.", thought_signature: "U1lOVEhfR0VNX0ZJTkFM" }] },
    ], ["$[1].parts[0].thought_signature", "$[1].parts[1].thought_signature", "$[1].parts[2].thought_signature", "$[3].parts[0].thought_signature"]),
    "signature-stripped": fixture([
      { role: "model", parts: [{ thought: true, text: "Synthetic replay lost its signature." },
        { functionCall: { id: "call_synthetic_g1", name: "weather", args: { city: "Testville" } } }] },
    ], [], true),
    "json-clone-mangled": fixture([
      { role: "model", parts: [{ thought: true, text: "Synthetic adapter byte conversion.", thought_signature: mangledBytes() }] },
    ], [], true),
    "empty-history": fixture([]),
    "garbage-input": fixture([null, 17, "not a Gemini turn", { role: "model", parts: "not an array" }], [], true),
    "text-only": fixture([
      { role: "user", parts: [{ text: "Hello" }] },
      { role: "model", parts: [{ text: "Synthetic plain response" }] },
    ]),
  },
  anthropic: {
    "happy-path": fixture([
      { role: "user", content: "Synthetic: what is the weather?" },
      { role: "assistant", content: [
        { type: "thinking", thinking: "Synthetic thought.\nKeep spacing:  two spaces. λ", signature: "U1lOVEhfQ0xBVURFX1NJRzE=" },
        { type: "redacted_thinking", data: "U1lOVEhfUkVEQUNURURfREFUQQ==" },
        { type: "text", text: "Synthetic: checking." },
        { type: "tool_use", id: "toolu_synthetic_1", name: "weather", input: { city: "Testville" } },
      ] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_synthetic_1", content: "21 degrees" }] },
      { role: "assistant", content: [
        { type: "thinking", thinking: "Synthetic follow-up thought.", signature: "U1lOVEhfQ0xBVURFX1NJRzI=" },
        { type: "text", text: "Synthetic: 21 degrees." },
      ] },
    ], ["$[1].content[0].thinking", "$[1].content[0].signature", "$[1].content[1].data", "$[3].content[0].thinking", "$[3].content[0].signature"]),
    "signature-stripped": fixture([
      { role: "assistant", content: [{ type: "thinking", thinking: "Synthetic replay with signature removed." }, { type: "text", text: "Answer" }] },
    ], [], true),
    "json-clone-mangled": fixture([
      { role: "assistant", content: [{ type: "thinking", thinking: "Synthetic adapter byte conversion.", signature: mangledBytes() }] },
    ], [], true),
    "empty-history": fixture([]),
    "garbage-input": fixture([null, 17, "not a Claude turn", { role: "assistant", content: 42 }], [], true),
    "text-only": fixture([{ role: "user", content: "Hello" }, { role: "assistant", content: [{ type: "text", text: "Synthetic plain response" }] }]),
  },
  openai: {
    "happy-path": fixture([
      { role: "user", content: "Synthetic: check the weather." },
      { type: "reasoning", id: "rs_synthetic_1", summary: [{ type: "summary_text", text: "Synthetic weather lookup" }], encrypted_content: "U1lOVEhfT1BFTkFJX0VOQ1JZUFRFRDE=" },
      { type: "function_call", id: "fc_synthetic_1", call_id: "call_synthetic_o1", name: "weather", arguments: "{\"city\":\"Testville\"}", status: "completed" },
      { type: "function_call_output", call_id: "call_synthetic_o1", output: "{\"temperature\":21}" },
      { type: "reasoning", id: "rs_synthetic_2", summary: [], encrypted_content: "U1lOVEhfT1BFTkFJX0VOQ1JZUFRFRDI=" },
      { type: "message", id: "msg_synthetic_1", role: "assistant", status: "completed", content: [{ type: "output_text", text: "Synthetic: 21 degrees.", annotations: [] }] },
    ], ["$[1].encrypted_content", "$[4].encrypted_content"]),
    // Stateless/store:false replay cannot recover this omitted payload locally.
    "signature-stripped": fixture([
      { type: "reasoning", id: "rs_synthetic_lost", summary: [{ type: "summary_text", text: "Synthetic summary is not the encrypted payload" }] },
    ], [], true),
    "json-clone-mangled": fixture([
      { type: "reasoning", id: "rs_synthetic_mangled", summary: [], encrypted_content: mangledBytes() },
    ], [], true),
    "empty-history": fixture([]),
    "garbage-input": fixture([null, 17, "not a Responses item", { type: "reasoning", summary: "not an array", encrypted_content: null }], [], true),
    "text-only": fixture([{ role: "user", content: "Hello" }, { role: "assistant", content: "Synthetic plain response" }]),
  },
  deepseek: {
    "happy-path": fixture([
      { role: "user", content: "Synthetic: check the weather." },
      { role: "assistant", content: null, reasoning_content: "Synthetic reasoning: keep  spaces.\nLine two\tλ🙂", tool_calls: [
        { id: "call_synthetic_d1", type: "function", function: { name: "weather", arguments: "{\"city\":\"Testville\"}" } },
      ] },
      { role: "tool", tool_call_id: "call_synthetic_d1", content: "21 degrees" },
      { role: "assistant", content: "Synthetic: 21 degrees.", reasoning_content: "Synthetic follow-up.\r\nDo not normalize. " },
    ], ["$[1].reasoning_content", "$[3].reasoning_content"]),
    // A plain assistant message need not be reasoning-enabled. Retain tool_calls
    // so this is a detectable broken reasoning/tool replay, not guessed history.
    "signature-stripped": fixture([
      { role: "assistant", content: null, tool_calls: [
        { id: "call_synthetic_d1", type: "function", function: { name: "weather", arguments: "{}" } },
      ] },
    ], [], true),
    "json-clone-mangled": fixture([
      { role: "assistant", content: "Synthetic answer", reasoning_content: mangledBytes() },
    ], [], true),
    "empty-history": fixture([]),
    "garbage-input": fixture([null, 17, "not a DeepSeek turn", { role: "assistant", content: [], reasoning_content: null }], [], true),
    "text-only": fixture([{ role: "user", content: "Hello" }, { role: "assistant", content: "Synthetic plain response" }]),
  },
};

for (const [provider, cases] of Object.entries(corpora)) {
  cases["frozen-input"] = clone(cases["happy-path"]);
  const directory = new URL(`${provider}/`, root);
  mkdirSync(directory, { recursive: true });
  for (const [name, data] of Object.entries(cases)) {
    writeFileSync(new URL(`${name}.json`, directory), `${JSON.stringify(data, null, 2)}\n`);
  }
  console.log(`${provider}: ${Object.keys(cases).length} fixtures`);
}

const written = Object.keys(corpora).flatMap((provider) => {
  const directory = new URL(`${provider}/`, root);
  return readdirSync(directory).filter((file) => file.endsWith(".json"))
    .map((file) => JSON.parse(readFileSync(new URL(file, directory), "utf8")));
});
const mustThrow = written.filter((entry) => entry.mustThrow === true).length;
console.log(JSON.stringify({ total: written.length, mustThrow, mustSucceed: written.length - mustThrow }));
