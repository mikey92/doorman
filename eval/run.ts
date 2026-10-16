// How often does Doorman fold away a real customer, and how many pitches does it catch?
//
//   node eval/run.ts run <model> [tag]   triage every message in eval/messages.jsonl, on the CPU
//   node eval/run.ts report              eval/EVAL.md from every run in eval/results
//
// Messages go through the same triage() the page uses. Only the runtime differs: here it is
// onnxruntime-node with the q4 weights; the page uses WebGPU with q4f16 where it can.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { MODELS, triage, type ChatMessage, type Generate, type ModelKey, type Triage } from "../src/triage.ts";

export interface Labeled {
  id: string;
  /** What the message really is, decided by reading it. */
  label: "customer" | "pitch";
  /** Where it came from: "inbox" (sent to my apps), "public" (a public post), see eval/DATA.md. */
  source: string;
  body: string;
}

const DATA = new URL("./messages.jsonl", import.meta.url);
const RESULTS = new URL("./results/", import.meta.url);

export function loadMessages(): Labeled[] {
  return readFileSync(DATA, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Labeled);
}

type Run = Record<string, Triage & { label: Labeled["label"]; source: string }>;

async function run(key: ModelKey, tag: string) {
  const { pipeline } = await import("@huggingface/transformers");
  const started = Date.now();
  const generator = await pipeline("text-generation", MODELS[key].id, { dtype: "q4", device: "cpu" });
  console.log(`${MODELS[key].label} loaded in ${Math.round((Date.now() - started) / 1000)} s`);
  const generate: Generate = async (messages: ChatMessage[], maxNewTokens: number) => {
    const out = await generator(messages, { max_new_tokens: maxNewTokens, do_sample: false });
    const reply = (out[0] as { generated_text: ChatMessage[] }).generated_text.at(-1);
    return typeof reply?.content === "string" ? reply.content : "";
  };
  const out: Run = {};
  const messages = loadMessages();
  for (const [i, m] of messages.entries()) {
    const t = await triage(generate, m.body);
    out[m.id] = { ...t, label: m.label, source: m.source };
    const mark = t.verdict === "unsure" ? "?" : (t.verdict === "pitch") === (m.label === "pitch") ? "ok" : "WRONG";
    console.log(`${i + 1}/${messages.length} ${mark} ${m.id} [${m.label} -> ${t.verdict}] ${t.quote || t.attempts.map((a) => a.raw.replace(/\s+/g, " ")).join(" | ")}`);
  }
  await generator.dispose();
  mkdirSync(RESULTS, { recursive: true });
  writeFileSync(new URL(`${key}${tag ? `-${tag}` : ""}.json`, RESULTS), JSON.stringify(out, null, 1));
}

export interface Score {
  customers: number;
  pitches: number;
  /** Customers folded away with the pitches: the mistake that costs a sale. */
  customersHidden: number;
  customersUnsure: number;
  pitchesCaught: number;
  pitchesUnsure: number;
  firstTry: number;
}

export function score(r: Run): Score {
  const all = Object.values(r);
  const c = all.filter((x) => x.label === "customer");
  const p = all.filter((x) => x.label === "pitch");
  return {
    customers: c.length,
    pitches: p.length,
    customersHidden: c.filter((x) => x.verdict === "pitch").length,
    customersUnsure: c.filter((x) => x.verdict === "unsure").length,
    pitchesCaught: p.filter((x) => x.verdict === "pitch").length,
    pitchesUnsure: p.filter((x) => x.verdict === "unsure").length,
    firstTry: all.filter((x) => x.verdict !== "unsure" && x.attempts.length === 1).length,
  };
}

function report() {
  const files = readdirSync(RESULTS).filter((f) => f.endsWith(".json")).sort();
  const rows = files.map((f) => {
    const s = score(JSON.parse(readFileSync(new URL(f, RESULTS), "utf8")) as Run);
    return `| ${f.replace(/\.json$/, "")} | ${s.customersHidden} of ${s.customers} | ${s.customersUnsure} | ${s.pitchesCaught} of ${s.pitches} | ${s.pitchesUnsure} | ${s.firstTry} |`;
  });
  const md = [
    "# Evaluation",
    "",
    "Every run triages every message in `eval/messages.jsonl` with the same `triage()` the page uses (greedy decoding, so a run is reproducible). See `eval/DATA.md` for where the messages come from.",
    "",
    "| Run | Customers hidden | Customers left as unsure | Pitches folded away | Pitches left as unsure | Settled on the first try |",
    "|---|---|---|---|---|---|",
    ...rows,
    "",
  ].join("\n");
  writeFileSync(new URL("./EVAL.md", import.meta.url), md);
  console.log(md);
}

const [cmd, a, b] = process.argv.slice(2);
if (cmd === "run" && a && a in MODELS) await run(a as ModelKey, b ?? "");
else if (cmd === "report") report();
else console.log("usage: node eval/run.ts run <gemma-3-1b|gemma-3-270m> [tag] | report");
