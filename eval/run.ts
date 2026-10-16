// How often does Doorman fold away a real customer, and how many pitches does it catch?
//
//   node eval/run.ts run <model> [tag]   rate every message in eval/messages.jsonl, on the CPU
//   node eval/run.ts report              eval/EVAL.md from every run in eval/results
//
// Messages go through the same triage() the page uses. Only the runtime differs: here it is
// onnxruntime-node with the q4 weights; the page uses WebGPU with q4f16 where it can.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { makeScorer } from "../src/model.ts";
import { CUSTOMER_BELOW, MODELS, PITCH_AT, triage, type ModelKey, type Triage } from "../src/triage.ts";

export interface Labeled {
  id: string;
  /** What the message really is, decided by reading it. */
  label: "customer" | "pitch";
  /** Where it came from: see eval/DATA.md. */
  source: string;
  body: string;
}

const DATA = new URL("./messages.jsonl", import.meta.url);
const RESULTS = new URL("./results/", import.meta.url);

export function loadMessages(): Labeled[] {
  return readFileSync(DATA, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Labeled);
}

type Run = Record<string, Triage & { label: Labeled["label"]; source: string; ms: number }>;

async function run(key: ModelKey, tag: string) {
  const { pipeline } = await import("@huggingface/transformers");
  const started = Date.now();
  const generator = await pipeline("text-generation", MODELS[key].id, { dtype: "q4", device: "cpu" });
  console.log(`${MODELS[key].label} loaded in ${Math.round((Date.now() - started) / 1000)} s`);
  const score = makeScorer(generator);
  const name = `${key}${tag ? `-${tag}` : ""}`;
  const file = new URL(`${name}.json`, RESULTS);
  mkdirSync(RESULTS, { recursive: true });
  // A run can be stopped and picked up again: messages already rated are kept.
  let out: Run = {};
  try {
    out = JSON.parse(readFileSync(file, "utf8")) as Run;
  } catch {
    /* a new run */
  }
  const messages = loadMessages();
  for (const [i, m] of messages.entries()) {
    if (out[m.id]) continue;
    const t0 = Date.now();
    const t = await triage(score, m.body);
    out[m.id] = { ...t, label: m.label, source: m.source, ms: Date.now() - t0 };
    const right = t.verdict === "unsure" ? "?" : (t.verdict === "pitch") === (m.label === "pitch") ? "ok" : "WRONG";
    console.log(`${i + 1}/${messages.length} ${right} ${m.id} [${m.label} -> ${t.verdict} ${t.score.toFixed(2)}] ${t.quote || t.rated[0]?.sentence || ""}`);
    writeFileSync(file, JSON.stringify(out, null, 1));
  }
  await generator.dispose();
}

export interface Score {
  customers: number;
  pitches: number;
  /** Customers folded away with the pitches: the mistake that costs a sale. */
  customersHidden: number;
  customersUnsure: number;
  pitchesCaught: number;
  pitchesUnsure: number;
}

export function score(r: Run, pitchAt = PITCH_AT, customerBelow = CUSTOMER_BELOW, source?: string): Score {
  const all = Object.values(r).filter((x) => !source || x.source === source);
  const c = all.filter((x) => x.label === "customer");
  const p = all.filter((x) => x.label === "pitch");
  const unsure = (x: { score: number }) => x.score >= customerBelow && x.score < pitchAt;
  return {
    customers: c.length,
    pitches: p.length,
    customersHidden: c.filter((x) => x.score >= pitchAt).length,
    customersUnsure: c.filter(unsure).length,
    pitchesCaught: p.filter((x) => x.score >= pitchAt).length,
    pitchesUnsure: p.filter(unsure).length,
  };
}

const row = (name: string, s: Score) =>
  `| ${name} | ${s.customersHidden} of ${s.customers} | ${s.customersUnsure} | ${s.pitchesCaught} of ${s.pitches} | ${s.pitchesUnsure} |`;
const HEAD = ["| | Customers hidden | Customers left as unsure | Pitches folded away | Pitches left as unsure |", "|---|---|---|---|---|"];

function report() {
  const files = readdirSync(RESULTS).filter((f) => f.endsWith(".json")).sort();
  const runs = files.map((f) => [f.replace(/\.json$/, ""), JSON.parse(readFileSync(new URL(f, RESULTS), "utf8")) as Run] as const);
  const lines = [
    "# Evaluation",
    "",
    "Every run rates every message in `eval/messages.jsonl` with the same `triage()` the page uses. Scores are read from the model's next-token logits, so a run is reproducible. See `eval/DATA.md` for where the messages come from and how they were labelled.",
    "",
    `## At the page's settings (pitch at log-odds ≥ ${PITCH_AT}, customer below ${CUSTOMER_BELOW})`,
    "",
    ...HEAD,
    ...runs.map(([n, r]) => row(n, score(r))),
    "",
    "## By source",
    "",
    ...HEAD,
    ...runs.flatMap(([n, r]) => [...new Set(Object.values(r).map((x) => x.source))].sort().map((src) => row(`${n} · ${src}`, score(r, PITCH_AT, CUSTOMER_BELOW, src)))),
    "",
    "## Where the line for \"pitch\" could go",
    "",
    "Every message is placed by its highest-rated sentence. Moving the bar trades customers hidden against pitches caught.",
    "",
    ...runs.flatMap(([n, r]) => [
      `### ${n}`,
      "",
      "| Pitch at log-odds ≥ | Customers hidden | Pitches folded away |",
      "|---|---|---|",
      ...[0, 2, 4, 6, 8, 9, 10, 12].map((t) => {
        const s = score(r, t, Math.min(t, CUSTOMER_BELOW));
        return `| ${t} | ${s.customersHidden} of ${s.customers} | ${s.pitchesCaught} of ${s.pitches} |`;
      }),
      "",
    ]),
  ];
  const md = lines.join("\n");
  writeFileSync(new URL("./EVAL.md", import.meta.url), md);
  console.log(md);
}

const [cmd, a, b] = process.argv.slice(2);
if (cmd === "run" && a && a in MODELS) await run(a as ModelKey, b ?? "");
else if (cmd === "report") report();
else console.log("usage: node eval/run.ts run <gemma-3-1b|gemma-3-270m> [tag] | report");
