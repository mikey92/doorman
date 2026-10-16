// Doorman's classifier: EmbeddingGemma vectors and a logistic regression trained on the
// hand-labelled forum posts. The inbox pitches (eval/inbox.jsonl) are never trained on: they are
// the test that matters, real messages sent to my own apps.
//
//   node eval/classify.ts embed   embed every message and its sentences → eval/cache/embeddings.json
//   node eval/classify.ts cv      5-fold cross-validation over forum topics → eval/CLASSIFIER.md
//   node eval/classify.ts fit     train on every forum post → public/classifier.json, then test on the inbox

import { readFileSync, writeFileSync } from "node:fs";
import { EMBEDDER, makeEmbedder, PREFIX } from "../src/embed.ts";
import { predict, train, type Classifier } from "../src/linear.ts";
import { sentences } from "../src/triage.ts";
import { loadMessages, type Labeled } from "./run.ts";

const CACHE = new URL("./cache/embeddings.json", import.meta.url);
const INBOX = new URL("./inbox.jsonl", import.meta.url);

function loadInbox(): Labeled[] {
  return readFileSync(INBOX, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Labeled);
}

type Vectors = Record<string, { message: number[]; sentences: { text: string; v: number[] }[] }>;

async function embed() {
  const { AutoModel, AutoTokenizer } = await import("@huggingface/transformers");
  const tokenizer = await AutoTokenizer.from_pretrained(EMBEDDER.id);
  const model = await AutoModel.from_pretrained(EMBEDDER.id, { dtype: "q4" });
  const e = makeEmbedder(tokenizer, model);
  const out: Vectors = {};
  const all = [...loadMessages(), ...loadInbox()];
  const started = Date.now();
  for (const [i, m] of all.entries()) {
    const ss = sentences(m.body);
    const [mv, ...sv] = await e([m.body.trim(), ...ss]);
    out[m.id] = { message: Array.from(mv, (x) => +x.toFixed(5)), sentences: ss.map((text, k) => ({ text, v: Array.from(sv[k], (x) => +x.toFixed(5)) })) };
    if (i % 20 === 0) console.log(`${i + 1}/${all.length} (${Math.round((Date.now() - started) / 1000)} s)`);
  }
  writeFileSync(CACHE, JSON.stringify(out));
  console.log(`embedded ${all.length} messages in ${Math.round((Date.now() - started) / 1000)} s`);
}

const vectors = () => JSON.parse(readFileSync(CACHE, "utf8")) as Vectors;
const topicOf = (id: string) => id.split("-")[1];
const y = (m: Labeled) => (m.label === "pitch" ? 1 : 0);

const OPTS = { l2: 1e-3, epochs: 400, lr: 0.5, customerWeight: 2 };

/** Out-of-fold P(pitch) for every forum message: each topic is scored by a model that never saw it. */
function crossValidate(ms: Labeled[], v: Vectors, k = 5): Map<string, number> {
  const topics = [...new Set(ms.map((m) => topicOf(m.id)))].sort();
  const fold = new Map(topics.map((t, i) => [t, i % k]));
  const p = new Map<string, number>();
  for (let f = 0; f < k; f++) {
    const tr = ms.filter((m) => fold.get(topicOf(m.id)) !== f);
    const c = train(tr.map((m) => v[m.id].message), tr.map(y), OPTS);
    for (const m of ms.filter((m) => fold.get(topicOf(m.id)) === f)) p.set(m.id, predict(c, v[m.id].message));
  }
  return p;
}

function table(ms: Labeled[], p: Map<string, number>, bars: number[]): string[] {
  const c = ms.filter((m) => m.label === "customer");
  const q = ms.filter((m) => m.label === "pitch");
  return [
    "| Pitch at P ≥ | Customers hidden | Pitches folded away |",
    "|---|---|---|",
    ...bars.map((t) => `| ${t} | ${c.filter((m) => p.get(m.id)! >= t).length} of ${c.length} | ${q.filter((m) => p.get(m.id)! >= t).length} of ${q.length} |`),
  ];
}

const BARS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98];

function cv() {
  const v = vectors();
  const ms = loadMessages();
  const p = crossValidate(ms, v);
  const worst = ms
    .map((m) => ({ m, p: p.get(m.id)! }))
    .filter(({ m, p }) => (m.label === "customer" ? p >= 0.5 : p < 0.5))
    .sort((a, b) => b.p - a.p);
  const md = [
    "# Classifier: cross-validation on the forum posts",
    "",
    `EmbeddingGemma 300M (q4) embeds each post with the prefix \`${PREFIX}\`; a logistic regression over the 768 numbers says how likely it is to be a pitch. Five folds split by forum topic, so no topic is ever scored by a model that saw it.`,
    "",
    ...table(ms, p, BARS),
    "",
    "## Every post on the wrong side of 0.5",
    "",
    ...worst.map(({ m, p }) => `- **${m.label}**, P = ${p.toFixed(2)}: ${m.body.replace(/\s+/g, " ").slice(0, 220)}… ([post](${(m as Labeled & { url?: string }).url}))`),
    "",
  ].join("\n");
  writeFileSync(new URL("./CLASSIFIER.md", import.meta.url), md);
  console.log(md.split("## Every")[0]);
  console.log(`${worst.length} on the wrong side of 0.5`);
}

function fit(pitchAt: number, customerBelow: number) {
  const v = vectors();
  const ms = loadMessages();
  const c = train(ms.map((m) => v[m.id].message), ms.map(y), OPTS);
  const out: Classifier = { model: EMBEDDER.id, prefix: PREFIX, bias: +c.bias.toFixed(6), weights: c.weights.map((w) => +w.toFixed(6)), pitchAt, customerBelow };
  writeFileSync(new URL("../public/classifier.json", import.meta.url), JSON.stringify(out));
  const inbox = loadInbox();
  const ps = inbox.map((m) => ({ m, p: predict(out, v[m.id].message) }));
  console.log(`inbox: ${ps.filter((x) => x.p >= pitchAt).length} of ${inbox.length} folded away, ${ps.filter((x) => x.p < customerBelow).length} called customers`);
  for (const { m, p } of ps.sort((a, b) => a.p - b.p)) console.log(p.toFixed(3), m.body.replace(/\s+/g, " ").slice(0, 100));
}

const [cmd, a, b] = process.argv.slice(2);
if (cmd === "embed") await embed();
else if (cmd === "cv") cv();
else if (cmd === "fit") fit(Number(a ?? 0.9), Number(b ?? 0.5));
else console.log("usage: node eval/classify.ts embed | cv | fit [pitchAt] [customerBelow]");
