// Doorman's classifier: EmbeddingGemma vectors and a logistic regression trained on the
// hand-labelled forum posts and my inbox's pitches. The evaluation keeps them apart: forum topics
// are scored by models that never saw them, and each inbox sender by a model that never saw them.
//
//   node eval/classify.ts embed   embed every message and its sentences → eval/cache/embeddings.json
//   node eval/classify.ts cv      cross-validation on the forum, and the inbox held out → eval/CLASSIFIER.md
//   node eval/classify.ts fit     train on everything → public/classifier.json

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
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
  mkdirSync(new URL("./", CACHE), { recursive: true });
  writeFileSync(CACHE, JSON.stringify(out));
  console.log(`embedded ${all.length} messages in ${Math.round((Date.now() - started) / 1000)} s`);
}

const vectors = () => JSON.parse(readFileSync(CACHE, "utf8")) as Vectors;
const topicOf = (id: string) => id.split("-")[1];
const y = (m: Labeled) => (m.label === "pitch" ? 1 : 0);

// Chosen by cross-validation on the forum posts alone (eval/CLASSIFIER.md). Hiding a customer
// counts three times as much as missing a pitch.
const OPTS = { l2: 1e-4, epochs: 3000, lr: 2, customerWeight: 3 };

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

const BARS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95];

/** Chance that a random pitch scores above a random customer (1 = always, 0.5 = a coin). */
function auc(ps: number[], cs: number[]): number {
  let n = 0;
  for (const a of ps) for (const b of cs) n += a > b ? 1 : a === b ? 0.5 : 0;
  return n / (ps.length * cs.length);
}

function inboxTable(ps: number[], bars: number[]): string[] {
  return ["| Pitch at P ≥ | Pitches folded away |", "|---|---|", ...bars.map((t) => `| ${t} | ${ps.filter((p) => p >= t).length} of ${ps.length} |`)];
}

function cv() {
  const v = vectors();
  const ms = loadMessages();
  const inbox = loadInbox();
  const p = crossValidate(ms, v);
  const worst = ms
    .map((m) => ({ m, p: p.get(m.id)! }))
    .filter(({ m, p }) => (m.label === "customer" ? p >= 0.5 : p < 0.5))
    .sort((a, b) => b.p - a.p);
  // The inbox, scored by a model that has only seen the forum.
  const forumOnly = train(ms.map((m) => v[m.id].message), ms.map(y), OPTS);
  const fromForum = inbox.map((m) => predict(forumOnly, v[m.id].message));
  // The inbox, each sender scored by a model trained on the forum and every other sender.
  const groupOf = (m: Labeled) => (m as Labeled & { group: string }).group;
  const heldOut = inbox.map((m) => {
    const tr = [...ms, ...inbox.filter((o) => groupOf(o) !== groupOf(m))];
    return predict(train(tr.map((o) => v[o.id].message), tr.map(y), OPTS), v[m.id].message);
  });
  const byLabel = (ids: Labeled[], label: string, score: (m: Labeled) => number) => ids.filter((m) => m.label === label).map(score);
  const forumAuc = auc(byLabel(ms, "pitch", (m) => p.get(m.id)!), byLabel(ms, "customer", (m) => p.get(m.id)!));
  // The 40 "dev" messages Gemma 3 1B was tried on (eval/EVAL.md), scored by a model trained on the other 205.
  const dev = ms.filter((m) => m.split === "dev");
  const rest = ms.filter((m) => m.split !== "dev");
  const onRest = train(rest.map((m) => v[m.id].message), rest.map(y), OPTS);
  const devP = (m: Labeled) => predict(onRest, v[m.id].message);
  const devPitches = byLabel(dev, "pitch", devP);
  const devCustomers = byLabel(dev, "customer", devP);
  const md = [
    "# Classifier",
    "",
    `EmbeddingGemma 300M (q4) embeds each message with the prefix \`${PREFIX}\`; a logistic regression over the 768 numbers says how likely it is to be a pitch. Settings: ${JSON.stringify(OPTS)}.`,
    "",
    "## Forum posts, five-fold cross-validation",
    "",
    "Folds are split by forum topic, so no topic is ever scored by a model that saw any of it.",
    "",
    ...table(ms, p, BARS),
    "",
    `AUC ${forumAuc.toFixed(3)}. The settings above (and the page's bar of 0.8) were chosen on this same table, so read it as a ceiling rather than a promise.`,
    "",
    "## The 40 messages Gemma 3 1B was tried on",
    "",
    `Scored by a model trained on the other ${rest.length} forum posts: AUC ${auc(devPitches, devCustomers).toFixed(3)}; the most pitch-like customer scores ${Math.max(...devCustomers).toFixed(2)}, the least pitch-like pitch ${Math.min(...devPitches).toFixed(2)}. Gemma 3 1B's numbers on the same messages are in eval/EVAL.md. (Other posts from the same forum topics are in that training set.)`,
    "",
    "## My inbox, trained on the forum alone",
    "",
    "Cold emails to an app developer are written differently from replies on a forum, and a model that has only read the forum is less sure of them.",
    "",
    ...inboxTable(fromForum, BARS),
    "",
    "## My inbox, one sender at a time",
    "",
    "Each sender's messages are scored by a model trained on the forum and on every other sender's messages: the shipped model has seen pitches like these, but never from this sender.",
    "",
    ...inboxTable(heldOut, BARS),
    "",
    "The inbox holds no customers, so this half of the test can only show pitches being caught, not customers being kept.",
    "",
    "## Forum posts on the wrong side of 0.5",
    "",
    ...worst.map(({ m, p }) => `- **${m.label}**, P = ${p.toFixed(2)}: ${m.body.replace(/\s+/g, " ").slice(0, 220)}… ([post](${(m as Labeled & { url?: string }).url}))`),
    "",
  ].join("\n");
  writeFileSync(new URL("./CLASSIFIER.md", import.meta.url), md);
  console.log(md.split("## Forum posts on the wrong")[0]);
}

/** The shipped classifier: trained on every forum post and every inbox pitch. */
function fit(pitchAt: number, customerBelow: number) {
  const v = vectors();
  const all = [...loadMessages(), ...loadInbox()];
  const c = train(all.map((m) => v[m.id].message), all.map(y), OPTS);
  const out: Classifier = { model: EMBEDDER.id, prefix: PREFIX, bias: +c.bias.toFixed(6), weights: c.weights.map((w) => +w.toFixed(6)), pitchAt, customerBelow };
  writeFileSync(new URL("../public/classifier.json", import.meta.url), JSON.stringify(out));
  console.log(`public/classifier.json: trained on ${all.length} messages; pitch at P >= ${pitchAt}, customer below ${customerBelow}`);
}

const [cmd, a, b] = process.argv.slice(2);
if (cmd === "embed") await embed();
else if (cmd === "cv") cv();
else if (cmd === "fit") fit(Number(a ?? 0.8), Number(b ?? 0.5));
else console.log("usage: node eval/classify.ts embed | cv | fit [pitchAt] [customerBelow]");
