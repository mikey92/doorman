import "./styles.css";
import { parseFile, parsePasted, readBinary, type Message } from "./parse.ts";
import { EMBEDDER, type Embed } from "./embed.ts";
import type { Classifier } from "./linear.ts";
import { triage, type Triage } from "./triage.ts";
import type { Device, FromWorker } from "./worker.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const filesInput = $<HTMLInputElement>("files");
const drop = $<HTMLLabelElement>("drop");
const paste = $<HTMLTextAreaElement>("paste");
const countLine = $<HTMLParagraphElement>("count");
const sortButton = $<HTMLButtonElement>("sort");
const statusLine = $<HTMLParagraphElement>("status");
const progress = $<HTMLProgressElement>("progress");
const results = $<HTMLElement>("results");

let messages: Message[] = [];
const verdicts = new Map<string, Triage>();
/** Bumped whenever the list changes, so a sort in progress stops instead of sorting a stale list. */
let generation = 0;
let sorting = false;
let nextMessage = 0;

function say(text: string) {
  statusLine.textContent = text;
}

function setMessages(next: Message[]) {
  // The same message twice (a thread exported twice, a file dropped again) is read once.
  const seen = new Set<string>();
  messages = next.filter((m) => {
    const key = m.body.slice(0, 300);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  countLine.textContent = messages.length ? `${messages.length} message${messages.length === 1 ? "" : "s"} loaded.` : "No messages yet.";
  // Messages already sorted keep their verdicts; only the new ones wait for the next sort.
  const kept = new Set(messages.map((m) => m.id));
  for (const id of [...verdicts.keys()]) if (!kept.has(id)) verdicts.delete(id);
  generation++;
  sortButton.disabled = sorting || messages.length === 0;
  render();
}

async function addFiles(files: FileList | File[]) {
  const added: Message[] = [];
  for (const f of Array.from(files)) {
    // Two files can share a name (two exports of "message.eml"); ids must not.
    for (const m of parseFile(f.name, await readBinary(f))) added.push({ ...m, id: `m${nextMessage++}` });
  }
  if (!added.length) say("No readable messages in those files.");
  setMessages([...messages, ...added]);
}

filesInput.addEventListener("change", () => {
  if (filesInput.files) void addFiles(filesInput.files);
  filesInput.value = "";
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer?.files.length) void addFiles(e.dataTransfer.files);
});
$("add-paste").addEventListener("click", () => {
  const m = parsePasted(paste.value, `m${nextMessage++}`);
  if (!m) return;
  paste.value = "";
  setMessages([...messages, m]);
});
$("clear").addEventListener("click", () => setMessages([]));
$("example").addEventListener("click", async () => {
  const res = await fetch("/example-inbox.json");
  const example = (await res.json()) as { messages: Message[] };
  setMessages(example.messages);
});

// ---------- where EmbeddingGemma runs ----------

async function pickDevice(): Promise<Device> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu;
  if (gpu) {
    try {
      if (await gpu.requestAdapter()) return "webgpu";
    } catch {
      /* fall through to the CPU */
    }
  }
  return "wasm";
}

let worker: Worker | null = null;
let loaded: Device | null = null;
let nextId = 0;
const waiting = new Map<number, { resolve: (v: Float32Array[]) => void; reject: (err: Error) => void }>();
let onLoad: { resolve: (seconds: number) => void; reject: (err: Error) => void } | null = null;
let classifier: Classifier | null = null;

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<FromWorker>) => {
    const msg = event.data;
    if (msg.type === "progress") {
      progress.hidden = false;
      progress.max = msg.total;
      progress.value = msg.loaded;
    } else if (msg.type === "ready") onLoad?.resolve(msg.seconds);
    else if (msg.type === "embedded") {
      waiting.get(msg.id)?.resolve(msg.vectors);
      waiting.delete(msg.id);
    } else if (msg.type === "error") {
      const err = new Error(msg.message);
      if (msg.id !== undefined) {
        waiting.get(msg.id)?.reject(err);
        waiting.delete(msg.id);
      } else onLoad?.reject(err);
    }
  };
  worker.onerror = (event) => {
    const err = new Error(event.message || "The model worker stopped.");
    for (const w of waiting.values()) w.reject(err);
    waiting.clear();
    onLoad?.reject(err);
    worker?.terminate();
    worker = null;
    loaded = null;
  };
  return worker;
}

async function loadModel(device: Device) {
  if (loaded === device) return;
  const w = getWorker();
  const seconds = await new Promise<number>((resolve, reject) => {
    onLoad = { resolve, reject };
    w.postMessage({ type: "load", device });
  });
  loaded = device;
  progress.hidden = true;
  say(`${EMBEDDER.label} ready in ${seconds} s.`);
}

const embed: Embed = (texts) => {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    getWorker().postMessage({ type: "embed", id, texts });
  });
};

sortButton.addEventListener("click", async () => {
  sorting = true;
  sortButton.disabled = true;
  const mine = generation;
  try {
    classifier ??= (await (await fetch("/classifier.json")).json()) as Classifier;
    let device = await pickDevice();
    say(`Loading ${EMBEDDER.label} (${EMBEDDER.mb.q4} MB the first time, then from your browser's cache)…`);
    const toCpu = async (err: unknown) => {
      // A GPU that WebGPU reports but ONNX Runtime can't use: the CPU still works.
      if (device !== "webgpu") throw err;
      device = "wasm";
      say(`Running on the GPU failed (${err instanceof Error ? err.message : String(err)}); trying the CPU instead…`);
      worker?.terminate();
      worker = null;
      loaded = null;
      await loadModel(device);
    };
    try {
      await loadModel(device);
    } catch (err) {
      await toCpu(err);
    }
    const list = messages;
    // New messages only; if everything has a verdict already, sort it all again.
    let todo = list.filter((m) => !verdicts.has(m.id));
    if (!todo.length) {
      verdicts.clear();
      render();
      todo = list;
    }
    const started = performance.now();
    for (const [i, m] of todo.entries()) {
      if (generation !== mine) break;
      say(`Reading on the ${device === "webgpu" ? "GPU" : "CPU"}: ${i + 1} of ${todo.length}…`);
      let t: Triage;
      try {
        t = await triage(embed, classifier, m.body);
      } catch (err) {
        await toCpu(err);
        t = await triage(embed, classifier, m.body);
      }
      if (generation !== mine) break;
      verdicts.set(m.id, t);
      render();
    }
    const s = ((performance.now() - started) / 1000).toFixed(1);
    if (generation !== mine) say("The list changed, so sorting stopped. Press Sort Inbox to sort what's new.");
    else say(`Sorted ${todo.length} message${todo.length === 1 ? "" : "s"} in ${s} s on the ${device === "webgpu" ? "GPU" : "CPU"}.`);
  } catch (err) {
    say(`Something went wrong: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    sorting = false;
    sortButton.disabled = messages.length === 0;
  }
});

// ---------- the sorted inbox ----------

function highlight(body: string, quote: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  const i = quote ? body.toLowerCase().indexOf(quote.toLowerCase().replace(/[.?!]+$/, "")) : -1;
  if (i < 0) {
    frag.append(body);
    return frag;
  }
  const end = i + quote.replace(/[.?!]+$/, "").length;
  const mark = document.createElement("mark");
  mark.textContent = body.slice(i, end);
  frag.append(body.slice(0, i), mark, body.slice(end));
  return frag;
}

/** The first couple of lines, so a card can be read without opening it. */
function preview(body: string): string {
  const flat = body.replace(/\s+/g, " ").trim();
  return flat.length > 220 ? `${flat.slice(0, 220).replace(/\s+\S*$/, "")}…` : flat;
}

function card(m: Message, t: Triage | undefined): HTMLLIElement {
  const li = document.createElement("li");
  li.className = "card";
  const head = document.createElement("div");
  head.className = "card-head";
  const who = document.createElement("strong");
  who.textContent = m.from || "Pasted message";
  const subject = document.createElement("span");
  subject.textContent = m.subject;
  head.append(who, subject);
  if (t) {
    const p = document.createElement("span");
    p.className = "p";
    p.textContent = `P(pitch) ${t.p.toFixed(2)}`;
    p.title = "How likely the classifier thinks this message is a sales pitch";
    head.append(p);
  }
  li.append(head);
  if (t?.quote) {
    const q = document.createElement("blockquote");
    const label = document.createElement("span");
    label.className = "q-label";
    label.textContent = t.verdict === "pitch" ? "Gave it away" : "Most pitch-like sentence";
    q.append(label, document.createElement("br"), t.quote);
    li.append(q);
  } else {
    const p = document.createElement("p");
    p.className = "preview";
    p.textContent = preview(m.body);
    li.append(p);
  }
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Whole message";
  const body = document.createElement("p");
  body.className = "body";
  body.append(highlight(m.body, t?.quote ?? ""));
  details.append(summary, body);
  li.append(details);
  return li;
}

function render() {
  const lists = { customer: $("customer"), unsure: $("unsure"), pitch: $("pitch") };
  for (const l of Object.values(lists)) l.replaceChildren();
  const counts = { customer: 0, unsure: 0, pitch: 0 };
  for (const m of messages) {
    const t = verdicts.get(m.id);
    if (!t) continue;
    counts[t.verdict]++;
    lists[t.verdict].append(card(m, t));
  }
  results.hidden = verdicts.size === 0;
  $("n-customer").textContent = String(counts.customer);
  $("n-unsure").textContent = String(counts.unsure);
  $("n-pitch").textContent = String(counts.pitch);
  $("unsure-group").hidden = counts.unsure === 0;
}

/** A CSV cell: quoted, with quotes doubled, and a leading = + - @ defused for spreadsheets. */
export function csvCell(v: string): string {
  const safe = /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
  return `"${safe.replace(/"/g, '""')}"`;
}

$("csv").addEventListener("click", () => {
  const rows = [["verdict", "p_pitch", "from", "subject", "date", "quote", "message"]];
  for (const m of messages) {
    const t = verdicts.get(m.id);
    if (t) rows.push([t.verdict, t.p.toFixed(3), m.from, m.subject, m.date, t.quote, m.body]);
  }
  const blob = new Blob([rows.map((r) => r.map(csvCell).join(",")).join("\r\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "doorman-verdicts.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});
