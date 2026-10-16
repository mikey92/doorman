import "./styles.css";
import { parseFile, parsePasted, type Message } from "./parse.ts";
import { MODELS, triage, type ChatMessage, type ModelKey, type Triage } from "./triage.ts";
import type { Device, Dtype, FromWorker } from "./worker.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const filesInput = $<HTMLInputElement>("files");
const drop = $<HTMLLabelElement>("drop");
const paste = $<HTMLTextAreaElement>("paste");
const countLine = $<HTMLParagraphElement>("count");
const modelInput = $<HTMLSelectElement>("model");
const sortButton = $<HTMLButtonElement>("sort");
const statusLine = $<HTMLParagraphElement>("status");
const progress = $<HTMLProgressElement>("progress");
const results = $<HTMLElement>("results");

let messages: Message[] = [];
const verdicts = new Map<string, Triage>();

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
  sortButton.disabled = messages.length === 0;
  verdicts.clear();
  render();
}

async function addFiles(files: FileList | File[]) {
  const added: Message[] = [];
  for (const f of Array.from(files)) added.push(...parseFile(f.name, await f.text()));
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
  const m = parsePasted(paste.value, `pasted-${Date.now()}`);
  if (!m) return;
  paste.value = "";
  setMessages([...messages, m]);
});
$("clear").addEventListener("click", () => setMessages([]));
$("example").addEventListener("click", async () => {
  const res = await fetch("/example-inbox.json");
  const example = (await res.json()) as { messages: Message[]; verdicts?: Record<string, Triage> };
  setMessages(example.messages);
  // The example comes with the verdicts Gemma gave it, so it shows straight away.
  for (const [id, t] of Object.entries(example.verdicts ?? {})) verdicts.set(id, t);
  render();
  if (verdicts.size) say("Example inbox, already sorted by Gemma 3 1B. Press Sort Inbox to sort it again here.");
});

// ---------- where Gemma runs ----------

async function pickDevice(): Promise<{ device: Device; dtype: Dtype }> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter) return { device: "webgpu", dtype: adapter.features.has("shader-f16") ? "q4f16" : "q4" };
    } catch {
      /* fall through to the CPU */
    }
  }
  return { device: "wasm", dtype: "q4" };
}

let worker: Worker | null = null;
let loaded: string | null = null;
let nextId = 0;
const waiting = new Map<number, { resolve: (text: string) => void; reject: (err: Error) => void }>();
let onLoad: { resolve: (seconds: number) => void; reject: (err: Error) => void } | null = null;

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
    else if (msg.type === "generated") {
      waiting.get(msg.id)?.resolve(msg.text);
      waiting.delete(msg.id);
    } else if (msg.type === "error") {
      const err = new Error(msg.message);
      if (msg.id !== undefined) {
        waiting.get(msg.id)?.reject(err);
        waiting.delete(msg.id);
      } else onLoad?.reject(err);
    }
  };
  return worker;
}

async function loadModel(key: ModelKey, device: Device, dtype: Dtype) {
  const tag = `${key}/${device}/${dtype}`;
  if (loaded === tag) return;
  const w = getWorker();
  const seconds = await new Promise<number>((resolve, reject) => {
    onLoad = { resolve, reject };
    w.postMessage({ type: "load", model: key, device, dtype });
  });
  loaded = tag;
  progress.hidden = true;
  say(`${MODELS[key].label} ready in ${seconds} s.`);
}

function generate(messages: ChatMessage[], maxNewTokens: number): Promise<string> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    getWorker().postMessage({ type: "generate", id, messages, maxNewTokens });
  });
}

sortButton.addEventListener("click", async () => {
  const key = modelInput.value as ModelKey;
  sortButton.disabled = true;
  try {
    const { device, dtype } = await pickDevice();
    const label = MODELS[key].label;
    say(`Loading ${label} (${MODELS[key].mb[dtype]} MB the first time, then from your browser's cache)…`);
    await loadModel(key, device, dtype);
    verdicts.clear();
    render();
    const started = performance.now();
    for (const [i, m] of messages.entries()) {
      say(`${label} is reading on the ${device === "webgpu" ? "GPU" : "CPU"}: ${i + 1} of ${messages.length}…`);
      verdicts.set(m.id, await triage(generate, m.body));
      render();
    }
    const s = Math.round((performance.now() - started) / 1000);
    say(`Sorted ${messages.length} messages in ${s} s on the ${device === "webgpu" ? "GPU" : "CPU"}.`);
  } catch (err) {
    say(`Something went wrong: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
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
  li.append(head);
  if (t?.quote) {
    const q = document.createElement("blockquote");
    q.textContent = t.quote;
    li.append(q);
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
}
