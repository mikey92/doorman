/// <reference lib="webworker" />
// EmbeddingGemma runs here, off the page's main thread, through Transformers.js and ONNX Runtime
// Web: WebGPU when the browser has it, WebAssembly on the CPU when it does not. The weights come
// from the Hugging Face Hub once and then live in the browser's cache.

import { AutoModel, AutoTokenizer, env } from "@huggingface/transformers";
import { EMBEDDER, makeEmbedder, type Embed } from "./embed.ts";

export type Device = "webgpu" | "wasm";

export type ToWorker =
  | { type: "load"; device: Device }
  | { type: "embed"; id: number; texts: string[] };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; seconds: number }
  | { type: "embedded"; id: number; vectors: Float32Array[] }
  | { type: "error"; id?: number; message: string };

env.allowLocalModels = false;

let embed: Embed | null = null;

function post(message: FromWorker, transfer: Transferable[] = []) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message, transfer);
}

async function load(device: Device) {
  const started = performance.now();
  const progress_callback = (p: { status: string; loaded?: number; total?: number }) => {
    if (p.status === "progress_total") post({ type: "progress", loaded: p.loaded ?? 0, total: p.total ?? 1 });
  };
  const tokenizer = await AutoTokenizer.from_pretrained(EMBEDDER.id);
  // EmbeddingGemma's activations don't support fp16, so q4 weights with fp32 activations everywhere.
  const model = await AutoModel.from_pretrained(EMBEDDER.id, { device, dtype: "q4", progress_callback } as never);
  embed = makeEmbedder(tokenizer, model);
  post({ type: "ready", seconds: Math.round((performance.now() - started) / 1000) });
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") await load(msg.device);
    else {
      if (!embed) throw new Error("The model is not loaded.");
      const vectors = await embed(msg.texts);
      post({ type: "embedded", id: msg.id, vectors }, vectors.map((v) => v.buffer as ArrayBuffer));
    }
  } catch (err) {
    post({ type: "error", id: msg.type === "embed" ? msg.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
