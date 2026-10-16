/// <reference lib="webworker" />
// Gemma runs here, off the page's main thread, through Transformers.js and ONNX Runtime Web:
// WebGPU when the browser has it, WebAssembly on the CPU when it does not. The weights come
// from the Hugging Face Hub once and then live in the browser's cache.

import { env, pipeline, type TextGenerationPipeline } from "@huggingface/transformers";
import { makeScorer } from "./model.ts";
import { MODELS, type ChatMessage, type ModelKey, type Score } from "./triage.ts";

export type Device = "webgpu" | "wasm";
export type Dtype = "q4f16" | "q4";

export type ToWorker =
  | { type: "load"; model: ModelKey; device: Device; dtype: Dtype }
  | { type: "score"; id: number; messages: ChatMessage[] };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; seconds: number }
  | { type: "scored"; id: number; p: number }
  | { type: "error"; id?: number; message: string };

env.allowLocalModels = false;

let generator: TextGenerationPipeline | null = null;
let score: Score | null = null;

function post(message: FromWorker) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message);
}

async function load(model: ModelKey, device: Device, dtype: Dtype) {
  const started = performance.now();
  await generator?.dispose();
  generator = (await pipeline("text-generation", MODELS[model].id, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === "progress_total") post({ type: "progress", loaded: p.loaded, total: p.total });
    },
  })) as TextGenerationPipeline;
  score = makeScorer(generator);
  post({ type: "ready", seconds: Math.round((performance.now() - started) / 1000) });
}

async function rate(id: number, messages: ChatMessage[]) {
  if (!score) throw new Error("The model is not loaded.");
  post({ type: "scored", id, p: await score(messages) });
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") await load(msg.model, msg.device, msg.dtype);
    else await rate(msg.id, msg.messages);
  } catch (err) {
    post({ type: "error", id: msg.type === "score" ? msg.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
