/// <reference lib="webworker" />
// Gemma runs here, off the page's main thread, through Transformers.js and ONNX Runtime Web:
// WebGPU when the browser has it, WebAssembly on the CPU when it does not. The weights come
// from the Hugging Face Hub once and then live in the browser's cache.

import { env, pipeline, type TextGenerationPipeline } from "@huggingface/transformers";
import { MODELS, type ChatMessage, type ModelKey } from "./triage.ts";

export type Device = "webgpu" | "wasm";
export type Dtype = "q4f16" | "q4";

export type ToWorker =
  | { type: "load"; model: ModelKey; device: Device; dtype: Dtype }
  | { type: "generate"; id: number; messages: ChatMessage[]; maxNewTokens: number };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; seconds: number }
  | { type: "generated"; id: number; text: string }
  | { type: "error"; id?: number; message: string };

env.allowLocalModels = false;

let generator: TextGenerationPipeline | null = null;

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
  post({ type: "ready", seconds: Math.round((performance.now() - started) / 1000) });
}

async function generate(id: number, messages: ChatMessage[], maxNewTokens: number) {
  if (!generator) throw new Error("The model is not loaded.");
  const out = await generator(messages, { max_new_tokens: maxNewTokens, do_sample: false });
  const reply = (out[0] as { generated_text: ChatMessage[] }).generated_text.at(-1);
  post({ type: "generated", id, text: typeof reply?.content === "string" ? reply.content : "" });
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") await load(msg.model, msg.device, msg.dtype);
    else await generate(msg.id, msg.messages, msg.maxNewTokens);
  } catch (err) {
    post({ type: "error", id: msg.type === "generate" ? msg.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
