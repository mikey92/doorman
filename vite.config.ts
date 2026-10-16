import { defineConfig, type Plugin } from "vitest/config";

// Transformers.js points ONNX Runtime at its WebAssembly build on jsDelivr (and caches it), so
// the 27 MB copy Vite finds inside onnxruntime-web is never fetched. Leave it out of dist:
// it is also over Cloudflare's 25 MiB limit for a single static file.
const dropUnusedWasm: Plugin = {
  name: "drop-unused-ort-wasm",
  generateBundle(_options, bundle) {
    for (const name of Object.keys(bundle)) if (name.endsWith(".wasm")) delete bundle[name];
  },
};

export default defineConfig({
  worker: { format: "es", plugins: () => [dropUnusedWasm] },
  plugins: [dropUnusedWasm],
  // Pre-bundling Transformers.js for the dev server breaks ONNX Runtime's own file loading.
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  build: { target: "es2022" },
  test: { include: ["test/**/*.test.ts"] },
});
