// EmbeddingGemma turns a message, or one sentence of it, into 768 numbers. Shared by the
// browser worker and the Node eval so both embed text exactly the same way.

export const EMBEDDER = { id: "onnx-community/embeddinggemma-300m-ONNX", label: "EmbeddingGemma 300M", mb: { q4: 197 } } as const;

/** EmbeddingGemma's prompt for classification tasks. */
export const PREFIX = "task: classification | query: ";

export type Embed = (texts: string[]) => Promise<Float32Array[]>;

interface Tokenizer { (texts: string[], opts: { padding: boolean; truncation: boolean; max_length: number }): unknown }
interface Model { (inputs: unknown): Promise<{ sentence_embedding: { dims: number[]; data: Float32Array } }> }

/** The text the model sees: the prefix, then at most this many tokens of the message. */
export const MAX_TOKENS = 512;

export function makeEmbedder(tokenizer: unknown, model: unknown): Embed {
  const tok = tokenizer as Tokenizer;
  const run = model as Model;
  return async (texts: string[]) => {
    const out: Float32Array[] = [];
    // Small batches keep memory flat on long messages.
    for (let i = 0; i < texts.length; i += 8) {
      const batch = texts.slice(i, i + 8).map((t) => PREFIX + t);
      const inputs = tok(batch, { padding: true, truncation: true, max_length: MAX_TOKENS });
      const { sentence_embedding } = await run(inputs);
      const [n, d] = sentence_embedding.dims;
      for (let k = 0; k < n; k++) out.push(sentence_embedding.data.slice(k * d, (k + 1) * d));
    }
    return out;
  };
}
