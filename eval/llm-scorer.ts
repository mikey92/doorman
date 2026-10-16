// One question, one forward pass: how much more likely is Gemma to answer "Yes" than "No"?
// Shared by the browser worker and the Node eval, so both read the model the same way.

import type { TextGenerationPipeline } from "@huggingface/transformers";
import type { ChatMessage, Score } from "./llm.ts";

export function makeScorer(generator: TextGenerationPipeline): Score {
  const { tokenizer, model } = generator;
  const id = (word: string) => {
    const ids = tokenizer.encode(word, { add_special_tokens: false });
    if (ids.length !== 1) throw new Error(`"${word}" is not one token for this model`);
    return ids[0];
  };
  const yes = id("Yes");
  const no = id("No");
  return async (messages: ChatMessage[]) => {
    const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true }) as Record<string, unknown>;
    let pair: [number, number] | null = null;
    // The next-token logits pass through every logits processor before a token is picked;
    // this one only reads them. Greedy, one token: nothing is sampled.
    const peek = (_ids: unknown, logits: { data: ArrayLike<number> }) => {
      pair ??= [logits.data[yes], logits.data[no]];
      return logits;
    };
    await model.generate({ ...inputs, max_new_tokens: 1, do_sample: false, logits_processor: [peek] } as never);
    if (!pair) throw new Error("No logits came back from the model.");
    const [y, n] = pair as [number, number];
    // Log-odds of "Yes" over "No". A small model leans to Yes; its probabilities pile up near 1,
    // where log-odds still tell 0.999 from 0.99999.
    return y - n;
  };
}
