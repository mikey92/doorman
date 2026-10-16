// Doorman's verdict on one message. EmbeddingGemma turns the message, and each of its sentences,
// into 768 numbers; a logistic regression trained on hand-labelled posts (eval/classify.ts) says
// how likely each is to be a pitch. The message decides the verdict; its most pitch-like sentence
// is quoted on the card, so a folded-away message always shows the words that gave it away.
// This file knows nothing about where the model runs: the browser worker and the Node eval both
// hand it an `embed` function.

import type { Embed } from "./embed.ts";
import { predict, type Classifier } from "./linear.ts";

/**
 * Where a message lands. "customer" and "unsure" stay at the top of the inbox; only "pitch" is
 * folded away. Hiding a customer costs a sale and a pitch left in view costs a few seconds, so
 * the bar for "pitch" is set high and the band below it is "unsure".
 */
export type Verdict = "customer" | "pitch" | "unsure";

export interface Rated { sentence: string; p: number }

export interface Triage {
  verdict: Verdict;
  /** P(pitch) for the whole message. */
  p: number;
  /** The most pitch-like sentence, quoted on the card unless the message is a customer's. */
  quote: string;
  rated: Rated[];
}

/** Sentences are taken from the start of a message; intent is almost always stated in the first lines. */
export const MAX_CHARS = 1500;
/** Sentences rated for the quote. */
export const MAX_SENTENCES = 16;

/** The start of the message, cut at a sentence or line end before MAX_CHARS. */
export function clip(body: string): string {
  const text = body.trim();
  if (text.length <= MAX_CHARS) return text;
  const head = text.slice(0, MAX_CHARS);
  const cut = Math.max(head.lastIndexOf(". "), head.lastIndexOf("\n"), head.lastIndexOf("? "), head.lastIndexOf("! "));
  return (cut > MAX_CHARS / 2 ? head.slice(0, cut + 1) : head).trim();
}

/** Sentences worth rating: split on end punctuation and line breaks, three words or more. */
export function sentences(body: string): string[] {
  return clip(body)
    .replace(/https?:\/\/\S+/g, (u) => u.replace(/[.?!]/g, "·"))
    .split(/(?<=[.?!])\s+|\n+/)
    .map((s) => s.replace(/·/g, ".").replace(/^[-•*\s]+/, "").trim())
    .filter((s) => s.split(/\s+/).length >= 3)
    .slice(0, MAX_SENTENCES);
}

export function verdictFor(c: Pick<Classifier, "pitchAt" | "customerBelow">, p: number): Verdict {
  return p >= c.pitchAt ? "pitch" : p < c.customerBelow ? "customer" : "unsure";
}

export async function triage(embed: Embed, c: Classifier, body: string): Promise<Triage> {
  const ss = sentences(body);
  // The whole message goes in as is; the embedder keeps its first MAX_TOKENS tokens.
  const [whole, ...parts] = await embed([body.trim(), ...ss]);
  const p = predict(c, whole);
  const rated = ss.map((sentence, i) => ({ sentence, p: predict(c, parts[i]) }));
  const verdict = verdictFor(c, p);
  const top = rated.reduce<Rated | null>((a, b) => (a && a.p >= b.p ? a : b), null);
  return { verdict, p, quote: verdict !== "customer" && top ? top.sentence : "", rated };
}
