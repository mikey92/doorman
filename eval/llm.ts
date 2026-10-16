// The part Gemma plays: for each sentence of a contact-form message, how likely is it that the
// sender is offering the business their own product or service? Gemma answers Yes or No, and we
// read how strongly it leans to "Yes" straight from its output, which only an open model lets
// you do. The sentence it rates highest is quoted on the card, so every verdict comes with the
// sender's own words. This file knows nothing about where the model runs: the browser worker and
// the Node eval both hand it a `score` function, so the eval measures exactly what the page does.

export interface ChatMessage { role: "user" | "assistant"; content: string }

/** log P("Yes") − log P("No") for the model's next token after these messages. */
export type Score = (messages: ChatMessage[]) => Promise<number>;

export const MODELS = {
  "gemma-3-1b": { id: "onnx-community/gemma-3-1b-it-ONNX-GQA", label: "Gemma 3 1B", mb: { q4f16: 763, q4: 859 } },
  "gemma-3-270m": { id: "onnx-community/gemma-3-270m-it-ONNX", label: "Gemma 3 270M", mb: { q4f16: 273, q4: 323 } },
} as const;

export type ModelKey = keyof typeof MODELS;

/**
 * Where a message lands. "customer" and "unsure" stay at the top of the inbox; only "pitch" is
 * folded away. Hiding a customer costs a sale and a pitch left in view costs a few seconds, so
 * the bar for "pitch" is set high and the band in between is "unsure".
 */
export type Verdict = "customer" | "pitch" | "unsure";

export interface Rated { sentence: string; /** Log-odds that this sentence offers something. */ score: number }

export interface Triage {
  verdict: Verdict;
  /** The sentence most likely to be an offer, quoted on the card. */
  quote: string;
  /** Highest log-odds of an offer over the message's sentences. */
  score: number;
  rated: Rated[];
}

/** At or above this log-odds, a message is folded away as a pitch. Set from the eval: see eval/EVAL.md. */
export const PITCH_AT = 9;
/** Below this, a message is a customer; in between, unsure. */
export const CUSTOMER_BELOW = 6;
/** Sentences rated per message: offers come early, and a long message is read in full anyway. */
export const MAX_SENTENCES = 12;
/** The model reads the start of a message; intent is almost always stated in the first lines. */
export const MAX_CHARS = 1200;

/** The start of the message, cut at a sentence or line end before MAX_CHARS. */
export function clip(body: string): string {
  const text = body.trim();
  if (text.length <= MAX_CHARS) return text;
  const head = text.slice(0, MAX_CHARS);
  const cut = Math.max(head.lastIndexOf(". "), head.lastIndexOf("\n"), head.lastIndexOf("? "), head.lastIndexOf("! "));
  return (cut > MAX_CHARS / 2 ? head.slice(0, cut + 1) : head).trim();
}

/** Sentences worth asking about: split on end punctuation and line breaks, three words or more. */
export function sentences(body: string): string[] {
  return clip(body)
    .replace(/https?:\/\/\S+/g, (u) => u.replace(/[.?!]/g, "·"))
    .split(/(?<=[.?!])\s+|\n+/)
    .map((s) => s.replace(/·/g, ".").replace(/^[-•*\s]+/, "").trim())
    .filter((s) => s.split(/\s+/).length >= 3);
}

function question(body: string, sentence: string): string {
  return [
    "This message came in through the contact form of a small online business (an app or a shop).",
    "",
    "Message:",
    '"""',
    body,
    '"""',
    "",
    `Sentence: "${sentence}"`,
    "",
    "In this sentence, is the sender offering the business their OWN product, app, service, agency or help?",
    "Answer Yes or No.",
  ].join("\n");
}

// Worked examples written for this file (none is in the eval set): an offer, a request from a
// customer, and a seller's friendly opener, which is not itself the offer.
const PITCH_EXAMPLE =
  "Hi there! I came across your app and love what you're building. I help Shopify apps get their first 50 reviews. Would you be open to a quick call this week?";
const CUSTOMER_EXAMPLE =
  "Hello, I installed your app yesterday but the discount isn't showing on my product pages. I'm on the Dawn theme. Could you take a look?";
const EXAMPLES: [string, string, "Yes" | "No"][] = [
  [PITCH_EXAMPLE, "I help Shopify apps get their first 50 reviews.", "Yes"],
  [CUSTOMER_EXAMPLE, "Could you take a look?", "No"],
  [PITCH_EXAMPLE, "Hi there! I came across your app and love what you're building.", "No"],
];

export function questionMessages(body: string, sentence: string): ChatMessage[] {
  return [
    ...EXAMPLES.flatMap(([b, s, a]): ChatMessage[] => [
      { role: "user", content: question(b, s) },
      { role: "assistant", content: a },
    ]),
    { role: "user", content: question(clip(body), sentence) },
  ];
}

export function verdictFor(score: number): Verdict {
  return score >= PITCH_AT ? "pitch" : score < CUSTOMER_BELOW ? "customer" : "unsure";
}

/** Rates each sentence; the message is as much a pitch as its most pitch-like sentence. */
export async function triage(score: Score, body: string): Promise<Triage> {
  const list = sentences(body).slice(0, MAX_SENTENCES);
  const rated: Rated[] = [];
  for (const s of list) rated.push({ sentence: s, score: await score(questionMessages(body, s)) });
  const top = rated.reduce<Rated | null>((a, b) => (a && a.score >= b.score ? a : b), null);
  const best = top?.score ?? -Infinity;
  return { verdict: verdictFor(best), quote: top && best >= CUSTOMER_BELOW ? top.sentence : "", score: best, rated };
}
