// The part Gemma plays: read one message from a contact form and say who wrote it, a customer or
// someone selling something, quoting the sentence that shows what they want. This file knows
// nothing about where the model runs; the browser worker and the Node eval both hand it a
// `generate` function, so the eval measures exactly what the page does.

export interface ChatMessage { role: "user" | "assistant"; content: string }

/** Runs the model on a chat and returns the text of its reply. */
export type Generate = (messages: ChatMessage[], maxNewTokens: number) => Promise<string>;

export const MODELS = {
  "gemma-3-1b": { id: "onnx-community/gemma-3-1b-it-ONNX-GQA", label: "Gemma 3 1B", mb: { q4f16: 763, q4: 859 } },
  "gemma-3-270m": { id: "onnx-community/gemma-3-270m-it-ONNX", label: "Gemma 3 270M", mb: { q4f16: 273, q4: 323 } },
} as const;

export type ModelKey = keyof typeof MODELS;

/**
 * Where a message lands. "customer" and "unsure" stay at the top of the inbox; only "pitch" is
 * folded away, and only when the model quotes a sentence, really in the message, where the
 * sender offers their own product or service.
 */
export type Verdict = "customer" | "pitch" | "unsure";

export interface Attempt { raw: string; offer: string | null; ask: string; grounded: boolean; ms: number }

export interface Triage {
  verdict: Verdict;
  /** For a pitch, the sentence that offers something; for a customer, the one that asks. */
  quote: string;
  attempts: Attempt[];
}

const MAX_NEW_TOKENS = 96;
/** The model reads the start of a message; intent is almost always stated in the first lines. */
export const MAX_CHARS = 1200;
/** A quote shorter than this shows nothing ("a free app"). */
export const MIN_QUOTE_WORDS = 5;

function request(body: string): string {
  return [
    "This message came in through the contact form of a small online business (an app or a shop).",
    "",
    "Message:",
    '"""',
    body,
    '"""',
    "",
    "Answer in exactly two lines:",
    "OFFER: copy, word for word, the sentence where the sender offers the business their OWN product, app, service or help. If the sender offers nothing of their own, write NONE.",
    "ASK: copy, word for word, the sentence that says what the sender wants.",
  ].join("\n");
}

// Three worked examples written for this file (none is in the eval set): a customer, a seller,
// and a seller who opens with a question, as many do.
const EXAMPLES: [string, string][] = [
  [
    "Hi, I installed your app yesterday but the discount isn't showing on my product pages. I'm on the Dawn theme. Could you take a look? Thanks, Maria",
    "OFFER: NONE\nASK: Could you take a look?",
  ],
  [
    "Hello! I came across your store and love what you're building. I help Shopify brands get more traffic with SEO and backlinks. Would you be open to a quick call this week?",
    "OFFER: I help Shopify brands get more traffic with SEO and backlinks.\nASK: Would you be open to a quick call this week?",
  ],
  [
    "Are you still looking for a way to get more reviews? Our app sends review requests by SMS and it's free for the first 100 orders. Happy to set it up for you.",
    "OFFER: Our app sends review requests by SMS and it's free for the first 100 orders.\nASK: Happy to set it up for you.",
  ],
];

export function triageMessages(body: string): ChatMessage[] {
  return [
    ...EXAMPLES.flatMap(([b, answer]): ChatMessage[] => [
      { role: "user", content: request(b) },
      { role: "assistant", content: answer },
    ]),
    { role: "user", content: request(clip(body)) },
  ];
}

/** The start of the message, cut at a sentence or line end before MAX_CHARS. */
export function clip(body: string): string {
  const text = body.trim();
  if (text.length <= MAX_CHARS) return text;
  const head = text.slice(0, MAX_CHARS);
  const cut = Math.max(head.lastIndexOf(". "), head.lastIndexOf("\n"), head.lastIndexOf("? "), head.lastIndexOf("! "));
  return (cut > MAX_CHARS / 2 ? head.slice(0, cut + 1) : head).trim();
}

const unquote = (x: string) => x.trim().replace(/^["'“”‘’]+|["'“”‘’]+$/g, "").trim();

/** Pulls OFFER and ASK out of a reply; tolerant of bold markers, quotes and case. OFFER is null when missing. */
export function parseAnswer(raw: string): { offer: string | null; ask: string } {
  const text = raw.replace(/\*\*/g, "");
  const offer = /OFFER\s*:\s*(.+)/i.exec(text)?.[1];
  const ask = /ASK\s*:\s*(.+)/i.exec(text)?.[1] ?? "";
  return { offer: offer === undefined ? null : unquote(offer), ask: unquote(ask) };
}

export const isNone = (offer: string) => /^(none|nothing|n\/a)\b/i.test(offer);

/** Lowercase words, apostrophes and punctuation dropped: what "word for word" is compared on. */
export function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[’']/g, "")
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}

/**
 * Is the quote really in the message? Its words must appear there as one unbroken run, allowing
 * the model to drop a few words from either end.
 */
export function isGrounded(quote: string, body: string): boolean {
  const q = words(quote);
  if (q.length < MIN_QUOTE_WORDS) return false;
  const b = ` ${words(body).join(" ")} `;
  if (b.includes(` ${q.join(" ")} `)) return true;
  // The longest run of the quote found in the message must cover nearly all of it.
  let best = 0;
  for (let i = 0; i < q.length; i++) {
    for (let j = q.length; j > i + best; j--) {
      if (b.includes(` ${q.slice(i, j).join(" ")} `)) {
        best = j - i;
        break;
      }
    }
  }
  return best >= MIN_QUOTE_WORDS && best >= q.length * 0.8;
}

function retryRequest(offer: string | null): string {
  if (offer === null) return "Answer in exactly two lines, OFFER: and ASK:, as above.";
  return (
    `"${offer}" is not a sentence from the message. ` +
    "Copy the sentence word for word from the message, or write NONE if the sender offers nothing of their own. Answer in exactly two lines."
  );
}

/**
 * Ask once; if the answer is malformed or its OFFER isn't in the message, ask once more with the
 * problem named. A grounded OFFER folds the message away as a pitch; NONE makes it a customer;
 * anything else stays at the top as unsure.
 */
export async function triage(generate: Generate, body: string): Promise<Triage> {
  let messages = triageMessages(body);
  const attempts: Attempt[] = [];
  for (let round = 0; round < 2; round++) {
    const started = Date.now();
    const raw = await generate(messages, MAX_NEW_TOKENS);
    const { offer, ask } = parseAnswer(raw);
    const none = offer !== null && isNone(offer);
    const grounded = offer !== null && !none && isGrounded(offer, body);
    attempts.push({ raw, offer, ask, grounded, ms: Date.now() - started });
    if (grounded) return { verdict: "pitch", quote: offer!, attempts };
    if (none) return { verdict: "customer", quote: isGrounded(ask, body) ? ask : "", attempts };
    messages = [
      ...messages,
      { role: "assistant", content: raw.trim() || "(no answer)" },
      { role: "user", content: retryRequest(offer) },
    ];
  }
  return { verdict: "unsure", quote: "", attempts };
}
