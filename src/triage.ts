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

export type Who = "customer" | "seller";

/**
 * Where a message lands. "customer" and "unsure" stay at the top of the inbox; only "pitch" is
 * folded away, and only when the model says seller AND quotes a sentence that is really there.
 */
export type Verdict = "customer" | "pitch" | "unsure";

export interface Attempt { raw: string; who: Who | null; quote: string; grounded: boolean; ms: number }

export interface Triage {
  verdict: Verdict;
  /** The sender's own sentence that shows what they want, when the model found one. */
  quote: string;
  attempts: Attempt[];
}

const MAX_NEW_TOKENS = 80;
/** The model reads the start of a message; intent is almost always stated in the first lines. */
export const MAX_CHARS = 1200;

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
    "WANTS: copy, word for word, the one sentence from the message that shows what the sender wants",
    "FROM: customer or seller",
    "",
    "customer = uses, or wants to use or buy, this business's product, and needs something from it.",
    "seller = offers the business their own service, product or partnership (marketing, reviews, installs, design, development, funding).",
  ].join("\n");
}

// Two worked examples, one each way, written for this file: none of them is in the eval set.
const EXAMPLES: [string, string][] = [
  [
    "Hi, I installed your app yesterday but the discount isn't showing on my product pages. I'm on the Dawn theme. Could you take a look? Thanks, Maria",
    "WANTS: Could you take a look?\nFROM: customer",
  ],
  [
    "Hello! I came across your store and love what you're building. I help Shopify brands get more traffic with SEO and backlinks. Would you be open to a quick call this week to discuss how I can help you grow?",
    "WANTS: Would you be open to a quick call this week to discuss how I can help you grow?\nFROM: seller",
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

/** Pulls WANTS and FROM out of a reply; tolerant of bold markers, quotes and case. */
export function parseAnswer(raw: string): { who: Who | null; quote: string } {
  const text = raw.replace(/\*\*/g, "");
  const wants = /WANTS\s*:\s*(.+)/i.exec(text)?.[1]?.trim() ?? "";
  const from = /FROM\s*:\s*\W*(customer|seller)/i.exec(text)?.[1]?.toLowerCase() as Who | undefined;
  return { who: from ?? null, quote: wants.replace(/^["'“”‘’]+|["'“”‘’]+$/g, "").trim() };
}

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
 * the model to drop a few words from either end. A quote of fewer than four words shows nothing.
 */
export function isGrounded(quote: string, body: string): boolean {
  const q = words(quote);
  if (q.length < 4) return false;
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
  return best >= 4 && best >= q.length * 0.8;
}

function retryRequest(parsed: { who: Who | null; quote: string }): string {
  if (!parsed.quote || !parsed.who) return "Answer in exactly two lines, WANTS: and FROM:, as above.";
  return (
    `"${parsed.quote}" is not a sentence from the message. ` +
    "Copy the sentence word for word from the message, then FROM: customer or seller. Answer in exactly two lines."
  );
}

/**
 * Ask once; if the answer is malformed or its quote isn't in the message, ask once more with the
 * problem named. A message is folded away as a pitch only on a grounded "seller"; a grounded
 * "customer" is a customer; anything else stays at the top as unsure.
 */
export async function triage(generate: Generate, body: string): Promise<Triage> {
  let messages = triageMessages(body);
  const attempts: Attempt[] = [];
  for (let round = 0; round < 2; round++) {
    const started = Date.now();
    const raw = await generate(messages, MAX_NEW_TOKENS);
    const parsed = parseAnswer(raw);
    const grounded = isGrounded(parsed.quote, body);
    attempts.push({ raw, who: parsed.who, quote: parsed.quote, grounded, ms: Date.now() - started });
    if (parsed.who && grounded) {
      return { verdict: parsed.who === "seller" ? "pitch" : "customer", quote: parsed.quote, attempts };
    }
    messages = [
      ...messages,
      { role: "assistant", content: raw.trim() || "(no answer)" },
      { role: "user", content: retryRequest(parsed) },
    ];
  }
  return { verdict: "unsure", quote: "", attempts };
}
