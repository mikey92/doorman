import { describe, expect, it } from "vitest";
import { clip, CUSTOMER_BELOW, MAX_CHARS, PITCH_AT, questionMessages, sentences, triage, verdictFor, type Score } from "../src/triage.ts";

const pitch =
  "Hello, I came across your Shopify app and noticed it's still early. I help new apps get their first reviews. Would you be open to a short call this week?";
const customer = "Hi! The net terms option doesn't show at checkout for my wholesale customers. Can you help me fix this before Friday?";

describe("sentences", () => {
  it("splits on end punctuation and lines, keeps URLs whole, and drops fragments", () => {
    expect(sentences("Hi!\nI help apps grow. See https://example.com/a.b for more.\n- Thanks a lot\nBest")).toEqual([
      "I help apps grow.",
      "See https://example.com/a.b for more.",
      "Thanks a lot",
    ]);
  });
});

describe("clip", () => {
  it("keeps short messages whole and cuts long ones at a sentence end", () => {
    expect(clip(customer)).toBe(customer);
    const long = "This is a sentence. ".repeat(100);
    const c = clip(long);
    expect(c.length).toBeLessThanOrEqual(MAX_CHARS);
    expect(c.endsWith(".")).toBe(true);
  });
});

describe("triage", () => {
  // A stand-in model: P(offer) is high for sentences containing "I help".
  const fake: Score = async (messages) => (messages.at(-1)!.content.includes('Sentence: "I help') ? 11 : 1);
  it("folds away a message whose most pitch-like sentence clears the bar, quoting it", async () => {
    const t = await triage(fake, pitch);
    expect(t).toMatchObject({ verdict: "pitch", quote: "I help new apps get their first reviews." });
    expect(t.rated).toHaveLength(3);
  });
  it("keeps a message with no offer as a customer, with nothing quoted", async () => {
    expect(await triage(fake, customer)).toMatchObject({ verdict: "customer", quote: "" });
  });
  it("puts the band between the two bars in front as unsure", () => {
    expect(verdictFor(PITCH_AT)).toBe("pitch");
    expect(verdictFor(CUSTOMER_BELOW)).toBe("unsure");
    expect(verdictFor(CUSTOMER_BELOW - 0.5)).toBe("customer");
  });
  it("asks about the sentence in the context of the whole message", () => {
    const last = questionMessages(customer, "Can you help me fix this before Friday?").at(-1)!.content;
    expect(last).toContain(customer);
    expect(last).toContain('Sentence: "Can you help me fix this before Friday?"');
  });
});
