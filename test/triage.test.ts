import { describe, expect, it } from "vitest";
import { clip, isGrounded, MAX_CHARS, parseAnswer, triage, type Generate } from "../src/triage.ts";

const pitch =
  "Hello, I came across your Shopify app and noticed it's still early. I help new apps get their first reviews. Would you be open to a short call this week?";
const customer = "Hi! The net terms option doesn't show at checkout for my wholesale customers. Can you help me fix this before Friday?";

describe("parseAnswer", () => {
  it("reads both lines, with bold markers and quotes", () => {
    expect(parseAnswer('**WANTS:** "Can you help me fix this before Friday?"\n**FROM:** customer')).toEqual({
      who: "customer",
      quote: "Can you help me fix this before Friday?",
    });
  });
  it("leaves who empty when FROM is missing or something else", () => {
    expect(parseAnswer("WANTS: a call\nFROM: marketer").who).toBeNull();
  });
});

describe("isGrounded", () => {
  it("accepts a sentence copied from the message, ignoring case and punctuation", () => {
    expect(isGrounded("would you be open to a short call this week", pitch)).toBe(true);
  });
  it("accepts a quote with a word or two trimmed or added at an end", () => {
    expect(isGrounded("Sure: would you be open to a short call this week?", pitch)).toBe(true);
  });
  it("rejects a paraphrase", () => {
    expect(isGrounded("They want to sell me app reviews.", pitch)).toBe(false);
  });
  it("rejects a quote too short to show anything", () => {
    expect(isGrounded("Hello", pitch)).toBe(false);
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
  const scripted = (...answers: string[]): Generate => {
    let i = 0;
    return async () => answers[i++] ?? "";
  };
  it("folds away a grounded seller", async () => {
    const t = await triage(scripted("WANTS: Would you be open to a short call this week?\nFROM: seller"), pitch);
    expect(t.verdict).toBe("pitch");
    expect(t.attempts).toHaveLength(1);
  });
  it("retries once when the quote is not in the message, then accepts a grounded answer", async () => {
    const t = await triage(
      scripted("WANTS: They sell reviews.\nFROM: seller", "WANTS: I help new apps get their first reviews.\nFROM: seller"),
      pitch,
    );
    expect(t.verdict).toBe("pitch");
    expect(t.attempts).toHaveLength(2);
  });
  it("keeps a message in front as unsure when the answer never holds up", async () => {
    const t = await triage(scripted("FROM: seller", "WANTS: something else entirely here\nFROM: seller"), pitch);
    expect(t.verdict).toBe("unsure");
  });
  it("keeps a grounded customer as a customer", async () => {
    const t = await triage(scripted("WANTS: Can you help me fix this before Friday?\nFROM: customer"), customer);
    expect(t.verdict).toBe("customer");
  });
});
