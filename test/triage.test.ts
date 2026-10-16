import { describe, expect, it } from "vitest";
import { clip, isGrounded, MAX_CHARS, parseAnswer, triage, type Generate } from "../src/triage.ts";

const pitch =
  "Hello, I came across your Shopify app and noticed it's still early. I help new apps get their first reviews. Would you be open to a short call this week?";
const customer = "Hi! The net terms option doesn't show at checkout for my wholesale customers. Can you help me fix this before Friday?";

describe("parseAnswer", () => {
  it("reads both lines, with bold markers and quotes", () => {
    expect(parseAnswer('**OFFER:** NONE\n**ASK:** "Can you help me fix this before Friday?"')).toEqual({
      offer: "NONE",
      ask: "Can you help me fix this before Friday?",
    });
  });
  it("leaves the offer null when the line is missing", () => {
    expect(parseAnswer("ASK: a call").offer).toBeNull();
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
  it("folds away a message whose offer is quoted from it", async () => {
    const t = await triage(scripted("OFFER: I help new apps get their first reviews.\nASK: Would you be open to a short call this week?"), pitch);
    expect(t).toMatchObject({ verdict: "pitch", quote: "I help new apps get their first reviews." });
    expect(t.attempts).toHaveLength(1);
  });
  it("retries once when the offer is not in the message, then accepts a grounded one", async () => {
    const t = await triage(scripted("OFFER: They sell reviews to apps.\nASK: a call", "OFFER: I help new apps get their first reviews.\nASK: a call"), pitch);
    expect(t.verdict).toBe("pitch");
    expect(t.attempts).toHaveLength(2);
  });
  it("keeps a message in front as unsure when the offer never holds up", async () => {
    const t = await triage(scripted("ASK: hm", "OFFER: something else entirely here today\nASK: x"), pitch);
    expect(t.verdict).toBe("unsure");
  });
  it("makes NONE a customer, quoting the ask when it is really there", async () => {
    const t = await triage(scripted("OFFER: NONE\nASK: Can you help me fix this before Friday?"), customer);
    expect(t).toMatchObject({ verdict: "customer", quote: "Can you help me fix this before Friday?" });
  });
});
