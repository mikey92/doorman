import { describe, expect, it } from "vitest";
import type { Embed } from "../src/embed.ts";
import { predict, train, type Classifier } from "../src/linear.ts";
import { sentences, triage, verdictFor } from "../src/triage.ts";

// A stand-in embedder: dimension 0 says "offers something", dimension 1 is constant.
const fake: Embed = async (texts) => texts.map((t) => Float32Array.from([/\bI help\b|our app/i.test(t) ? 1 : 0, 1]));
const c: Classifier = { model: "test", prefix: "", bias: -3, weights: [8, 0], pitchAt: 0.9, customerBelow: 0.5 };

describe("triage", () => {
  it("folds away a pitch and quotes its most pitch-like sentence", async () => {
    const t = await triage(fake, c, "Hello there friend. I help new apps get their first reviews. Can we talk this week?");
    expect(t).toMatchObject({ verdict: "pitch", quote: "I help new apps get their first reviews." });
  });
  it("keeps a customer in front with nothing quoted", async () => {
    expect(await triage(fake, c, "The net terms option is missing at checkout. Can you fix it?")).toMatchObject({ verdict: "customer", quote: "" });
  });
  it("calls the band between the bars unsure", () => {
    expect(verdictFor(c, 0.9)).toBe("pitch");
    expect(verdictFor(c, 0.5)).toBe("unsure");
    expect(verdictFor(c, 0.49)).toBe("customer");
  });
});

describe("sentences", () => {
  it("splits on end punctuation and lines, keeps URLs whole, and drops fragments", () => {
    expect(sentences("Hi!\nI help apps grow. See https://example.com/a.b for more.\n- Thanks a lot\nBest")).toEqual([
      "I help apps grow.",
      "See https://example.com/a.b for more.",
      "Thanks a lot",
    ]);
  });
});

describe("train", () => {
  it("separates two clusters", () => {
    const xs = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]];
    const w = train(xs, [1, 1, 0, 0], { epochs: 300 });
    expect(predict(w, [1, 0])).toBeGreaterThan(0.8);
    expect(predict(w, [0, 1])).toBeLessThan(0.2);
  });
});
