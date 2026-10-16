import { describe, expect, it } from "vitest";
import { parseEmail, parseFile, parsePasted, unwrap } from "../src/parse.ts";

const shopify = [
  "From: Shopify <mailer@shopify.com>",
  "Subject: =?UTF-8?Q?Acme_has_a_question_about_Upright_Cleaner?=",
  "Content-Type: multipart/alternative; boundary=\"b1\"",
  "",
  "--b1",
  "Content-Type: text/plain; charset=UTF-8",
  "Content-Transfer-Encoding: quoted-printable",
  "",
  "New message about Upright Cleaner. Acme sent you a message. See it below.",
  "",
  "Hi, the cleaner deleted a product I wanted to keep. Can I get it back=3F",
  "",
  "Reply to this message in your Partner Dashboard.",
  "--b1",
  "Content-Type: text/html; charset=UTF-8",
  "",
  "<p>ignored</p>",
  "--b1--",
  "",
].join("\n");

describe("parseEmail", () => {
  it("reads the plain part, decodes it, and cuts the contact form's wrapper away", () => {
    const m = parseEmail(shopify, "a")!;
    expect(m.from).toBe("Shopify");
    expect(m.subject).toBe("Acme has a question about Upright Cleaner");
    expect(m.body).toBe("Hi, the cleaner deleted a product I wanted to keep. Can I get it back?");
  });
});

describe("parseFile", () => {
  it("splits an mbox into its messages", () => {
    const mbox = `From 123@xxx Mon Oct 12 10:00:00 2026\n${shopify}\nFrom 456@xxx Mon Oct 12 11:00:00 2026\nFrom: Bob <b@example.com>\nSubject: hi\n\nIs there a free trial?\n`;
    const ms = parseFile("all.mbox", mbox);
    expect(ms.map((m) => m.body)).toEqual(["Hi, the cleaner deleted a product I wanted to keep. Can I get it back?", "Is there a free trial?"]);
  });
});

describe("parsePasted", () => {
  it("takes plain pasted text as the message", () => {
    expect(parsePasted("  Do you ship to Canada?  ", "p")!.body).toBe("Do you ship to Canada?");
  });
});

describe("unwrap", () => {
  it("drops the earlier message quoted under a reply", () => {
    expect(unwrap("Thanks, that worked!\n\nOn Mon, Oct 12, 2026 at 9:00 AM Me wrote:\n> try this")).toBe("Thanks, that worked!");
  });
});

describe("parseEmail with attachments and HTML", () => {
  it("reads the body, not a text attachment that comes first at the top level", () => {
    const raw = [
      "From: A <a@example.com>",
      'Content-Type: multipart/mixed; boundary="m"',
      "",
      "--m",
      'Content-Type: multipart/alternative; boundary="a"',
      "",
      "--a",
      "Content-Type: text/plain",
      "",
      "Where is my order?",
      "--a--",
      "--m",
      "Content-Type: text/plain",
      'Content-Disposition: attachment; filename="log.txt"',
      "",
      "attached log",
      "--m--",
      "",
    ].join("\n");
    expect(parseEmail(raw, "x")!.body).toBe("Where is my order?");
  });
  it("skips a text attachment listed before the body", () => {
    const raw = [
      'Content-Type: multipart/mixed; boundary="m"',
      "",
      "--m",
      "Content-Type: text/plain",
      'Content-Disposition: attachment; filename="notes.txt"',
      "",
      "attached notes",
      "--m",
      "Content-Type: text/html",
      "",
      "<p>I&#8217;m stuck at checkout &#x2014; help?</p>",
      "--m--",
      "",
    ].join("\n");
    expect(parseEmail(raw, "y")!.body).toBe("I’m stuck at checkout — help?");
  });
});

describe("real-inbox quirks", () => {
  const latin1 = (s: string) => Array.from(new TextEncoder().encode(s), (b) => String.fromCharCode(b)).join("");

  it("doesn't cut an mbox message at a body line that starts with 'From '", () => {
    const mbox = "From 1@xxx Wed Oct 14 10:00:00 +0000 2026\nFrom: A <a@x.com>\nSubject: s\n\nHello.\nFrom my experience the app breaks.\nNote: please fix.\n";
    expect(parseFile("t.mbox", mbox).map((m) => m.body)).toEqual(["Hello.\nFrom my experience the app breaks.\nNote: please fix."]);
  });
  it("reads the HTML part when the plain part is empty", () => {
    const raw = 'Content-Type: multipart/alternative; boundary="a"\n\n--a\nContent-Type: text/plain\n\n\n--a\nContent-Type: text/html\n\n<p>Refund please</p>\n--a--\n';
    expect(parseEmail(raw, "x")!.body).toBe("Refund please");
  });
  it("keeps a customer's own 'Reply to…' line", () => {
    expect(unwrap("Hi, I have a problem.\nReply to this email when you can.\nThanks")).toBe("Hi, I have a problem.\nReply to this email when you can.\nThanks");
  });
  it("decodes an 8-bit part in its declared charset", () => {
    const bytes = new Uint8Array([0xbe, 0xc8, 0xb3, 0xe7]); // "안녕" in EUC-KR
    const raw = "Subject: =?UTF-8?B?7JWI?= =?UTF-8?B?64WV?=\nContent-Type: text/plain; charset=euc-kr\nContent-Transfer-Encoding: 8bit\n\n" + String.fromCharCode(...bytes);
    const [m] = parseFile("k.eml", raw);
    expect(m.body).toBe("안녕");
    expect(m.subject).toBe("안녕");
  });
  it("reads raw UTF-8 in a binary-read file", () => {
    expect(parseFile("u.eml", latin1("Subject: hi\n\nCafé ☕ order?"))[0].body).toBe("Café ☕ order?");
  });
  it("treats pasted text as a message unless it starts with a header", () => {
    expect(parsePasted("Hi there\nDate: Oct 5 is when I ordered.\n\nStill nothing.", "p")!.body).toBe("Hi there\nDate: Oct 5 is when I ordered.\n\nStill nothing.");
  });
});
