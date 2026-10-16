// Turning what people actually have into messages: a pasted message, .eml files, or a Gmail
// Takeout .mbox. Only the plain-text part is read; the sender's text is cut out of the wrapper
// that Shopify (and similar contact forms) put around it.
//
// Files are read as raw bytes ("binary" strings, one char per byte), because a part sent as
// 8bit can be in any charset: its bytes are decoded only once its Content-Type says which.

export interface Message {
  id: string;
  from: string;
  subject: string;
  date: string;
  body: string;
}

/**
 * An mbox separator: "From <address> <asctime date>". Gmail Takeout doesn't escape "From " at the
 * start of a body line, so anything less exact would cut a message where a sentence begins "From".
 */
const SEPARATOR = /^From \S+ +\w{3} \w{3} +\d{1,2} \d\d:\d\d:\d\d(?: [+-]\d{4})? \d{4}$/;

/** Splits an mbox file into raw messages on its "From " separator lines. */
export function splitMbox(text: string): string[] {
  const out: string[] = [];
  let current: string[] | null = null;
  for (const line of text.replace(/\r\n/g, "\n").split("\n")) {
    if (SEPARATOR.test(line)) {
      if (current) out.push(current.join("\n"));
      current = [];
    } else current?.push(line);
  }
  if (current) out.push(current.join("\n"));
  return out.filter((m) => /^[\w-]+:/m.test(m.slice(0, 2000)));
}

function decodeWords(header: string, binary: boolean): string {
  // Raw 8-bit header bytes are UTF-8 far more often than not.
  const text = binary ? decodeBytes(bytesOf(header, true), "utf-8") : header;
  // Whitespace between two encoded words is not part of the text (RFC 2047).
  return text.replace(/(\?=)\s+(?==\?)/g, "$1").replace(/=\?([^?]+)\?([bqBQ])\?([^?]*)\?=/g, (_, charset: string, enc: string, data: string) => {
    const bytes = enc.toUpperCase() === "B" ? base64Bytes(data) : qpBytes(data.replace(/_/g, " "));
    return decodeBytes(bytes, charset);
  });
}

function base64Bytes(data: string): Uint8Array {
  const bin = atob(data.replace(/[^A-Za-z0-9+/=]/g, ""));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

/** The bytes a string stands for: one per char if it was read as binary, else its UTF-8. */
function bytesOf(text: string, binary: boolean): Uint8Array {
  return binary ? Uint8Array.from(text, (c) => c.charCodeAt(0) & 0xff) : new TextEncoder().encode(text);
}

function qpBytes(data: string, binary = false): Uint8Array {
  const text = data.replace(/=\r?\n/g, "");
  const out: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const hex = text.slice(i + 1, i + 3);
    if (text[i] === "=" && /^[0-9A-Fa-f]{2}$/.test(hex)) {
      out.push(parseInt(hex, 16));
      i += 2;
    } else {
      for (const b of bytesOf(text[i], binary)) out.push(b);
    }
  }
  return Uint8Array.from(out);
}

function decodeBytes(bytes: Uint8Array, charset = "utf-8"): string {
  try {
    return new TextDecoder(charset.toLowerCase()).decode(bytes);
  } catch {
    return new TextDecoder("utf-8").decode(bytes);
  }
}

interface Part { headers: Map<string, string>; body: string; binary: boolean }

function splitPart(raw: string, binary: boolean): Part {
  const text = raw.replace(/\r\n/g, "\n");
  const end = text.indexOf("\n\n");
  const head = end < 0 ? text : text.slice(0, end);
  const headers = new Map<string, string>();
  for (const line of head.replace(/\n[ \t]+/g, " ").split("\n")) {
    const m = /^([\w-]+):\s*(.*)$/.exec(line);
    if (m) headers.set(m[1].toLowerCase(), m[2]);
  }
  return { headers, body: end < 0 ? "" : text.slice(end + 2), binary };
}

function param(header: string, name: string): string | undefined {
  return new RegExp(`${name}="?([^";]+)"?`, "i").exec(header)?.[1];
}

function decodePart(part: Part): string {
  const type = part.headers.get("content-type") ?? "text/plain";
  const enc = (part.headers.get("content-transfer-encoding") ?? "").toLowerCase();
  const charset = param(type, "charset") ?? "utf-8";
  if (enc === "base64") return decodeBytes(base64Bytes(part.body), charset);
  if (enc === "quoted-printable") return decodeBytes(qpBytes(part.body, part.binary), charset);
  return part.binary ? decodeBytes(bytesOf(part.body, true), charset) : part.body;
}

const isAttachment = (p: Part) => /^\s*attachment/i.test(p.headers.get("content-disposition") ?? "");

/**
 * The text of a message: the first text/plain part that isn't an attachment, looking inside
 * nested multiparts in order; failing that, the first text/html part with its tags taken out.
 */
function textOf(part: Part): string | null {
  // Some mailers send an empty text/plain part next to the real HTML one.
  const plain = findPart(part, "text/plain");
  return plain?.trim() ? plain : (findPart(part, "text/html") ?? plain);
}

function findPart(part: Part, want: "text/plain" | "text/html"): string | null {
  const type = (part.headers.get("content-type") ?? "text/plain").toLowerCase();
  if (type.startsWith("multipart/")) {
    const boundary = param(part.headers.get("content-type")!, "boundary");
    if (!boundary) return null;
    const children = part.body.split(`--${boundary}`).slice(1).filter((p) => !p.startsWith("--")).map((p) => splitPart(p.replace(/^\n/, ""), part.binary));
    for (const c of children) {
      const t = findPart(c, want);
      if (t !== null) return t;
    }
    return null;
  }
  if (isAttachment(part) || !type.startsWith(want)) return null;
  const text = decodePart(part);
  return want === "text/html" ? htmlToText(text) : text;
}

export function htmlToText(html: string): string {
  return html
    .replace(/<(style|script)[\s\S]*?<\/\1>/gi, "")
    .replace(/<br\s*\/?>|<\/(p|div|tr|li|h\d)>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&#(\d+);/g, (_, n: string) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n: string) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n");
}

/**
 * Contact forms wrap the sender's words in their own: Shopify's app listing sends "New message
 * about <app>. <name> sent you a message. See it below." and a footer. Keep only the message.
 */
export function unwrap(body: string): string {
  let text = body.replace(/\r\n/g, "\n");
  const start = /sent you a message\.?\s*See it below\.?/i.exec(text);
  if (start) {
    text = text.slice(start.index + start[0].length);
    // The form's own footer, only when the form's own header was there: a customer may well
    // write "Reply to this when you can".
    const end = /\n\s*(Reply to this message in|You can reply to this|This message was sent (to you )?(through|via|from)|You're receiving this (email|message) because|View (it )?in (your )?Partner Dashboard)/i.exec(text);
    if (end) text = text.slice(0, end.index);
  }
  // Quoted earlier messages in a reply are not what this sender is asking now.
  const quoted = /\n\s*On .{5,80} wrote:\s*\n/i.exec(text);
  if (quoted) text = text.slice(0, quoted.index);
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

export function parseEmail(raw: string, id: string, binary = false): Message | null {
  const part = splitPart(raw, binary);
  const text = textOf(part);
  if (text === null) return null;
  const body = unwrap(text);
  if (!body) return null;
  return {
    id,
    from: decodeWords(part.headers.get("from") ?? "", binary).replace(/\s*<[^>]*>/, "").replace(/"/g, "").trim(),
    subject: decodeWords(part.headers.get("subject") ?? "", binary),
    date: part.headers.get("date") ?? "",
    body,
  };
}

/** Pasted text: an email with headers if it looks like one, otherwise the message itself. */
export function parsePasted(text: string, id: string): Message | null {
  // Headers only if the paste starts with one; "Date: Oct 5" further down is part of a message.
  if (/^(from|subject|date|received|return-path|delivered-to|mime-version|to|message-id):/i.test(text.trimStart()) && /\n\s*\n/.test(text)) {
    const m = parseEmail(text, id);
    if (m) return m;
  }
  const body = unwrap(text);
  return body ? { id, from: "", subject: "", date: "", body } : null;
}

/** A dropped file, read as binary (one char per byte) so 8-bit parts decode in their own charset. */
export function parseFile(name: string, binary: string): Message[] {
  if (/\.mbox$/i.test(name) || SEPARATOR.test(binary.slice(0, binary.indexOf("\n")).replace(/\r$/, ""))) {
    return splitMbox(binary).flatMap((raw, i) => parseEmail(raw, `${name}#${i}`, true) ?? []);
  }
  if (/\.eml$/i.test(name) || /^[\w-]+: /.test(binary)) return [parseEmail(binary, name, true)].filter((m): m is Message => m !== null);
  // Anything else is a text file of one message.
  return [parsePasted(decodeBytes(bytesOf(binary, true), "utf-8"), name)].filter((m): m is Message => m !== null);
}

/** Reads a File as a binary string, one char per byte. */
export async function readBinary(file: Blob): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let out = "";
  for (let i = 0; i < bytes.length; i += 0x8000) out += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return out;
}
