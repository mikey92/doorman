// Turning what people actually have into messages: a pasted message, .eml files, or a Gmail
// Takeout .mbox. Only the plain-text part is read; the sender's text is cut out of the wrapper
// that Shopify (and similar contact forms) put around it.

export interface Message {
  id: string;
  from: string;
  subject: string;
  date: string;
  body: string;
}

/** Splits an mbox file into raw messages on its "From " separator lines. */
export function splitMbox(text: string): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split(/\n(?=From [^\n]*\n)/)
    .map((m) => m.replace(/^From [^\n]*\n/, ""))
    .filter((m) => /^[\w-]+:/m.test(m.slice(0, 2000)));
}

function decodeWords(header: string): string {
  return header.replace(/=\?([^?]+)\?([bqBQ])\?([^?]*)\?=/g, (_, charset: string, enc: string, data: string) => {
    const bytes = enc.toUpperCase() === "B" ? base64Bytes(data) : qpBytes(data.replace(/_/g, " "));
    return decodeBytes(bytes, charset);
  });
}

function base64Bytes(data: string): Uint8Array {
  const bin = atob(data.replace(/[^A-Za-z0-9+/=]/g, ""));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

function qpBytes(data: string): Uint8Array {
  const text = data.replace(/=\r?\n/g, "");
  const out: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const hex = text.slice(i + 1, i + 3);
    if (text[i] === "=" && /^[0-9A-Fa-f]{2}$/.test(hex)) {
      out.push(parseInt(hex, 16));
      i += 2;
    } else {
      // Characters outside Latin-1 in a mislabelled part: keep them as UTF-8.
      for (const b of new TextEncoder().encode(text[i])) out.push(b);
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

interface Part { headers: Map<string, string>; body: string }

function splitPart(raw: string): Part {
  const text = raw.replace(/\r\n/g, "\n");
  const end = text.indexOf("\n\n");
  const head = end < 0 ? text : text.slice(0, end);
  const headers = new Map<string, string>();
  for (const line of head.replace(/\n[ \t]+/g, " ").split("\n")) {
    const m = /^([\w-]+):\s*(.*)$/.exec(line);
    if (m) headers.set(m[1].toLowerCase(), m[2]);
  }
  return { headers, body: end < 0 ? "" : text.slice(end + 2) };
}

function param(header: string, name: string): string | undefined {
  return new RegExp(`${name}="?([^";]+)"?`, "i").exec(header)?.[1];
}

function decodePart(part: Part): string {
  const type = part.headers.get("content-type") ?? "text/plain";
  const enc = (part.headers.get("content-transfer-encoding") ?? "").toLowerCase();
  const charset = param(type, "charset") ?? "utf-8";
  if (enc === "base64") return decodeBytes(base64Bytes(part.body), charset);
  if (enc === "quoted-printable") return decodeBytes(qpBytes(part.body), charset);
  return part.body;
}

const isAttachment = (p: Part) => /^\s*attachment/i.test(p.headers.get("content-disposition") ?? "");

/**
 * The text of a message: the first text/plain part that isn't an attachment, looking inside
 * nested multiparts in order; failing that, the first text/html part with its tags taken out.
 */
function textOf(part: Part): string | null {
  return findPart(part, "text/plain") ?? findPart(part, "text/html");
}

function findPart(part: Part, want: "text/plain" | "text/html"): string | null {
  const type = (part.headers.get("content-type") ?? "text/plain").toLowerCase();
  if (type.startsWith("multipart/")) {
    const boundary = param(part.headers.get("content-type")!, "boundary");
    if (!boundary) return null;
    const children = part.body.split(`--${boundary}`).slice(1).filter((p) => !p.startsWith("--")).map((p) => splitPart(p.replace(/^\n/, "")));
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
  if (start) text = text.slice(start.index + start[0].length);
  const end = /\n\s*(Reply to|You can reply|This message was sent|You're receiving this|View in Partner Dashboard)/i.exec(text);
  if (end) text = text.slice(0, end.index);
  // Quoted earlier messages in a reply are not what this sender is asking now.
  const quoted = /\n\s*On .{5,80} wrote:\s*\n/i.exec(text);
  if (quoted) text = text.slice(0, quoted.index);
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

export function parseEmail(raw: string, id: string): Message | null {
  const part = splitPart(raw);
  const text = textOf(part);
  if (text === null) return null;
  const body = unwrap(text);
  if (!body) return null;
  return {
    id,
    from: decodeWords(part.headers.get("from") ?? "").replace(/\s*<[^>]*>/, "").replace(/"/g, "").trim(),
    subject: decodeWords(part.headers.get("subject") ?? ""),
    date: part.headers.get("date") ?? "",
    body,
  };
}

/** Pasted text: an email with headers if it looks like one, otherwise the message itself. */
export function parsePasted(text: string, id: string): Message | null {
  if (/^(from|subject|date|received|return-path|delivered-to|mime-version):/im.test(text.slice(0, 500)) && /\n\s*\n/.test(text)) {
    const m = parseEmail(text, id);
    if (m) return m;
  }
  const body = unwrap(text);
  return body ? { id, from: "", subject: "", date: "", body } : null;
}

export function parseFile(name: string, text: string): Message[] {
  if (/\.mbox$/i.test(name) || /^From [^\n]*\n[\w-]+:/.test(text)) {
    return splitMbox(text).flatMap((raw, i) => parseEmail(raw, `${name}#${i}`) ?? []);
  }
  if (/\.eml$/i.test(name)) return [parseEmail(text, name)].filter((m): m is Message => m !== null);
  return [parsePasted(text, name)].filter((m): m is Message => m !== null);
}
