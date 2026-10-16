// Real merchants asking for help, and real vendors selling, from the public Shopify Community
// "Shopify Apps" board (a Discourse forum with a JSON API). Topics' first posts are merchants
// asking for something; many replies are app makers recommending their own app. Every post is
// labelled by hand afterwards in eval/messages.jsonl; this script only gathers the raw text.
//
//   node eval/forum.ts [pages]   → eval/cache/forum.json

import { mkdirSync, writeFileSync } from "node:fs";
import { htmlToText } from "../src/parse.ts";

const BASE = "https://community.shopify.com";
const CATEGORY = "c/shopify-apps/186";
const UA = { "User-Agent": "Doorman-eval/0.1 (https://github.com/mikey92/doorman)" };

interface Post { id: number; post_number: number; username: string; cooked: string; created_at: string; reply_to_post_number: number | null }

async function json<T>(path: string): Promise<T> {
  for (let tries = 0; ; tries++) {
    const res = await fetch(`${BASE}/${path}`, { headers: UA });
    if (res.ok) return (await res.json()) as T;
    if (res.status !== 429 || tries > 3) throw new Error(`${res.status} ${path}`);
    await new Promise((r) => setTimeout(r, 5000 * (tries + 1)));
  }
}

const pages = Number(process.argv[2] ?? 3);
const out: { topic: number; title: string; url: string; posts: { n: number; user: string; date: string; replyTo: number | null; text: string }[] }[] = [];
for (let page = 0; page < pages; page++) {
  const list = await json<{ topic_list: { topics: { id: number; slug: string; title: string; posts_count: number }[] } }>(`${CATEGORY}.json?page=${page}`);
  for (const t of list.topic_list.topics) {
    if (t.title.startsWith("About the")) continue;
    const topic = await json<{ post_stream: { posts: Post[] } }>(`t/${t.id}.json`);
    out.push({
      topic: t.id,
      title: t.title,
      url: `${BASE}/t/${t.slug}/${t.id}`,
      posts: topic.post_stream.posts.map((p) => ({
        n: p.post_number,
        user: p.username,
        date: p.created_at.slice(0, 10),
        replyTo: p.reply_to_post_number,
        text: htmlToText(p.cooked).trim(),
      })),
    });
    console.log(`${t.id} ${topic.post_stream.posts.length} posts: ${t.title}`);
    await new Promise((r) => setTimeout(r, 400));
  }
}
mkdirSync(new URL("./cache/", import.meta.url), { recursive: true });
writeFileSync(new URL("./cache/forum.json", import.meta.url), JSON.stringify(out, null, 1));
