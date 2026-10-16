# Doorman

**Your app's or shop's contact form, sorted: customers first, sales pitches folded away. It runs in your browser; nothing you load leaves the page.**

Live: **https://doorman.mikey9220.workers.dev** · Evaluation: [eval/CLASSIFIER.md](eval/CLASSIFIER.md) · Data: [eval/DATA.md](eval/DATA.md)

In six weeks my three Shopify apps' contact form, and the email behind it, got 23 different messages from 21 senders. All 23 were people selling me something: installs, reviews, a promo post in their Discord. Not one was a customer. The day one is, I don't want it buried under the next "I came across your app and was really impressed".

![Sorting the example inbox: customers stay on top, pitches fold away with the giveaway sentence highlighted](media/demo.gif)

Load messages (paste one, drop `.eml` files, or a Gmail Takeout `.mbox`) and press **Sort Inbox**. **EmbeddingGemma 300M**, running in a Web Worker on your GPU, turns each message and each of its sentences into 768 numbers; a logistic regression trained on 245 hand-labelled forum posts and my inbox's pitches says how likely each is to be a pitch. Customers stay on top. Close calls stay on top too, marked "not sure". Pitches fold away, each with the sentence that gave it away highlighted. Nothing is deleted or answered.

## How well it works

| | Customers hidden | Pitches folded away |
|---|---|---|
| Public forum posts, 5-fold cross-validation by topic (P ≥ 0.8) | 2 of 115 | 102 of 130 |
| My inbox, each sender scored by a model that never saw them (P ≥ 0.8) | — | 23 of 23 |

Between 0.5 and 0.8 a message stays on top as "not sure". Full tables, and every forum post on the wrong side: [eval/CLASSIFIER.md](eval/CLASSIFIER.md).

## Run it

```sh
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests, no network, no model
npm run build      # static site in dist/
```

No server and no API keys. EmbeddingGemma's weights (197 MB, q4) come from the Hugging Face Hub the first time and then stay in the browser's cache. Without WebGPU it runs on the CPU through WebAssembly, more slowly.

## How it works

| Step | Code |
|---|---|
| Read `.eml`, `.mbox` or pasted text; keep the text part; cut the contact form's wrapper ("… sent you a message. See it below.") and quoted replies | `src/parse.ts` |
| Split into sentences | `src/triage.ts` |
| Embed the message and each sentence with EmbeddingGemma 300M (`task: classification \| query: ` prefix), via Transformers.js on WebGPU | `src/embed.ts`, `src/worker.ts` |
| P(pitch) from 768 weights and a bias (`public/classifier.json`); the message's P decides; its most pitch-like sentence is quoted | `src/linear.ts`, `src/triage.ts` |
| Train and evaluate: cross-validation by forum topic, then a test on my own inbox, never trained on | `eval/classify.ts` |

`eval/run.ts` is the first attempt, kept because the write-up talks about it: Gemma 3 1B asked "is this sentence an offer? Yes or No", read from its next-token logits.

## Data and licences

- Code: MIT.
- Training and evaluation posts: public posts on the Shopify Community forum, quoted for evaluation with links to each post. My inbox's pitches are anonymised.
- Model: [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) under the Gemma Terms of Use; ONNX export by onnx-community.
