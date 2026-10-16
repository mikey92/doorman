# Where the messages come from

Two sets, both real, both read and labelled by hand.

## `messages.jsonl`: the public Shopify Apps forum (245 posts)

`eval/forum.ts` gathered the 60 most recently active topics on the [Shopify Community "Shopify Apps" board](https://community.shopify.com/c/shopify-apps/186) on 16 October 2026: 637 posts. Every post was read in full and given one of four classes:

| Class | Posts | What it means |
|---|---|---|
| asker | 117 | Needs something: a merchant or developer asking for help, an app, or advice |
| pitch | 282 | Sells their own app, service, agency or freelance help, openly or after some genuinely helpful advice |
| helper | 194 | Helps with nothing of their own to sell |
| other | 44 | Thanks, "+1", hidden by moderators, empty |

150 calls were marked unsure (mostly "is this account really the app's maker?") and left out. So were helpers and others, which a contact form doesn't get. Of the confident pitches, only the first from each account is kept: one vendor posting the same template eleven times would otherwise grade the classifier on that template. Posts of five words or fewer are dropped.

That leaves **115 askers (labelled `customer`) and 130 pitches**, each with a link to the post. Askers on a forum are close to customers writing to an app: they describe a problem with a store and ask for help. The pitches are the same people who write to app developers' contact forms, doing it in public.

One thing worth knowing: the forum's topics are often about choosing an app, so an asker's message is about apps and the pitches answering it are about apps. Telling them apart is not a matter of topic.

## `inbox.jsonl`: everything that reached my own apps (23 pitches)

Every pitch that reached my three Shopify apps' listing contact form (Upright Cleaner, Upright Net Terms, Upright Upsell) or my email about them between 29 August and 13 October 2026: 35 messages. Three were a single image with no text; of the other 32, 23 are left once identical copies are removed. Names, links and phone numbers are replaced with `[name]`, `[link]` and `[phone]`.

Not one message in that time was from a customer. That is why Doorman exists, and it is also why this set has no customers in it.

`eval/CLASSIFIER.md` scores this set twice: with a model trained on the forum alone, and one sender at a time with a model trained on the forum and every other sender. The shipped classifier is trained on both sets.
