# Classifier

EmbeddingGemma 300M (q4) embeds each message with the prefix `task: classification | query: `; a logistic regression over the 768 numbers says how likely it is to be a pitch. Settings: {"l2":0.0001,"epochs":3000,"lr":2,"customerWeight":3}.

## Forum posts, five-fold cross-validation

Folds are split by forum topic, so no topic is ever scored by a model that saw any of it.

| Pitch at P ≥ | Customers hidden | Pitches folded away |
|---|---|---|
| 0.5 | 5 of 115 | 125 of 130 |
| 0.6 | 3 of 115 | 121 of 130 |
| 0.7 | 3 of 115 | 112 of 130 |
| 0.8 | 2 of 115 | 102 of 130 |
| 0.9 | 0 of 115 | 79 of 130 |
| 0.95 | 0 of 115 | 52 of 130 |

AUC 0.991. The settings above (and the page's bar of 0.8) were chosen on this same table, so read it as a ceiling rather than a promise.

## The 40 messages Gemma 3 1B was tried on

Scored by a model trained on the other 205 forum posts: AUC 1.000; the most pitch-like customer scores 0.28, the least pitch-like pitch 0.83. Gemma 3 1B's numbers on the same messages are in eval/EVAL.md. (Other posts from the same forum topics are in that training set.)

## My inbox, trained on the forum alone

Cold emails to an app developer are written differently from replies on a forum, and a model that has only read the forum is less sure of them.

| Pitch at P ≥ | Pitches folded away |
|---|---|
| 0.5 | 22 of 23 |
| 0.6 | 19 of 23 |
| 0.7 | 18 of 23 |
| 0.8 | 13 of 23 |
| 0.9 | 4 of 23 |
| 0.95 | 1 of 23 |

## My inbox, one sender at a time

Each sender's messages are scored by a model trained on the forum and on every other sender's messages: the shipped model has seen pitches like these, but never from this sender.

| Pitch at P ≥ | Pitches folded away |
|---|---|
| 0.5 | 23 of 23 |
| 0.6 | 23 of 23 |
| 0.7 | 23 of 23 |
| 0.8 | 23 of 23 |
| 0.9 | 19 of 23 |
| 0.95 | 16 of 23 |

The inbox holds no customers, so this half of the test can only show pitches being caught, not customers being kept.

## Forum posts on the wrong side of 0.5

- **customer**, P = 0.83: Thanks everyone. Here’s a specific link to that kit I had screenshotted. https://birthsupplies.com/product/hillary-kieser-gentle-birth-care-birth-kit/ We sell each item individually as well, but have a few hundred custom… ([post](https://community.shopify.com/t/grouped-product-app/659487/8))
- **customer**, P = 0.82: The Village Refill The Village Refill & Zero Waste Shop Award-winning refill shop in Oxfordshire. Wide range of refillable, eco-friendly, vegan & organic products. We believe small everyday choices can make a lasting imp… ([post](https://community.shopify.com/t/bundle-with-dynamic-price-updates/628728/8))
- **customer**, P = 0.73: @Bristan_FARRE We are not on Plus. Thanks for the suggestion to test post-purchase offer. It does sound like the safest option to try first. Also, good point to start with one and measure the results. We sell women’s clo… ([post](https://community.shopify.com/t/recommendations-for-best-upsell-apps/680038/8))
- **customer**, P = 0.58: Hello again Richard, I like how Inkifi website looks, its so clean. I would literally would like to have a carbon copy of that website. The editor is always impressive! On your suggestion I got in touch with Taopix - the… ([post](https://community.shopify.com/t/whats-the-best-photobook-maker-or-mockup-generator-for-a-new-print-shop/82885/8))
- **customer**, P = 0.57: Hi @EasifyApps-Zoe , yes well, I wanted to offer at least 6 size options to choose from and if they need another size, that they could do an input by themselves as well. the same goes for the Quantities, though the bigge… ([post](https://community.shopify.com/t/how-can-customers-customize-sticker-size-and-quantity-on-my-site/264056/3))
- **pitch**, P = 0.45: I’d use Flow for tagging orders for review, but my concern is that I’d still have to chase a new address or cancel the order. For a known shipping restriction, I’d rather catch it before the order is placed. With validat… ([post](https://community.shopify.com/t/how-are-you-handling-undeliverable-addresses-po-box-carrier-restrictions-before-checkout/615359/11))
- **pitch**, P = 0.33: Hi,DavidWard1 Yes, this is possible through custom code. We can add logic to check the cart total after discounts/coupons are applied, and then automatically add or remove the free gift based on the final cart value thre… ([post](https://community.shopify.com/t/gift-with-purchase-app-that-uses-cart-total-after-rebate-or-coupon/614597/2))
- **pitch**, P = 0.33: Hey, for now I don’t know what theme you are using but a lot of themes have come with this feature to show inventory status on the variant level. You can actually set a threshold in the settings below which the status wi… ([post](https://community.shopify.com/t/looking-for-a-low-stock-alert-app-with-variant-level-control-forecasting-replacing-merchbees/409978/2))
- **pitch**, P = 0.27: @Lyn-Bui Thanks much for recommending Cadny Rack. And yes, this is easily doable using the “Same product” offer. Please let me know if I can help with anything. Tom iScreen Shoter - Google Chrome - 260614010547997×740 10… ([post](https://community.shopify.com/t/looking-for-an-app-to-do-a-bogo-50-sale/631720/10))
- **pitch**, P = 0.10: SaraD24: Hey everyone! I currently have an app installed to allow my customers to make bundles. It is great, except that each product can only be tied to one bundle. I need an app that will allow me to tie each product t… ([post](https://community.shopify.com/t/what-is-a-good-app-that-allows-multiple-bundles/332074/8))
