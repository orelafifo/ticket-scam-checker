# Real-world evaluation

Test set: 80 listings (40 scams from published sources: 11 verbatim, 29 reconstructed from documented UK cases; 40 constructed genuine listings). None were used in training.

| Approach | Scams caught | Recall | Precision | F1 | False alarms |
|---|---|---|---|---|---|
| Rules only | 30/40 | 75% | 83% | 79% | 6/40 |
| Text model only | 38/40 | 95% | 54% | 69% | 32/40 |
| AI only (Gemini) | 39/40 | 98% | 89% | 93% | 5/40 |
| Hybrid (our checker) | 19/40 | 48% | 90% | 62% | 2/40 |

For comparison, the hybrid on its own synthetic test set: recall 100%, precision 93%, F1 96%.

Recall by source type:

| Approach | Verbatim quotes | Reconstructed cases |
|---|---|---|
| Rules only | 64% | 79% |
| Text model only | 100% | 93% |
| AI only (Gemini) | 91% | 100% |
| Hybrid (our checker) | 18% | 59% |

## Scams the hybrid missed (21)

- **S01** (The Music (AU), 2020; score 0.06): "Selling 4x tickets pm me." - flags: social_or_messaging, no_seat_details
- **S02** (The Music (AU), 2020; score 0.34): "Selling 4x tickets pm me. My PayPal wasn't set up for Goods & Services payments, I could only accept Friends & Family. Why do you have issue" - flags: social_or_messaging, unsafe_payment, no_seat_details
- **S03** (The Music (AU), 2020; score 0.07): "I have tickets for sale at a reduced price" - flags: social_or_messaging, no_seat_details
- **S04** (Facebook scam blog, 2019; score 0.08): "Me and my family can't make it to this wonderful event. Kindly DM me for cheap tickets at a discount rate if you are interested" - flags: social_or_messaging, off_platform, no_seat_details
- **S07** (Music in Minnesota; score 0.11): "Family went out of town so I am unable to attend! PM me!" - flags: social_or_messaging, new_account, few_followers, no_seat_details
- **S08** (Music in Minnesota; score 0.18): "My friends bailed on me so I'm selling our tickets CHEAP! DM me!" - flags: social_or_messaging, off_platform, new_account, few_followers, no_seat_details
- **S09** (Music in Minnesota; score 0.13): "How much are you willing to pay? I prefer you pay me with an Amazon card. You can get it at any store near you or online at Amazon.com!" - flags: social_or_messaging, new_account, few_followers, no_seat_details
- **S10** (NOW Toronto; score 0.07): "I bought four tickets before I realized we had other plans that day." - flags: social_or_messaging, no_seat_details
- **S11** (NOW Toronto; score 0.08): "I'm currently in the United States for surgery that's why I cannot attend event." - flags: social_or_messaging, no_seat_details
- **S15** (Lloyds 2024 (sold-out shows at inflated prices); score 0.07): "Sold out show!! Have 2 VIP standing for Taylor Swift at Wembley, £450 each. Pay upfront to secure, tickets emailed after." - flags: social_or_messaging, no_seat_details
- **S17** (Lloyds 2026 (football: FA Cup final); score 0.05): "Season ticket holder, can't make the FA Cup final so selling my seat for £120. Need payment today as others are asking." - flags: social_or_messaging, no_seat_details
- **S22** (GOV.UK/Action Fraud 2025 (last-minute fake ticket companies); score 0.16): "Last minute Glastonbury tickets with coach travel, £395 inc. Limited release, book via our website before they're gone" - flags: unknown_website, sob_story, no_seat_details
- **S23** (Santander 2023 (social media pressure); score 0.26): "There's a lot of interest in these so I need the full amount upfront today to hold them for you. 2 x Central Cee, O2, £80 each." - flags: social_or_messaging, deposit_or_hold, no_seat_details
- **S24** (Santander 2023 (fake websites, not yet on sale); score 0.16): "Pre-sale tickets for Beyonce's 2027 UK tour now available, guaranteed entry. Book now before general sale." - flags: unknown_website, no_seat_details
- **S25** (Santander 2023 (no-show handover); score 0.22): "Can meet you outside the venue to hand them over, just need a deposit first so I know you're serious" - flags: social_or_messaging, deposit_or_hold, no_seat_details
- **S28** (Oasis warning 2026 (fake code messages); score 0.15): "Oasis Live '27: you've been selected for priority access. Pay the £25 registration fee here to receive your code" - flags: unknown_website, no_seat_details
- **S31** (Eventbrite (unsolicited Instagram DM); score 0.2): "hey saw u wanted tickets for the sold out show, i have 1 floor ticket, £150, dm me and i'll send it after u pay" - flags: social_or_messaging, off_platform, new_account, few_followers, no_seat_details
- **S32** (How-To Geek (excess tickets, copied posts); score 0.17): "Accidentally bought too many tickets for Sabrina Carpenter at the O2, selling 2 at a discount. Can send screenshots of the order." - flags: social_or_messaging, new_account, few_followers, no_seat_details
- **S33** (How-To Geek (fell ill); score 0.07): "I've come down with flu so can't make it to Fred again.. tonight, 1 ticket going cheap, quick sale. PM me." - flags: social_or_messaging, urgency, no_seat_details
- **S34** (NOW Toronto (paying into a friend's account); score 0.05): "Hi, I'm selling 2 tickets for Arctic Monkeys. I'm abroad for surgery so can't attend. You can pay into my friend's account as mine is frozen" - flags: social_or_messaging, no_seat_details
- **S35** (The Music AU (like and comment first); score 0.09): "Comment 'interested' and like this post and I'll message you! 4 tickets for Dua Lipa, reduced price." - flags: social_or_messaging, no_seat_details

## Genuine listings the hybrid wrongly flagged (2)

- **G08** (WhatsApp; score 0.47): "Hey guys, [FRIEND] can't come to Wizkid anymore so we've got a spare. £65, just bank transfer me and I'll transfer it on the app straight aw" - flags: social_or_messaging, unsafe_payment, no_seat_details
- **G32** (Facebook (post, group or Marketplace); score 0.44): "2 tickets for Sam Fender, St James Park. Paid £85 each, will take £80. Happy to do bank transfer if easier, or paypal G&S, and I'll send via" - flags: social_or_messaging, unsafe_payment, no_seat_details

## What this shows (interpretation, written after the v1 run)
- The synthetic results do not carry over to real examples: the hybrid drops sharply on real scams. This is the expected effect of training only on template data (a 'distribution shift').
- Real scam posts are often short and vague ("Selling 4x tickets pm me"), with no price or payment detail. The synthetic scams almost always carried several obvious flags, so the model learned that one or two weak signals mean 'genuine'.
- Some real tactics are not covered by any rule yet: 'pm me', paying 'upfront', paying into a friend's account, gift-card wording, registration fees for codes, 'like and comment first', vague 'reduced price' posts.
- The text-only model catches almost everything but raises a false alarm on most genuine posts, so it is not usable alone; rules alone are the most balanced today.
- Next iteration: widen the rules, make the synthetic training data more realistic (short vague scams, messy genuine sellers), then re-test on a NEW held-out set collected independently. This 80-row set has now been used for error analysis, so it becomes the development set.

## Method notes
- Thresholds and the rules-only decision rule were fixed before running the test.
- Event check and website-age lookup were switched off for all approaches (see docstring).
- Unknown seller details were treated as neutral (365-day account, 200 followers).
- Limitations: small test set; genuine listings are constructed, not collected; reconstructed scams were written by the developer, which risks unconscious bias towards the rules. Next step: an independent, real-world labelled set via a bank or platform partner.