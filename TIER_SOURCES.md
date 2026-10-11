# v2.5 expert evidence tiers: which source supports each red flag

**What this file is for.** In v2.5 every red flag starts from an expert starting weight, set by its tier. Training then adjusts each weight (see `hybrid_model.py` and `select_v25.py`). This table shows why each flag is in its tier, so the starting weights come from published evidence and not from guesswork.

**Before you cite any of this:**
- Open each link and check that it says what the table claims.
- Add each source to Zotero.
- Rows marked *(Claude's suggestion)* were placed by Claude, not chosen by you. Agree with them or change them.

| Tier | Starting weight | What the tier means |
|---|---|---|
| Hard stop | 3.0 (also forces High) | Almost never seen in a genuine sale; official sources say "this is a scam" |
| Strong | 2.5 | Named by several official sources as a main scam sign |
| Moderate | 1.5 | Common in scams, but genuine fans also do it sometimes |
| Weak | 0.5 | Only a small sign on its own; it matters when combined with other flags |
| Good sign | -2.0 | Official / face-value resale with buyer protection |

## Hard stop
| Flag | Plain English | Main sources |
|---|---|---|
| asks_for_codes | Asks for a verification / login code | Eventbrite (account takeover); Oasis 2026 warning (fake code messages) |
| lookalike_link | Link copies a real ticket site's name | GOV.UK / Action Fraud 2025 (fake Twickets websites); BBB 2022 (Ticketmaster lookalike sites) |
| third_party_payment | Pay a friend / someone else's account | NOW Toronto (paying into a friend's account) |
| hide_reference | Tells you not to mention tickets in the payment reference | Lloyds 2024; Santander 2023 (to dodge bank checks) |
| event_mismatch | Date or venue doesn't match the official event | Santander 2023 (fake events); real event data from Ticketmaster / Skiddle APIs |
| code_or_access_sale | Sells presale codes / access, not tickets | Oasis 2026 warning (codes for sale) |
| format_mismatch | Paper / PDF / emailed tickets for an app-only event | NOW Toronto (emailed tickets); app-only rules on the official event page |

## Strong
| Flag | Plain English | Main sources |
|---|---|---|
| unsafe_payment | Bank transfer / friends-and-family only | Lloyds 2024, 2026; Santander 2023; GOV.UK / Action Fraud 2025; UK Finance 2026 (most purchase scams paid by bank transfer) |
| payment_switch | Moves you off a protected payment method | GOV.UK / Action Fraud 2025 (pushed off payment platforms) |
| deposit_or_hold | Wants a deposit to "hold" tickets | How-To Geek; Music in Minnesota |
| sudden_seller | Account suddenly selling tickets (possible hacked account) | Which? 2025 (hacked Instagram, Oasis); WCPO 2024; CBS 2023 |
| unknown_website | Unknown ticket website | GOV.UK / Action Fraud 2025 (last-minute fake ticket companies) |
| new_domain | Website registered only recently | Santander 2023 (fake websites); RDAP domain-age check |
| very_cheap | Price under 50% of face value ("too good to be true") | Action Fraud; Which? 2025 ("if it seems too good to be true") |
| event_not_found *(Claude's suggestion)* | Event can't be found on official listings | Santander 2023 (tickets for events not on sale / not real) |
| before_on_sale *(Claude's suggestion)* | Selling before tickets officially go on sale | Santander 2023 (fake websites, not yet on sale) |

## Moderate
| Flag | Plain English | Main sources |
|---|---|---|
| off_platform | Moves the chat to WhatsApp / DMs | Lloyds 2026 (Facebook to WhatsApp) |
| send_after_payment | Tickets only sent after you pay | Santander 2023 (no-show handover) |
| screenshot_proof | "Proof" is a photo or screenshot of tickets | Lloyds 2024 (pictures of real tickets) |
| vague_listing | Very short, vague post | How-To Geek (copied posts) |
| cheap_wording | "Selling at cost / below face value" wording | WCPO 2024 (hacked friend, face value) |
| engagement_bait | "Like and comment first" | The Music AU 2020 |
| social_or_messaging | Sold through social media or a messaging app | Lloyds 2024 / 2026 (most ticket scams start on Facebook / Instagram) |
| sob_story | "Can't go any more / fell ill" story *(you chose moderate)* | How-To Geek (fell ill) |
| insecure_or_short_link *(Claude's suggestion)* | Shortened or non-https link | BBB 2022 (lookalike sites); general phishing guidance |

## Weak
| Flag | Plain English | Main sources |
|---|---|---|
| new_account | Seller account is new | Eventbrite; How-To Geek |
| few_followers | Very few followers | Eventbrite |
| no_seat_details | No block / row / seat given | Music in Minnesota |
| urgency | "Quick, first to pay gets them" | Santander 2023 (social media pressure) |
| trust_claims | "100% legit / genuine" *(you chose weak)* | Music in Minnesota; How-To Geek |
| too_cheap | Price under 80% of face value | Lloyds 2024 (genuine resellers rarely sell far below face value) |
| high_demand *(Claude's suggestion)* | Sold-out / high-demand show | Lloyds 2024 (Taylor Swift); Lloyds 2025 (Oasis) |
| secondary_market *(Claude's suggestion)* | Secondary resale site (not face-value) | GOV.UK / Action Fraud 2025 |

## Good sign
| Flag | Plain English | Main sources |
|---|---|---|
| official_platform | Official seller or face-value resale with buyer protection (Twickets, Ticketmaster Fan-to-Fan, etc.) | GOV.UK / Action Fraud 2025; STAR (Society of Ticket Agents and Retailers) advice |

## Why there are two price cut-offs
Genuine fans often sell a little below face value, for example at 80–95% to get rid of tickets quickly. So "under 80%" is only a **weak** sign. Under 50% is the "too good to be true" price that official warnings describe, so it is **strong**.

## Links
- GOV.UK / Action Fraud 2025: https://www.gov.uk/government/news/16m-lost-to-gig-ticket-scams-as-public-urged-to-take-caution
- Lloyds 2024 (Taylor Swift): https://www.lloydsbankinggroup.com/assets/pdfs/media/press-releases/2024-press-releases/lloyds-bank/2024.04.17-lloyds-bank-urgent-warning-over-taylor-swift-ticket-scams.pdf
- Lloyds 2026 (football): https://www.lloydsbank.com/news-and-insights/2026/football-fans-urged-to-show-fraudsters-the-red-card.html
- Santander 2023: https://www.santander.co.uk/about-santander/media-centre/press-releases/santander-issues-scam-warning-as-ticket-scams-double-in
- Which? 2025: https://www.which.co.uk/news/article/a-hacker-scammed-my-friends-on-instagram-with-fake-oasis-tickets-aqNjr2u6LbJy
- UK Finance Annual Fraud Report 2026: https://www.ukfinance.org.uk/policy-and-guidance/reports/annual-fraud-report-2026
- Oasis 2026 warning (The Manc): https://themanc.com/news/oasis-live-27-ticket-scams-warning/
- BBB 2022 (via JC Post): https://jcpost.com/posts/db7b1915-5bd3-4197-8b89-02e7d691fc04
- Eventbrite: https://www.eventbrite.com/blog/spot-ticket-fraud-social-media-ds00/
- How-To Geek: https://www.howtogeek.com/never-buy-tickets-on-social-media-scam/
- Music in Minnesota: https://www.musicinminnesota.com/facebook-ticket-scams-how-to-avoid-them/
- NOW Toronto: https://nowtoronto.com/music/features/facebook-ticket-scam-concerts/
- The Music AU 2020: https://themusic.com.au/news/facebook-ticket-scammers-fake-tickets/IJw4MjU0NzY/27-02-20
- WCPO 2024: https://www.wcpo.com/money/consumer/dont-waste-your-money/hacked-facebook-friends-offering-bogus-taylor-swift-tickets
- CBS 2023: https://www.cbsnews.com/boston/news/scammers-hack-facebook-accounts-fake-taylor-swift-tickets

**Honest limit for the report:** the tiers are a judgement made from these sources. They are not measured statistics. The training data then checks and adjusts them, and blind test #3 checks the result.
