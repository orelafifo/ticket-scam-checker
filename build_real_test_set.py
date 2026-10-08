"""
build_real_test_set.py - a held-out test set based on REAL published scam examples.

Why: the model is trained on synthetic listings, so testing it on more synthetic
listings flatters it. This test set comes from outside the training data.

Three kinds of row (column `source_type`):
  verbatim       - scam wording quoted word-for-word in a published source
                   (some sources are US/Australian, which shows the scripts travel)
  reconstructed  - a listing rebuilt from a documented UK case or bank/police
                   warning, keeping only the details the source describes
  constructed_genuine - realistic genuine listings written to match how official
                   resale platforms and real fans list tickets (no verified real
                   genuine listings are publicly available - a stated limitation)

Pseudonymisation: names, handles and phone numbers are replaced with [SELLER],
[FRIEND], [PHONE]; seller details are approximate. Unknown seller details are
left blank and treated as neutral (not suspicious) by the evaluation.

The training templates were NOT reused: every row is worded independently.
Output: data/real_test_set.csv
"""
import csv
from pathlib import Path

LLOYDS_TS = "https://www.lloydsbankinggroup.com/assets/pdfs/media/press-releases/2024-press-releases/lloyds-bank/2024.04.17-lloyds-bank-urgent-warning-over-taylor-swift-ticket-scams.pdf"
LLOYDS_FB = "https://www.lloydsbank.com/news-and-insights/2026/football-fans-urged-to-show-fraudsters-the-red-card.html"
WHICH_OASIS = "https://www.which.co.uk/news/article/a-hacker-scammed-my-friends-on-instagram-with-fake-oasis-tickets-aqNjr2u6LbJy"
GOV_UK = "https://www.gov.uk/government/news/16m-lost-to-gig-ticket-scams-as-public-urged-to-take-caution"
SANTANDER = "https://www.santander.co.uk/about-santander/media-centre/press-releases/santander-issues-scam-warning-as-ticket-scams-double-in"
BBB = "https://jcpost.com/posts/db7b1915-5bd3-4197-8b89-02e7d691fc04"
OASIS26 = "https://themanc.com/news/oasis-live-27-ticket-scams-warning/"
WCPO = "https://www.wcpo.com/money/consumer/dont-waste-your-money/hacked-facebook-friends-offering-bogus-taylor-swift-tickets"
CBS = "https://www.cbsnews.com/boston/news/scammers-hack-facebook-accounts-fake-taylor-swift-tickets"
EVENTBRITE = "https://www.eventbrite.com/blog/spot-ticket-fraud-social-media-ds00/"
HTG = "https://www.howtogeek.com/never-buy-tickets-on-social-media-scam/"
NOW = "https://nowtoronto.com/music/features/facebook-ticket-scam-concerts/"
TMAU = "https://themusic.com.au/news/facebook-ticket-scammers-fake-tickets/IJw4MjU0NzY/27-02-20"
BLOG = "https://facebookeventtickerscammersaretakingover.home.blog/"
MIM = "https://www.musicinminnesota.com/facebook-ticket-scams-how-to-avoid-them/"

FB, IG, X, WA, TXT = ("Facebook (post, group or Marketplace)", "Instagram", "X (Twitter)",
                      "WhatsApp", "Text message / iMessage")
UNKNOWN_SITE = "A website I don't recognise"

# (source_type, source, url, platform, link, text, price, face, seat, age, followers, sudden)
# blank ("") = unknown
SCAMS = [
    # ---------------- verbatim / near-verbatim quotes ----------------
    ("verbatim", "The Music (AU), 2020", TMAU, FB, "", "Selling 4x tickets pm me.", "", "", 0, "", "", 0),
    ("verbatim", "The Music (AU), 2020", TMAU, FB, "",
     "Selling 4x tickets pm me. My PayPal wasn't set up for Goods & Services payments, I could only accept "
     "Friends & Family. Why do you have issues trusting people?", "", "", 0, "", "", 0),
    ("verbatim", "The Music (AU), 2020", TMAU, FB, "", "I have tickets for sale at a reduced price", "", "", 0, "", "", 0),
    ("verbatim", "Facebook scam blog, 2019", BLOG, FB, "",
     "Me and my family can't make it to this wonderful event. Kindly DM me for cheap tickets at a discount "
     "rate if you are interested", "", "", 0, "", "", 0),
    ("verbatim", "Facebook scam blog, 2019", BLOG, FB, "",
     "Yes, good seats, how many do you need? OK, they are digital tickets. I can send them to your email. You "
     "have PayPal? Make sure you use 'Friends & Family option'. I am selling tickets for two shows that I can "
     "no longer attend since my wife is having a baby. They are $80 but I will sell them to you for $60 since "
     "I do not want them to go to waste. Seats 15 & 16, Section 115, Row 11, send me a screenshot when you are done.",
     60, 80, 1, "", "", 0),
    ("verbatim", "Facebook scam blog, 2019", BLOG, FB, "",
     "I am a good, honest, Christian man. I would not try to scam you. You can trust me. You can send me half "
     "first. You will see the tickets after you send payment. I have been scammed before too. Buy the gift card "
     "and send me the code. Let me know when you are done and send me the screenshot, then I will send the tickets",
     "", "", 0, "", "", 0),
    ("verbatim", "Music in Minnesota", MIM, FB, "", "Family went out of town so I am unable to attend! PM me!",
     "", "", 0, 20, 15, 0),
    ("verbatim", "Music in Minnesota", MIM, FB, "", "My friends bailed on me so I'm selling our tickets CHEAP! DM me!",
     "", "", 0, 20, 15, 0),
    ("verbatim", "Music in Minnesota", MIM, FB, "",
     "How much are you willing to pay? I prefer you pay me with an Amazon card. You can get it at any store "
     "near you or online at Amazon.com!", "", "", 0, 20, 15, 0),
    ("verbatim", "NOW Toronto", NOW, FB, "", "I bought four tickets before I realized we had other plans that day.",
     "", "", 0, "", "", 0),
    ("verbatim", "NOW Toronto", NOW, FB, "",
     "I'm currently in the United States for surgery that's why I cannot attend event.", "", "", 0, "", "", 0),

    # ---------------- reconstructed from documented UK cases / warnings ----------------
    ("reconstructed", "Which? 2025 (hacked Instagram, Oasis)", WHICH_OASIS, IG, "",
     "Gutted, can't make Oasis at Heaton Park anymore 😭 2 tickets going at face value £150 each, bank transfer "
     "and I'll send them straight over. DM me quick, loads of interest", 150, 150, 0, 2500, 600, 1),
    ("reconstructed", "Which? 2025 (hacked Instagram, Oasis)", WHICH_OASIS, IG, "",
     "Hey! Random one but I've got 2 spare Oasis tickets, saw you were gutted you didn't get any. Can do them "
     "for what I paid. Just bank transfer me and I'll forward them on the app tonight", 150, 150, 0, 2500, 600, 1),
    ("reconstructed", "Lloyds 2024 (Taylor Swift, Facebook)", LLOYDS_TS, FB, "",
     "Taylor Swift Eras Tour Wembley x2, Saturday. Can't go anymore. £120 each (paid £190). Pics of the tickets "
     "attached. Payment by bank transfer, will send after.", 120, 190, 0, "", "", 0),
    ("reconstructed", "Lloyds 2024 (sold-out shows at inflated prices)", LLOYDS_TS, FB, "",
     "Sold out show!! Have 2 VIP standing for Taylor Swift at Wembley, £450 each. Pay upfront to secure, "
     "tickets emailed after.", 450, 190, 0, "", "", 0),
    ("reconstructed", "Lloyds 2026 (football: Facebook to WhatsApp)", LLOYDS_FB, FB, "",
     "2 x Arsenal v Liverpool, Clock End, £95 each. Message me on WhatsApp [PHONE] to sort, bank transfer only, "
     "will transfer to your app once paid", 95, 95, 0, "", "", 0),
    ("reconstructed", "Lloyds 2026 (football: FA Cup final)", LLOYDS_FB, FB, "",
     "Season ticket holder, can't make the FA Cup final so selling my seat for £120. Need payment today as "
     "others are asking.", 120, 120, 0, "", "", 0),
    ("reconstructed", "Lloyds 2026 (football: Champions League final)", LLOYDS_FB, X, "",
     "Champions League final tickets available, cat 2, DM for price. Payment via bank transfer or crypto.",
     "", "", 0, "", "", 0),
    ("reconstructed", "GOV.UK/Action Fraud 2025 (fake Twickets accounts)", GOV_UK, IG, "",
     "Twickets Official Resale: Oasis tickets now available at face value! Message us to reserve, pay by bank "
     "transfer to secure your booking.", "", "", 0, 30, 400, 0),
    ("reconstructed", "GOV.UK/Action Fraud 2025 (fake Twickets websites)", GOV_UK, UNKNOWN_SITE,
     "https://twickets-oasis-resale.com/wembley",
     "Official fan-to-fan resale. Oasis Wembley, Standing, £148.50. Checkout now, limited availability.",
     148.5, 148.5, 0, "", "", 0),
    ("reconstructed", "GOV.UK/Action Fraud 2025 (pushed off payment platforms)", GOV_UK, X, "",
     "Saw your comment on the Coldplay post, I have 2 tickets I can't use. PayPal is blocking me at the moment "
     "so it would need to be bank transfer or USDT. Can send screenshots as proof.", "", "", 0, "", "", 0),
    ("reconstructed", "GOV.UK/Action Fraud 2025 (last-minute fake ticket companies)", GOV_UK, UNKNOWN_SITE,
     "https://glasto-tickets-2026.co.uk",
     "Last minute Glastonbury tickets with coach travel, £395 inc. Limited release, book via our website "
     "before they're gone", 395, 395, 0, "", "", 0),
    ("reconstructed", "Santander 2023 (social media pressure)", SANTANDER, FB, "",
     "There's a lot of interest in these so I need the full amount upfront today to hold them for you. "
     "2 x Central Cee, O2, £80 each.", 80, 80, 0, "", "", 0),
    ("reconstructed", "Santander 2023 (fake websites, not yet on sale)", SANTANDER, UNKNOWN_SITE,
     "https://beyonce-uk-tour-tickets.com",
     "Pre-sale tickets for Beyonce's 2027 UK tour now available, guaranteed entry. Book now before general sale.",
     "", "", 0, "", "", 0),
    ("reconstructed", "Santander 2023 (no-show handover)", SANTANDER, FB, "",
     "Can meet you outside the venue to hand them over, just need a deposit first so I know you're serious",
     "", "", 0, "", "", 0),
    ("reconstructed", "BBB 2022 (Ticketmaster lookalike sites)", BBB, UNKNOWN_SITE,
     "https://www.ticketfaster.co.uk/tame-impala",
     "Tame Impala - O2 Arena. Best available seats from £189. Secure your tickets now.", 189, 75, 1, "", "", 0),
    ("reconstructed", "Oasis warning 2026 (codes for sale)", OASIS26, X, "",
     "Got a spare Oasis presale code for Live '27, £40 and I'll send it over now. Bank transfer please.",
     "", "", 0, "", "", 0),
    ("reconstructed", "Oasis warning 2026 (fake code messages)", OASIS26, TXT, "https://oasis-live27-access.com",
     "Oasis Live '27: you've been selected for priority access. Pay the £25 registration fee here to receive "
     "your code", "", "", 0, "", "", 0),
    ("reconstructed", "WCPO 2024 (hacked friend, face value)", WCPO, FB, "",
     "Hey everyone, I've got Taylor Swift tickets at face value, can't use them. First come first served, I'll "
     "take payment by bank transfer.", "", "", 0, 2000, 500, 1),
    ("reconstructed", "CBS 2023 (hacked account in local groups)", CBS, FB, "",
     "Selling 4 tickets for the Stray Kids show, can't go. Message me, I accept bank transfer or Revolut.",
     "", "", 0, 2000, 350, 1),
    ("reconstructed", "Eventbrite (unsolicited Instagram DM)", EVENTBRITE, IG, "",
     "hey saw u wanted tickets for the sold out show, i have 1 floor ticket, £150, dm me and i'll send it after u pay",
     150, 150, 0, 15, 10, 0),
    ("reconstructed", "How-To Geek (excess tickets, copied posts)", HTG, X, "",
     "Accidentally bought too many tickets for Sabrina Carpenter at the O2, selling 2 at a discount. Can send "
     "screenshots of the order.", "", "", 0, 25, 20, 0),
    ("reconstructed", "How-To Geek (fell ill)", HTG, X, "",
     "I've come down with flu so can't make it to Fred again.. tonight, 1 ticket going cheap, quick sale. PM me.",
     "", "", 0, "", "", 0),
    ("reconstructed", "NOW Toronto (paying into a friend's account)", NOW, FB, "",
     "Hi, I'm selling 2 tickets for Arctic Monkeys. I'm abroad for surgery so can't attend. You can pay into my "
     "friend's account as mine is frozen.", "", "", 0, "", "", 0),
    ("reconstructed", "The Music AU (like and comment first)", TMAU, FB, "",
     "Comment 'interested' and like this post and I'll message you! 4 tickets for Dua Lipa, reduced price.",
     "", "", 0, "", "", 0),
    ("reconstructed", "NOW Toronto (wrong currency, emailed tickets)", NOW, FB, "",
     "2 tickets for Rema in Manchester, 25 euros each, I can send by email after paypal friends and family",
     "", "", 0, "", "", 0),
    ("reconstructed", "Lloyds 2024 (pictures of real tickets)", LLOYDS_TS, FB, "",
     "Pic attached of the actual tickets. Burna Boy, London Stadium, block 223 row K. £70 each. Bank transfer "
     "only, sending the PDF straight after.", 70, 70, 1, "", "", 0),
    ("reconstructed", "Lloyds 2025 (Oasis, inflated prices)", LLOYDS_TS, FB, "",
     "Oasis Wembley pitch standing x4, £600 each, guaranteed. Payment via bank transfer to secure.",
     600, 150, 0, "", "", 0),
    ("reconstructed", "GOV.UK/Action Fraud 2025 (fake Twickets messages)", GOV_UK, TXT,
     "https://twickets-confirm.com/order",
     "This is the Twickets resale team. Your waiting list tickets are ready, please confirm with a payment of "
     "£89 using the link", 89, 89, 0, "", "", 0),
    ("reconstructed", "Eventbrite (spoofed transfer / account takeover)", EVENTBRITE, WA, "",
     "To get the tickets I need to transfer them to your Ticketmaster account, can you send me the 6 digit "
     "code you'll get by text?", "", "", 0, "", "", 0),
]

TM, TWK, DICE, AXS, VIA, STUB = ("Ticketmaster / Ticketmaster Resale", "Twickets", "DICE",
                                 "AXS / AXS Official Resale", "viagogo", "StubHub")
OFFICIAL_SELLER = (3650, 1000)   # platform listings: seller account details not applicable
PERSON = (1800, 400)             # ordinary fan account

# (platform, link, text, price, face, seat, (age, followers))
GENUINE = [
    (TWK, "https://www.twickets.live/en/uk/event/123", "2 x Standing tickets, Tems, O2 Arena, Sat 6 Feb. Original price "
     "£72.45 each. Tickets will be transferred via Ticketmaster after purchase.", 72.45, 72.45, 0, OFFICIAL_SELLER),
    (TM, "", "Fan-to-Fan Resale. Section 104, Row L, Seats 12-13. £95.00 each incl. fees.", 95, 95, 1, OFFICIAL_SELLER),
    (DICE, "https://dice.fm/event/abc", "Ticket available from the waiting list: Fontaines D.C., Alexandra Palace, "
     "General Admission, £45.50", 45.5, 45.5, 0, OFFICIAL_SELLER),
    (AXS, "", "AXS Official Resale - Block 115 Row F, 2 tickets, £78 each (face value).", 78, 78, 1, OFFICIAL_SELLER),
    (X, "", "selling 1 ticket for boygenius at ally pally, can't go anymore cos of a uni deadline. face value (£55), "
     "will transfer through ticketmaster, paypal goods & services so you're covered. dm me", 55, 55, 0, PERSON),
    (FB, "", "Hi all, I have 2 seated tickets for Coldplay at Wembley on the 22nd, block 506 row 22. My sister's wedding "
     "got moved so we can't go. £85 each which is what we paid, happy to list them on Twickets so you're "
     "protected.", 85, 85, 1, PERSON),
    (IG, "", "Spare ticket for Raye at the O2 next Friday, standing. Selling for face value, I'll put it on Twickets "
     "tonight, link in bio once it's up", 65, 65, 0, PERSON),
    (WA, "", "Hey guys, [FRIEND] can't come to Wizkid anymore so we've got a spare. £65, just bank transfer me and I'll "
     "transfer it on the app straight away", 65, 65, 0, PERSON),
    (VIA, "https://www.viagogo.co.uk/Concert-Tickets/Rock/Arctic-Monkeys", "2 tickets, Section 112 Row M, Arctic "
     "Monkeys, Emirates Stadium. £210 per ticket. Instant download available.", 210, 85, 1, OFFICIAL_SELLER),
    (STUB, "https://www.stubhub.co.uk/event/1", "Category 1, Block 120, 2 tickets together. £340 each. Guaranteed by "
     "StubHub FanProtect.", 340, 120, 1, OFFICIAL_SELLER),
    (X, "", "Selling 2x Charli xcx tickets for Co-op Live, block 213 row C. Paid £68, selling for £70 to cover fees. "
     "Ticketmaster transfer, paypal g&s.", 70, 68, 1, PERSON),
    ("Gumtree or other classified ads", "", "2 standing tickets for Kasabian at Leicester. Face value £60. Can meet at "
     "the box office and transfer via the app in person.", 60, 60, 0, PERSON),
    ("TikTok", "", "can't go to the Olivia Rodrigo show anymore, putting my 2 tickets on Ticketmaster resale tonight "
     "at face value, keep an eye out!", 70, 70, 0, PERSON),
    (FB, "", "Need gone today! 1 standing ticket for Bicep, face value £40. Transfer on DICE. PayPal G&S.",
     40, 40, 0, PERSON),
    (TXT, "", "Hiya it's me, do you still want my Blur ticket for Wembley? £90, it's in my Ticketmaster app so I'll "
     "transfer it once you've paid. Bank or PayPal whatever's easier", 90, 90, 0, PERSON),
    ("Skiddle", "https://www.skiddle.com/festivals/parklife", "Resale ticket: Parklife Saturday Day Ticket, £79.50, "
     "sold via Skiddle official resale", 79.5, 79.5, 0, OFFICIAL_SELLER),
    ("See Tickets", "https://www.seetickets.com/event/wireless", "Wireless Festival Friday, 1 x General Admission, £95 "
     "+ booking fee, e-ticket sent by See Tickets", 95, 95, 0, OFFICIAL_SELLER),
    (IG, "", "selling 2 tickets for Tyler, the Creator at the O2, block 401 row G. £80 each (face). I've sold on here "
     "before, happy to do paypal goods & services + ticketmaster transfer so it's safe for both of us",
     80, 80, 1, PERSON),
    (FB, "", "Can't go to Lewis Capaldi in Glasgow, 1 ticket, will take £40 (paid £60) just want someone to use it. "
     "Ticketmaster transfer, paypal goods and services.", 40, 60, 0, PERSON),
    (X, "", "anyone want a ticket for Little Simz at the Roundhouse tomorrow? standing, £35 face value, will send via "
     "DICE (it's the only way it works anyway)", 35, 35, 0, PERSON),
    (FB, "", "2x Hozier tickets, OVO Arena Wembley, block D row 8. £70 each. Tickets are on the AXS app, I'll transfer "
     "once paid through PayPal goods and services.", 70, 70, 1, PERSON),
    (TWK, "https://www.twickets.live/en/uk/event/456", "1 x Seated, Block 212 Row P, Lana Del Rey, Hyde Park. Price "
     "£89.50 (original price). Delivery: Ticketmaster transfer.", 89.5, 89.5, 1, OFFICIAL_SELLER),
    (TM, "https://www.ticketmaster.co.uk/event/789", "Resale price £60.00 (original £72.00). Standing. Sabrina "
     "Carpenter, O2.", 60, 72, 0, OFFICIAL_SELLER),
    (DICE, "", "Waiting list ticket released: Ezra Collective, Brixton Academy, £38.75", 38.75, 38.75, 0, OFFICIAL_SELLER),
    (IG, "", "Spare ticket for Loyle Carner at Ally Pally, face value £50, standing. I can transfer on Ticketmaster "
     "first and you send the money after, I trust people on here", 50, 50, 0, PERSON),
    (X, "", "if anyone needs a ticket for the Arsenal game sat, I've got a spare on my membership, can only transfer "
     "to another member through the club app, face value £72", 72, 72, 0, PERSON),
    (FB, "", "Selling 2 tickets to Hamilton at Victoria Palace, stalls row H seats 5-6, £120 each (face value). Can "
     "transfer through the ATG app, payment by PayPal goods and services.", 120, 120, 1, PERSON),
    ("Snapchat", "", "my mate dropped out of Reading, his weekend ticket is £300, he'll transfer it on the app, "
     "payment through paypal G&S", 300, 300, 0, PERSON),
    (VIA, "https://www.viagogo.com/Festival-Tickets/Creamfields", "Creamfields 4-day camping, £320 each, instant "
     "download, viagogo guarantee", 320, 260, 0, OFFICIAL_SELLER),
    (STUB, "https://www.stubhub.co.uk/event/2", "England v Ireland, Twickenham, Block L3, £180 per ticket.",
     180, 110, 1, OFFICIAL_SELLER),
    ("Other official seller (Eventim, Gigantic...)", "https://www.gigantic.com/nothing-but-thieves",
     "Nothing But Thieves, Manchester Academy, General Admission £35 + fee", 35, 35, 0, OFFICIAL_SELLER),
    (FB, "", "2 tickets for Sam Fender, St James Park. Paid £85 each, will take £80. Happy to do bank transfer if "
     "easier, or paypal G&S, and I'll send via Ticketmaster first if you'd prefer", 80, 85, 0, PERSON),
    (X, "", "1 ticket for Wet Leg at the Forum tonight at face value £30 to a fan, must be able to accept a dice "
     "transfer", 30, 30, 0, PERSON),
    (TM, "", "Ticketmaster Resale, 2 x seated, Block 101 Row B, Pet Shop Boys, O2, £84.15 each", 84.15, 84.15, 1,
     OFFICIAL_SELLER),
    (AXS, "https://www.axs.com/uk/events/1", "AXS Resale: Kendrick Lamar, Tottenham Hotspur Stadium, Block 520, £110",
     110, 110, 1, OFFICIAL_SELLER),
    (FB, "", "Spare ticket for Stormzy at the O2, block 109 row T, face value £75, PayPal goods & services and "
     "Ticketmaster transfer only please, no bank transfers", 75, 75, 1, PERSON),
    (WA, "", "Mum's not well so we can't go to Elton John, 2 tickets, £120 each, I'll transfer them on the "
     "Ticketmaster app and you can pay me after", 120, 120, 0, PERSON),
    (IG, "", "Have 2 spare for Beabadoobee at Brixton, standing, £40 each face value. Transfer via AXS, paypal g&s, "
     "can show my order email.", 40, 40, 0, PERSON),
    (FB, "", "Selling my 2 Dave tickets for the O2, block 113 row J. Paid £70 each, selling at £70. I'll list them "
     "on Twickets, just reply and I'll send you the link when they're up.", 70, 70, 1, PERSON),
    (X, "", "got a spare for SZA at Tottenham, seated, £95 which is face. it's an AXS ticket so I'll transfer through "
     "AXS, paypal goods and services only", 95, 95, 0, PERSON),
]

rows = []
for i, (stype, src, url, plat, link, text, price, face, seat, age, foll, sudden) in enumerate(SCAMS, 1):
    rows.append(dict(id=f"S{i:02d}", label=1, source_type=stype, source=src, source_url=url, platform=plat,
                     link=link, text=text, price=price, face_value=face, has_seat_details=seat,
                     account_age_days=age, followers=foll, sudden_seller=sudden))
for i, (plat, link, text, price, face, seat, (age, foll)) in enumerate(GENUINE, 1):
    rows.append(dict(id=f"G{i:02d}", label=0, source_type="constructed_genuine",
                     source="Constructed from official resale formats and typical fan posts", source_url="",
                     platform=plat, link=link, text=text, price=price, face_value=face, has_seat_details=seat,
                     account_age_days=age, followers=foll, sudden_seller=0))

Path("data").mkdir(exist_ok=True)
with open("data/real_test_set.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"Saved {len(rows)} rows to data/real_test_set.csv "
      f"({sum(r['label'] for r in rows)} scams, {sum(1 - r['label'] for r in rows)} genuine)")
