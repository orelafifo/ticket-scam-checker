"""
step1_make_dataset.py - build a SYNTHETIC labelled dataset of resale listings.

There is no public dataset of labelled ticket-scam listings, so we generate one
based on published scam indicators. Be upfront about this in your video: it's a
limitation, and "real data via a bank/promoter partnership" is your next phase.

Changes in v2:
  * Listings are assembled from mix-and-match phrase pools (opener + details +
    reason + payment + delivery + extras) instead of 8 fixed templates. This
    reduces "template leakage" (the model learning one template's wording).
  * New red flags appear in the text: payment switch, deposit/hold, hidden
    payment reference, "100% legit" trust claims, asking for codes, PDF tickets.
  * Each row also gets seller + event-check fields (sudden_seller, event found,
    details match, high demand, before on-sale, app-only tickets).
  * v2.2: each row also records WHERE it was sold (official / secondary resale /
    social media & messaging / unknown website) plus link warning signs.

To stop an unrealistic 100%, we add overlap on purpose: messy genuine sellers
(bank transfer, urgency, WhatsApp) and sneaky scams (polished, seat details,
only one red flag), plus ~3% label noise.

Output: data/listings.csv
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

ARTISTS = ["Burna Boy", "Tems", "SZA", "Stray Kids", "SEVENTEEN", "Beyonce",
           "Central Cee", "Rema", "Olivia Rodrigo", "BLACKPINK", "Dave", "Coldplay"]
VENUES = ["O2 Arena", "Wembley Stadium", "OVO Arena Wembley", "Co-op Live",
          "Tottenham Hotspur Stadium", "AO Arena", "Utilita Arena"]

# ---------------------------------------------------------------- phrase pools
OPENERS = [
    "Selling {n} tickets for {artist} at {venue}.",
    "{n}x {artist} {venue}.",
    "Spare tickets for {artist}, {venue}.",
    "Got {n} {artist} tickets for {venue} I can't use.",
    "{artist} @ {venue}, {n} tickets available.",
    "Anyone want {artist} tickets? {venue}.",
]
SCAM_OPENERS = OPENERS + [
    "{artist} tickets at {venue}!!",
    "Last minute {n}x {artist} at {venue}.",
    "SELLING {artist} {venue} TICKETS",
]
GENUINE_DETAILS = [
    "{section}. £{price} each, face value.",
    "{section}, paid £{price}, selling for the same.",
    "£{price} each, {section}.",
    "{section}. Selling at what I paid (£{price}).",
]
SCAM_DETAILS = [
    "Only £{price}!!",
    "£{price} each.",
    "Cheap at £{price}.",
    "{section}. £{price} each.",
]
GENUINE_REASONS = ["Can't make the date anymore.", "Can't go anymore, gutted.", "Work came up so can't make it.", "Clashes with a wedding.", "Friend dropped out.",
                   "Bought too many in the presale.", "Moving away that month.", ""]
SCAM_REASONS = ["Can't go anymore due to a family emergency.", "Work came up last minute.",
                "I'm ill and can't go.", "Can't go anymore.", ""]
PROTECTED_PAYMENT = [
    "PayPal goods and services only.",
    "Happy to use PayPal G&S so you're covered.",
    "Will list them on Twickets so you're protected.",
    "Can do it through the Ticketmaster resale so it's safe for both of us.",
    "Paying through the official exchange is fine.",
]
UNSAFE_PAYMENT = [
    "Bank transfer only.",
    "PayPal friends and family only.",
    "Revolut or bank transfer.",
    "Crypto or gift cards accepted.",
    # payment switch
    "PayPal isn't working on my account so bank transfer instead.",
    "Can't take PayPal at the moment, bank instead please.",
    # deposit / hold
    "£20 deposit to hold them for you, rest before the show.",
    "Pay half now and half when you get them.",
    # hide reference
    "Don't put tickets in the reference or PayPal holds the money.",
    "Send it as a gift so there are no fees.",
]
GENUINE_DELIVERY = [
    "Tickets transferred through the Ticketmaster app.",
    "Transfer on the official app straight away.",
    "Happy to show proof of purchase.",
    "Can transfer as soon as payment goes through the platform.",
]
SCAM_DELIVERY = [
    "Will send after payment.",
    "Tickets sent once paid.",
    "I'll email you the PDF tickets once paid.",
    "Can send screenshots of the tickets.",
    "I'll transfer them after you pay.",
]
SCAM_EXTRAS = [
    "Loads of people messaging, first to pay gets them.",
    "Serious buyers only.",
    "DM me on WhatsApp.",
    "Message me on Telegram.",
    "100% legit, can vouch.",
    "Genuine seller, not a scam.",
    "I'll send you a 6 digit code to check you're real, just send the code back.",
    "Won't last!!",
]
# Genuine-but-messy extras: realistic overlap with scam wording
MESSY_EXTRAS = [
    "Bank transfer is easiest for me but can do PayPal.",
    "Need them gone quick.",
    "Message me on WhatsApp if it's easier.",
    "Can send a screenshot of my order confirmation.",
    "Genuine seller btw, check my profile.",
    "Tickets are PDFs by email for this venue.",  # only used when event isn't app-only
]
# Sneaky scams: polished, protected-sounding, but one quiet red flag
SNEAKY_FLAGS = [
    "Happy to transfer on the app, just prefer friends and family to avoid fees.",
    "Easier to chat on WhatsApp, payment by bank transfer.",
    "Small deposit to hold them and I'll transfer on the app.",
    "PayPal is limited on my account right now so bank transfer instead.",
    "Please leave the reference blank, PayPal is funny about tickets.",
    random.choice(["Can email the PDF tickets over tonight.", "Will send the tickets as a PDF.",
                   "They're e-tickets, I'll forward you the PDF.", "PDF tickets, sent by email."]),
]


def section():
    return random.choice([f"Block {random.randint(101, 430)} Row {random.choice('ABCDEFGHJK')} "
                          f"Seats {random.randint(1, 30)}-{random.randint(31, 40)}",
                          "Standing", "Floor standing", "Lower tier"])


def pick(pool, p=1.0):
    return random.choice(pool) if random.random() < p else ""


def event_fields(is_scam: bool, mentions_pdf: bool) -> dict:
    """Simulate what event_check.py would return. found: 1 yes, 0 no, -1 couldn't check."""
    r = random.random()
    if is_scam:
        found = -1 if r < 0.10 else 0 if r < 0.25 else 1
        details_match = int(random.random() > 0.25) if found == 1 else -1
        high_demand = int(random.random() < 0.80)
        before_on_sale = int(random.random() < 0.12)
        app_only = int(random.random() < 0.85)
    else:
        found = -1 if r < 0.06 else 0 if r < 0.09 else 1       # not found = API coverage gap
        details_match = int(random.random() > 0.03) if found == 1 else -1
        high_demand = int(random.random() < 0.50)
        before_on_sale = int(random.random() < 0.01)
        app_only = 0 if mentions_pdf else int(random.random() < 0.85)
    return dict(event_found=found, event_details_match=details_match, event_high_demand=high_demand,
                event_before_on_sale=before_on_sale, event_app_only=app_only)


def channel_fields(kind: str) -> dict:
    """Simulate WHERE the listing was found (what channels.py would return).
    Shares are illustrative, guided by bank reports that most ticket scams start
    on social media; genuine fans also sell on social media, so there is overlap."""
    shares = {  # official, secondary, social, unknown_site, not given
        "genuine": [0.35, 0.15, 0.40, 0.03, 0.07],
        "messy":   [0.10, 0.10, 0.70, 0.00, 0.10],
        "scam":    [0.02, 0.06, 0.70, 0.12, 0.10],
        "sneaky":  [0.03, 0.15, 0.60, 0.15, 0.07],
    }[kind]
    cat = random.choices(["official", "secondary", "social", "unknown_site", "none"], weights=shares)[0]
    is_scam = kind in ("scam", "sneaky")
    site = cat == "unknown_site"
    return dict(
        ch_category=cat,
        ch_lookalike=int(site and is_scam and random.random() < 0.5),
        ch_new_domain=int(site and random.random() < (0.8 if is_scam else 0.2)),
        ch_insecure_or_short=int(site and random.random() < (0.4 if is_scam else 0.05)),
    )


def make(kind: str) -> dict:
    """kind: genuine | messy | scam | sneaky"""
    is_scam = kind in ("scam", "sneaky")
    face = random.choice([45, 55, 70, 85, 95, 120, 150])
    if kind == "scam":
        price = int(face * random.uniform(0.35, 0.85))
    elif kind == "sneaky":
        price = int(face * random.uniform(0.85, 1.05))
    else:
        price = int(face * random.uniform(0.9, 1.1))
    seat = section() if (kind != "scam" or random.random() < 0.3) else "Standing"

    if kind == "genuine":
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS),
                 pick(PROTECTED_PAYMENT), pick(GENUINE_DELIVERY, 0.8)]
    elif kind == "messy":
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS),
                 pick(MESSY_EXTRAS), pick(PROTECTED_PAYMENT, 0.4), pick(GENUINE_DELIVERY, 0.6)]
    elif kind == "scam":
        parts = [pick(SCAM_OPENERS), pick(SCAM_DETAILS), pick(SCAM_REASONS),
                 pick(UNSAFE_PAYMENT), pick(SCAM_DELIVERY, 0.8),
                 pick(SCAM_EXTRAS, 0.7), pick(SCAM_EXTRAS, 0.3)]
    else:  # sneaky
        flag = pick(SNEAKY_FLAGS)
        if "PDF" in flag:  # vary the wording so the model can't memorise one sentence
            flag = random.choice(["Can email the PDF tickets over tonight.", "Will send the tickets as a PDF.",
                                  "They're e-tickets, I'll forward you the PDF.", "PDF tickets, sent by email."])
        parts = [pick(OPENERS), pick(GENUINE_DETAILS), pick(GENUINE_REASONS, 0.5), flag]

    text = " ".join(p for p in parts if p).format(
        n=random.choice([1, 2, 2, 3, 4]), artist=random.choice(ARTISTS),
        venue=random.choice(VENUES), section=seat, price=price)
    has_seat = int("Row" in seat)
    mentions_pdf = "pdf" in text.lower()

    # --- seller account ---
    sudden = 0
    if is_scam:
        sudden = int(random.random() < 0.25)      # hacked friend's account
        if sudden:
            age, followers = random.randint(400, 3000), random.randint(150, 1500)
        else:
            age = random.choice([random.randint(1, 45), random.randint(1, 45), random.randint(100, 900)])
            followers = random.choice([random.randint(0, 40), random.randint(0, 40), random.randint(80, 600)])
    else:
        sudden = int(random.random() < 0.03)
        age = random.choice([random.randint(200, 3000), random.randint(200, 3000), random.randint(10, 60)])
        followers = random.choice([random.randint(100, 1500), random.randint(100, 1500), random.randint(5, 40)])

    return dict(text=text, account_age_days=age, followers=followers, sudden_seller=sudden,
                price_ratio=round(price / face, 2), has_seat_details=has_seat,
                **event_fields(is_scam, mentions_pdf), **channel_fields(kind), label=int(is_scam))


rows = ([make("genuine") for _ in range(240)] + [make("messy") for _ in range(70)] +
        [make("scam") for _ in range(220)] + [make("sneaky") for _ in range(80)])

# ~3% label noise, because real-world labels are never perfect
df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
flip = df.sample(frac=0.03, random_state=1).index
df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

Path("data").mkdir(exist_ok=True)
df.to_csv("data/listings.csv", index=False)
print(f"Saved {len(df)} listings to data/listings.csv  "
      f"(scams: {df.label.sum()}, genuine: {(1 - df.label).sum()})")
