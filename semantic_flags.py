"""
semantic_flags.py - v2.6: the AI "meaning matcher" (MiniLM sentence embeddings).

The problem in v2.5: a red flag only fired if the listing used words that were in the
rulebook. "Kindly DM if interested" slipped through because only "DM me" was listed.

How this works:
  * MiniLM (sentence-transformers/all-MiniLM-L6-v2, ~80 MB) is a small language model
    that runs on this computer - nothing is sent to an outside service. It turns a
    sentence into 384 numbers that capture its MEANING, so sentences that mean the
    same thing get similar numbers even when they share no words.
  * For each red flag we write a few EXAMPLE phrases ("prototypes"), e.g. for
    off_platform: "message me privately", "contact me on WhatsApp".
  * The listing is split into short clauses. Each clause is compared with every
    prototype (cosine similarity, 0 = unrelated, 1 = same meaning).
  * The flag fires if a clause is close enough to a red-flag prototype (similarity >=
    SIM_THRESHOLD, chosen by cross-validation in select_v26.py) AND closer to it than to
    the "normal listing" anchors (MARGIN) - this stops ordinary sentences such as
    "transfer through the Ticketmaster app" being mistaken for "bank transfer".

Design choices (say these in the report):
  * The AI helps SPOT flags; the expert-tier model still DECIDES, so every result can
    still be explained ("flag X fired because of this sentence, which means the same
    as that example").
  * Hard-stop flags (codes, hiding the payment reference...) stay rules-only: a fuzzy
    match should never force a "High risk" on its own.
  * If the model can't be loaded (no internet on first run, package missing) the
    checker carries on with the rules alone and says so - it never crashes.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.70   # chosen by source-grouped cross-validation (select_v26.py, best F1 of 0.55-0.70)
MARGIN = 0.05          # must beat the closest "normal listing" anchor by this much
CACHE_FILE = Path(__file__).parent / "model" / "embedding_cache.npz"

# Example phrases for each red flag. Rules-only flags (hard stops, price, seller,
# event, channel) are not here on purpose.
PROTOTYPES = {
    "off_platform": [
        "dm me", "message me privately", "send me a private message", "contact me on whatsapp",
        "text me on my number", "message me for the price", "add me on snapchat",
        "drop me a message if you are interested", "inbox me for details", "hit me up",
        "dm if interested", "price in dm", "contact me on telegram",
    ],
    "unsafe_payment": [
        "pay by bank transfer", "send the money to my bank account", "payment through revolut",
        "pay with paypal friends and family", "pay with gift cards", "pay in bitcoin or crypto",
        "payment via twint", "send it on cash app", "i will give you my sort code and account number",
        "pay me on monzo",
    ],
    "send_after_payment": [
        "i will send the tickets after you pay", "tickets sent once payment is received",
        "you get the tickets after paying", "i'll transfer them as soon as the money arrives",
    ],
    "deposit_or_hold": [
        "pay a deposit to hold them", "pay half now and the rest later",
        "send money upfront to reserve the tickets", "a small fee to secure them for you",
    ],
    "payment_switch": [
        "my paypal isn't working so can you pay by bank", "card payments are not working, use bank transfer",
        "paypal has blocked my account so please send it another way",
    ],
    "urgency": [
        "first to pay gets them", "lots of people are interested so be quick", "i need the payment today",
        "they will go fast", "serious buyers only, first come first served",
    ],
    "sob_story": [
        "i can't go anymore because of a family emergency", "my friend dropped out so i have spares",
        "i'm ill and can't attend", "work came up so i can't make it", "i'm stuck abroad and can't come back",
    ],
    "trust_claims": [
        "i am a legit seller", "this is not a scam", "100% genuine tickets, trust me",
        "loads of people can vouch for me",
    ],
}

# "Normal listing" sentences: a clause must be closer to a red flag than to these.
ANCHORS = [
    "tickets will be transferred through the ticketmaster app", "pay through paypal goods and services",
    "the seats are in block 112 row f", "selling tickets for the concert", "sold at face value",
    "the event is on friday at the arena", "listed on twickets at the original price",
    "standing tickets for the gig", "happy to meet in person at the venue",
    "transfer via the official resale site", "two tickets available", "hello, i am selling tickets",
]


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:   # offline first run, package missing...
        print(f"[semantic_flags] MiniLM not available ({e.__class__.__name__}); using rules only.")
        return None


def available() -> bool:
    return _model() is not None


_cache: dict[str, np.ndarray] | None = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            d = np.load(CACHE_FILE)
            _cache = {k: d[k] for k in d.files}
    return _cache


def save_cache():
    if _cache:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE_FILE, **_cache)


def embed(sentences: list[str]) -> np.ndarray:
    cache = _load_cache()
    keys = [hashlib.md5(s.encode()).hexdigest() for s in sentences]
    todo = [s for s, k in zip(sentences, keys) if k not in cache]
    if todo:
        vecs = _model().encode(todo, normalize_embeddings=True, batch_size=64)
        for s, v in zip(todo, vecs):
            cache[hashlib.md5(s.encode()).hexdigest()] = v.astype(np.float32)
    return np.array([cache[k] for k in keys])


@lru_cache(maxsize=1)
def _prototype_matrix():
    names, phrases = [], []
    for flag, ps in PROTOTYPES.items():
        names += [flag] * len(ps)
        phrases += ps
    return np.array(names), phrases, embed(phrases), embed(ANCHORS)


def clauses(text: str) -> list[str]:
    """Split a (normalised) listing into short clauses."""
    parts = re.split(r"[.!?;\n]+|,\s(?=[a-z])|\s-\s|\s\|\s", text)
    parts = [p.strip(" ,") for p in parts]
    return [p for p in parts if len(p.split()) >= 2][:30]


def semantic_flags(text: str, threshold: float | None = None, margin: float | None = None) -> dict:
    """Returns {flag: evidence} for flags whose meaning was found.
    evidence = (clause from the listing, closest example phrase, similarity)."""
    threshold = SIM_THRESHOLD if threshold is None else threshold
    margin = MARGIN if margin is None else margin
    if not available():
        return {}
    cl = clauses(text)
    if not cl:
        return {}
    names, phrases, P, A = _prototype_matrix()
    C = embed(cl)
    sim = C @ P.T                      # clause x prototype
    anchor = (C @ A.T).max(axis=1)     # closest normal sentence for each clause
    found = {}
    for flag in PROTOTYPES:
        cols = np.where(names == flag)[0]
        s = sim[:, cols]
        i, j = np.unravel_index(s.argmax(), s.shape)
        best = float(s[i, j])
        if best >= threshold and best >= anchor[i] + margin:
            found[flag] = (cl[i], phrases[cols[j]], round(best, 2))
    return found


if __name__ == "__main__":
    from normalise import normalise
    for s in ["Kindly Dm if interested", "HMU", "plsease dm", "slide into my dms",
              "Payment by TWINT only", "Tickets transferred via the Ticketmaster app",
              "pay with PayPal goods and services", "Block 112 row F, face value £85"]:
        print(f"{s!r:45} -> {semantic_flags(normalise(s))}")
