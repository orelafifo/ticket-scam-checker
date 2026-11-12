"""
normalise.py - v2.6 step 1: tidy the listing text before anything reads it.

Scam posts are written in chat-speak: "hmu", "plz dm", "pleaaase", "F&F", emojis.
The red-flag rules and the AI meaning-matcher (semantic_flags.py) both work much
better on normal English, so we rewrite the text first:

  1. lower case, emojis and odd symbols removed, "!!!" -> "!"
  2. stretched words shortened  ("pleaaase" -> "please")
  3. slang / abbreviations expanded from a small dictionary  ("hmu" -> "hit me up")
  4. small typos in KEY words fixed with fuzzy matching  ("plsease" -> "please",
     "revolute" -> "revolut")  - only for a short list of words that matter, so we
     never "correct" names of artists or venues.

The original text is kept for display; only the checker reads the normalised copy.
"""
from __future__ import annotations

import re
from difflib import get_close_matches

# Abbreviation -> what it means. Whole words only. Add new ones here (human approved).
SLANG = {
    # contact / moving the chat
    "hmu": "hit me up", "hml": "hit me up", "dm": "dm", "dms": "dm", "d.m": "dm", "d.m.": "dm",
    "pm": "pm", "pms": "pm", "msg": "message", "mssg": "message", "inbx": "inbox",
    "whatsap": "whatsapp", "watsapp": "whatsapp", "whatsaap": "whatsapp", "wapp": "whatsapp",
    "tele": "telegram",
    # politeness / filler
    "pls": "please", "plz": "please", "plez": "please", "pleasee": "please", "kindly": "kindly",
    "u": "you", "ur": "your", "ya": "you", "thx": "thanks", "ty": "thanks",
    "asap": "asap", "rn": "right now", "tmrw": "tomorrow", "tmr": "tomorrow", "2moro": "tomorrow",
    # tickets
    "tix": "tickets", "tkts": "tickets", "tkt": "ticket", "tics": "tickets", "ticks": "tickets",
    # payment
    "f&f": "friends and family", "fnf": "friends and family", "f+f": "friends and family",
    "ff": "friends and family", "g&s": "goods and services", "gns": "goods and services",
    "revo": "revolut", "cashapp": "cash app",
    "btc": "bitcoin", "eth": "ethereum",
}

# Words we are allowed to fuzzy-correct (typos in these change the meaning for the checker).
KEY_WORDS = ["please", "message", "messages", "interested", "tickets", "ticket", "whatsapp",
             "telegram", "snapchat", "revolut", "monzo", "transfer", "payment", "deposit",
             "paypal", "friends", "family", "inbox", "private", "privately", "upfront",
             "legit", "genuine", "screenshot", "screenshots", "selling", "available"]

_MULTI = [  # multi-word phrases, applied before word-by-word slang
    (r"\bf\s*&\s*f\b", "friends and family"), (r"\bf\s*n\s*f\b", "friends and family"),
    (r"\bg\s*&\s*s\b", "goods and services"), (r"\bd\.?\s?m\.?\b", "dm"),
    (r"\bhit me up\b", "hit me up"), (r"\bslide (in|into) (my|me) dms?\b", "dm me"),
    (r"\bdrop (me )?a dm\b", "dm me"), (r"\bshoot (me )?a (dm|message|text)\b", "message me"),
    (r"\bcash ?app\b", "cash app"),
]
_WORD = re.compile(r"[a-z0-9&+.'£$€@/:-]+|[^\sa-z0-9]")


def normalise(text: str) -> str:
    t = str(text or "").lower()
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"[^\x00-\x7f£€]", " ", t)          # emojis and other symbols
    t = re.sub(r"([!?.])\1+", r"\1", t)              # "!!!" -> "!"
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)         # "pleaaase" -> "pleaase"
    for pattern, repl in _MULTI:
        t = re.sub(pattern, repl, t)

    out = []
    for tok in _WORD.findall(t):
        core = tok.strip(".,:;'")
        if core in SLANG:
            tok = tok.replace(core, SLANG[core])
        elif core.isalpha() and len(core) >= 5 and core not in KEY_WORDS:
            match = get_close_matches(core, KEY_WORDS, n=1, cutoff=0.84)
            if match:
                tok = tok.replace(core, match[0])
        out.append(tok)
    t = " ".join(out)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    for s in ["Kindly Dm if interested", "HMU!!! 🔥🔥", "plsease dm me", "pleaaase inbox me",
              "pay via F&F or revolute", "slide into my dms for the price", "2 tix for Burna Boy"]:
        print(f"{s!r:40} -> {normalise(s)!r}")
