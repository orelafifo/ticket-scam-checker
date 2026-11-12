"""
extract_event.py - v2.6: read the artist / team, venue and date out of the listing text.

In v2.5 the event check only ran if the buyer typed the artist, venue and date into
three boxes - most people didn't, so the check rarely ran. Now the checker reads them
from the listing itself.

How (all on this computer, nothing sent anywhere):
  1. spaCy's English model (en_core_web_sm) does Named Entity Recognition (NER): a
     trained AI model that labels people (PERSON), organisations (ORG), places
     (FAC / GPE) and dates (DATE) in a sentence.
  2. NER alone is weak on venue names ("O2 Arena" was labelled a PERSON in testing),
     so an EntityRuler adds a list of known UK venues, plus
     patterns like "@ The O2" and "Arsenal v Chelsea".
  3. The date text ("Friday 9th April 2027") is turned into a real date.
The buyer can always correct what was read (the boxes are under "Add more details").
"""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

VENUES = [
    "The O2", "O2 Arena", "Wembley Stadium", "Wembley", "OVO Arena Wembley", "Co-op Live", "AO Arena",
    "Manchester Arena", "Utilita Arena", "Tottenham Hotspur Stadium", "Emirates Stadium", "Emirates",
    "Old Trafford", "Anfield", "Etihad Stadium", "Stamford Bridge", "London Stadium", "Villa Park",
    "Goodison Park", "St James' Park", "Twickenham", "Hyde Park", "Royal Albert Hall", "O2 Academy Brixton",
    "Brixton Academy", "Alexandra Palace", "Eventim Apollo", "Hammersmith Apollo", "OVO Hydro", "Hydro",
    "Principality Stadium", "Murrayfield", "Bp Pulse Live", "Resorts World Arena", "First Direct Arena",
    "Motorpoint Arena", "Cardiff International Arena", "SSE Arena", "3Arena", "Roundhouse", "Printworks",
    "Kia Oval", "Lord's", "Wimbledon", "Silverstone", "Ascot",
]
VENUE_WORDS = r"(arena|stadium|academy|hall|park|apollo|live|centre|center|theatre|theater|o2|hydro|palace|field)"
CAP = r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,2}"
VS = re.compile(rf"({CAP})\s+(?:v|vs|versus)\.?\s+({CAP})")
BEFORE_TICKETS = re.compile(rf"({CAP})\s+(?:tickets?|tix|tkts)\b")
NOT_ARTIST = re.compile(r"^(selling|spare|hello|hi|hey|i'm|two|three|four|my|the|for|got|have)$", re.I)
AT = re.compile(r"(?:@|\bat)\s+((?:the\s+)?[A-Z0-9][\w'&-]*(?:\s+[A-Z0-9][\w'&-]*){0,3})")


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[extract_event] spaCy model not available ({e.__class__.__name__}); using patterns only.")
        return None
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "FAC", "pattern": v} for v in VENUES]
    patterns += [{"label": "FAC", "pattern": [{"LOWER": w.lower()} for w in v.split()]} for v in VENUES]
    ruler.add_patterns(patterns)
    return nlp


def _parse_date(s: str, today: date) -> date | None:
    from dateutil import parser
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s, flags=re.I)
    try:
        d = parser.parse(s, fuzzy=True, dayfirst=True, default=pd.Timestamp(today.year, 1, 1)).date()
    except (ValueError, OverflowError):
        return None
    if not re.search(r"\b20\d\d\b", s) and d < today:   # "21 June" with no year = the next one
        d = d.replace(year=d.year + 1)
    if not re.search(r"\d", s):          # "Friday" alone isn't a usable date
        return None
    return d


def extract_event(text: str, today: date | None = None) -> dict:
    """-> {"artist": str|None, "venue": str|None, "date": date|None, "how": str}"""
    today = today or date.today()
    text = str(text or "")
    out = {"artist": None, "venue": None, "date": None}
    how = []

    m = VS.search(text)
    if m:   # football / sport: "Arsenal v Chelsea"
        out["artist"] = f"{m.group(1).strip()} v {m.group(2).strip()}"
        how.append("match pattern 'X v Y'")

    nlp = _nlp()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not out["artist"] and not re.search(VENUE_WORDS, ent.text, re.I):
                out["artist"] = ent.text
            elif ent.label_ in ("FAC",) and not out["venue"]:
                out["venue"] = ent.text
            elif ent.label_ == "ORG" and re.search(VENUE_WORDS, ent.text, re.I) and not out["venue"]:
                out["venue"] = ent.text
            elif ent.label_ == "PERSON" and re.search(VENUE_WORDS, ent.text, re.I) and not out["venue"]:
                out["venue"] = ent.text
            elif ent.label_ == "DATE" and not out["date"]:
                out["date"] = _parse_date(ent.text, today)
        if not out["artist"]:   # bands are often tagged as organisations
            orgs = [e.text for e in doc.ents if e.label_ in ("ORG", "WORK_OF_ART")
                    and not re.search(VENUE_WORDS, e.text, re.I)]
            if orgs:
                out["artist"] = orgs[0]
        how.append("spaCy named-entity recognition")

    if not out["artist"]:   # "... Taylor Swift tickets ..."
        for m in BEFORE_TICKETS.finditer(text):
            words = [w for w in m.group(1).split() if not NOT_ARTIST.match(w)]
            cand = " ".join(words)
            if cand and not re.search(VENUE_WORDS, cand, re.I):
                out["artist"] = cand
                how.append("'<name> tickets' pattern")
                break
    if not out["venue"]:
        m = AT.search(text)
        if m and re.search(VENUE_WORDS, m.group(1), re.I):
            out["venue"] = m.group(1)
            how.append("'@ venue' pattern")
    if not out["date"]:
        m = re.search(r"\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                      r"(\s+20\d\d)?|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", text, re.I)
        if m:
            out["date"] = _parse_date(m.group(0), today)
    if out["venue"]:
        out["venue"] = re.sub(r"^(the\s+)", "The ", out["venue"].strip(" ,."), flags=re.I)
    out["how"] = " + ".join(how)
    return out


if __name__ == "__main__":
    for t in ["Hello, I'm selling 3x Olivia Rodrigo Tickets The Unraveled Tour @ The O2 Arena, London on "
              "Friday 9th April 2027. Kindly Dm if interested.",
              "2x Burna Boy tickets O2 Arena, £40 each, bank transfer only",
              "Selling 2 Taylor Swift tickets for Wembley Stadium 21st June, standing, dm me",
              "Arsenal v Chelsea tickets 3rd May Emirates, pm me",
              "spare ticket for tonight, dm"]:
        print(extract_event(t, today=date(2026, 11, 12)), "<-", t[:50])
