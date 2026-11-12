"""
redact.py - v2.6: remove personal details from listing text before it is stored (misses log)
or sent to the Gemini explainer. UK GDPR data minimisation; OWASP LLM Top 10 (sensitive
information disclosure). Simple patterns - a first version to test and improve for 002.
"""
import re

PATTERNS = [
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]"),
    (r"(?<!\w)@[\w.]{2,30}", "[handle]"),
    (r"\b\d{2}[- ]?\d{2}[- ]?\d{2}\b(?=.{0,40}\b\d{8}\b)|\b\d{8}\b", "[bank number]"),
    (r"\bsort code[:\s]*\d{2}[- ]?\d{2}[- ]?\d{2}\b", "sort code [bank number]"),
    (r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", "[IBAN]"),
    (r"(\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b", "[phone]"),
    (r"https?://\S+|www\.\S+", "[link]"),
]


def redact(text: str) -> str:
    t = str(text or "")
    for pattern, repl in PATTERNS:
        t = re.sub(pattern, repl, t, flags=re.I if "sort code" in pattern else 0)
    return t


if __name__ == "__main__":
    print(redact("Message me on 07712 345678 or jo.bloggs@gmail.com, @jo_tickets. "
                 "Sort code 12-34-56 account 12345678. https://bit.ly/x"))
