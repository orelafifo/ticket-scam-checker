"""
explainer.py - Layer 3: the LLM turns the score + red flags into plain English.

The LLM does NOT decide if it's a scam - the classifier (plus hard-stop rules)
does. The LLM only explains. If no API key is set, or the call fails, it falls
back to a template so the demo always works.

Pick ONE provider (Gemini is free via Google AI Studio):
    export GEMINI_API_KEY="your-key"        # pip install google-genai
    export ANTHROPIC_API_KEY="your-key"     # pip install anthropic
"""
import os
import time

SYSTEM_PROMPT = """You are a friendly fraud-prevention assistant for music fans buying resale tickets.
You are given a risk level from a trained model, the red flags it found, where the ticket is
being sold, any good signs, and the result of an official event check. Do NOT change the risk
level or invent new red flags. In under 90 words:
1) say the risk level in one sentence, 2) explain the red flags in plain English,
3) give one concrete safer next step that fits where they are buying (e.g. pay with PayPal Goods & Services or a credit card,
use a face-value resale site like Twickets or the official Ticketmaster exchange, never share codes).
Warm, clear, no jargon, UK English."""


# Advice that fits WHERE they're buying (category from channels.py)
CHANNEL_ADVICE = {
    "official": "You're buying through an official or face-value platform, which protects buyers, "
                "so keep the payment and ticket transfer inside that platform.",
    "secondary": "Resale sites like viagogo and StubHub have buyer guarantees, but tickets are often "
                 "far above face value and can be cancelled by the event, so compare with Twickets "
                 "or the official resale first.",
    "social": "Social media and messaging apps give you no buyer protection, so never pay by bank "
              "transfer or friends-and-family there.",
    "unknown_site": "Check the web address carefully and only buy from sites you can verify "
                    "independently, not from a link someone sent you.",
}


def fallback(level, flags, event_note="", channel_category=None):
    advice = CHANNEL_ADVICE.get(channel_category, "")
    if not flags:
        return (f"{level} risk. We didn't spot common scam signs in this listing. "
                + (advice + " " if advice else "") +
                "Pay with a protected method (PayPal Goods & Services or a credit card) "
                "and only accept tickets transferred through the official app.")
    bullets = "; ".join(f.lower() for f in flags)
    return (f"{level} risk. This listing shows signs that often appear in ticket scams: {bullets}. "
            "Safer option: don't pay by bank transfer or friends-and-family, never share a code "
            "someone sends you, and look for the same tickets on a face-value resale site like "
            "Twickets or the official exchange." + (" " + advice if advice else ""))


def _prompt(listing_text, score, level, flags, event_note, where, good_signs):
    return (f"Listing:\n{listing_text}\n\nWhere it's being sold: {where or 'not given'}\n"
            f"Model scam score: {score:.0%}\nRisk level: {level}\n"
            f"Red flags found: {flags or 'none'}\nGood signs: {good_signs or 'none'}\n"
            f"Event check: {event_note or 'not checked'}")


# Gemini model names change often (Google retires old ones), so try a few in order.
# Set GEMINI_MODEL in .env to force a specific one.
GEMINI_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-flash-latest",
                             "gemini-flash-lite-latest", "gemini-3-flash-preview",
                             "gemini-2.5-flash"] if m]
# Errors where trying again / another model can help:
#   404 NOT_FOUND = model retired, 503 UNAVAILABLE = model overloaded right now
RETRYABLE = ("404", "NOT_FOUND", "503", "UNAVAILABLE", "INCOMPLETE")


def _gemini(prompt):
    from google import genai
    client = genai.Client()
    last_error = None
    for attempt in range(2):                 # two passes through the list, short pause between
        for model in GEMINI_MODELS:
            try:
                # Newer Gemini models "think" before answering, and that thinking uses up
                # the output allowance - so give plenty of room (the prompt keeps the
                # answer itself under 90 words).
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                             max_output_tokens=2048))
                text = (resp.text or "").strip()
                finish = str(getattr((resp.candidates or [None])[0], "finish_reason", ""))
                if "MAX_TOKENS" in finish or len(text.split()) < 20:
                    raise RuntimeError("INCOMPLETE answer from Gemini")  # never show a cut-off sentence
                return text
            except Exception as e:
                last_error = e
                if any(code in str(e) for code in RETRYABLE):  # retired or busy -> try the next model
                    continue
                raise                                          # bad key, quota etc. -> stop and report
        time.sleep(2)
    raise last_error


def _short(e):
    """Readable reason for the fallback message (never includes the key)."""
    text = str(e)
    for hint, meaning in [("API_KEY_INVALID", "API key not valid - check GEMINI_API_KEY in .env"),
                          ("PERMISSION_DENIED", "key not allowed to use this model"),
                          ("RESOURCE_EXHAUSTED", "free-tier limit reached - wait a minute and retry"),
                          ("UNAVAILABLE", "Gemini is busy right now - try again in a minute"),
                          ("INCOMPLETE", "Gemini's answer was cut off, so the standard explanation is shown"),
                          ("NOT_FOUND", "model name not available - set GEMINI_MODEL in .env")]:
        if hint in text:
            return meaning
    return f"{e.__class__.__name__}: {text[:150]}"


def explain(listing_text, score, level, flags, event_note="", where="", good_signs=None,
            channel_category=None):
    prompt = _prompt(listing_text, score, level, flags, event_note, where, good_signs)
    try:
        if os.getenv("GEMINI_API_KEY"):
            return _gemini(prompt)
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"), max_tokens=300,
                system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
    except Exception as e:  # never let the demo crash on camera
        return (fallback(level, flags, channel_category=channel_category)
                + f"\n\n(AI explanation unavailable: {_short(e)})")
    return fallback(level, flags, channel_category=channel_category)
