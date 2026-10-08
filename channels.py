"""
channels.py - Layer 0: WHERE is the ticket being sold?

The same listing is far riskier on WhatsApp than on Twickets, so the checker
asks where the buyer found it (and optionally takes the link) BEFORE scoring.

Two inputs:
  1. platform - picked from a list (WhatsApp, X, viagogo, Ticketmaster resale...)
  2. link     - optional URL the buyer is about to use

What we do with a link (privacy and security by design):
  * We NEVER open or download the page. Visiting an unknown link could expose
    the user or our server to malware, and we don't need the page content.
  * We only look at the web address itself: is it a known site, a lookalike of
    a known site, a link shortener, or insecure (http)?
  * For unfamiliar websites we look up the domain's registration date using
    RDAP (the public registry service that replaced WHOIS). Brand-new domains
    are a classic fake-ticket-site sign. If the lookup fails we simply skip it.

Category meanings:
  official   - primary seller or face-value / capped resale with buyer protection
  secondary  - open resale marketplaces (viagogo, StubHub): buyer guarantees exist,
               but prices are often far above face value and promoters can cancel
               tickets resold against their terms
  social     - social media, messaging apps and classified ads: no buyer protection
  unknown_site - a website we don't recognise
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Platforms the user can pick (label -> category)
# ---------------------------------------------------------------------------
PLATFORMS = {
    "Ticketmaster / Ticketmaster Resale": "official",
    "AXS / AXS Official Resale": "official",
    "See Tickets": "official",
    "DICE": "official",
    "Twickets": "official",
    "Skiddle": "official",
    "Other official seller (Eventim, Gigantic...)": "official",
    "viagogo": "secondary",
    "StubHub": "secondary",
    "Other resale website": "secondary",
    "Instagram": "social",
    "Facebook (post, group or Marketplace)": "social",
    "X (Twitter)": "social",
    "TikTok": "social",
    "WhatsApp": "social",
    "Telegram": "social",
    "Snapchat": "social",
    "Text message / iMessage": "social",
    "Gumtree or other classified ads": "social",
    "A website I don't recognise": "unknown_site",
}
PLATFORM_CHOICES = ["Choose where you found the tickets..."] + list(PLATFORMS)

# ---------------------------------------------------------------------------
# Known web addresses (registrable domain -> (category, friendly name))
# ---------------------------------------------------------------------------
KNOWN_DOMAINS = {
    # official / face-value
    "ticketmaster.co.uk": ("official", "Ticketmaster"), "ticketmaster.com": ("official", "Ticketmaster"),
    "livenation.co.uk": ("official", "Live Nation"), "axs.com": ("official", "AXS"),
    "seetickets.com": ("official", "See Tickets"), "dice.fm": ("official", "DICE"),
    "twickets.live": ("official", "Twickets"), "twickets.co.uk": ("official", "Twickets"),
    "skiddle.com": ("official", "Skiddle"), "gigantic.com": ("official", "Gigantic"),
    "eventim.co.uk": ("official", "Eventim"), "ticketweb.uk": ("official", "TicketWeb"),
    "gigsandtours.com": ("official", "Gigs and Tours"),
    # secondary marketplaces
    "viagogo.co.uk": ("secondary", "viagogo"), "viagogo.com": ("secondary", "viagogo"),
    "stubhub.co.uk": ("secondary", "StubHub"), "stubhub.com": ("secondary", "StubHub"),
    "gigsberg.com": ("secondary", "Gigsberg"), "seatgeek.com": ("secondary", "SeatGeek"),
    "vividseats.com": ("secondary", "Vivid Seats"),
    # social media / messaging / classifieds
    "instagram.com": ("social", "Instagram"), "facebook.com": ("social", "Facebook"),
    "fb.com": ("social", "Facebook"), "fb.me": ("social", "Facebook"),
    "x.com": ("social", "X"), "twitter.com": ("social", "X"),
    "tiktok.com": ("social", "TikTok"), "reddit.com": ("social", "Reddit"),
    "discord.com": ("social", "Discord"), "discord.gg": ("social", "Discord"),
    "whatsapp.com": ("social", "WhatsApp"), "wa.me": ("social", "WhatsApp"),
    "t.me": ("social", "Telegram"), "telegram.me": ("social", "Telegram"),
    "snapchat.com": ("social", "Snapchat"), "gumtree.com": ("social", "Gumtree"),
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy"}
# brand words scammers copy into fake domains
BRANDS = ["ticketmaster", "livenation", "axs", "seetickets", "dice", "twickets", "skiddle",
          "viagogo", "stubhub", "eventim", "gigantic", "theo2", "wembley"]
SECOND_LEVEL = {"co.uk", "org.uk", "ac.uk", "com.au", "co.nz", "co.za", "com.br"}


# ---------------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------------
def registrable_domain(url: str) -> tuple[str, str]:
    """Returns (scheme, registrable domain) e.g. ('https', 'ticketmaster.co.uk')."""
    url = url.strip()
    if not re.match(r"^[a-z]+://", url, re.I):
        url = "https://" + url  # people often paste without the https://
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    parts = host.split(".")
    keep = 3 if len(parts) >= 3 and ".".join(parts[-2:]) in SECOND_LEVEL else 2
    scheme = parsed.scheme.lower() if "://" in url[:10] else "https"
    return scheme, ".".join(parts[-keep:])


def _label(domain: str) -> str:
    """'ticketmaster-uk.co.uk' -> 'ticketmaster-uk'"""
    return domain.split(".")[0]


def is_lookalike(domain: str) -> str | None:
    """Returns the brand being imitated, or None.
    Catches both 'ticketmaster-uk-sale.com' (brand word inside) and
    'ticketrnaster.co.uk' (spelling trick)."""
    if domain in KNOWN_DOMAINS:
        return None
    label = _label(domain).replace("-", "")
    for known, (_, name) in KNOWN_DOMAINS.items():
        known_label = _label(known)
        if len(known_label) < 5:  # short names like 'x', 'axs', 'dice' give false alarms
            continue
        if SequenceMatcher(None, label, known_label).ratio() >= 0.85:
            return name
    for brand in BRANDS:
        if len(brand) >= 5 and brand in label:
            return brand
    return None


def domain_age_days(domain: str, timeout: float = 5) -> int | None:
    """Days since the domain was registered, from RDAP. None if unknown."""
    try:
        import requests
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout,
                         headers={"Accept": "application/rdap+json"})
        if r.status_code != 200:
            return None
        for event in r.json().get("events", []):
            if event.get("eventAction") == "registration":
                registered = datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
                return (datetime.now(timezone.utc) - registered).days
    except Exception:
        return None
    return None


def analyse_link(url: str, check_age: bool = True) -> dict:
    """Look at a web address WITHOUT visiting it."""
    scheme, domain = registrable_domain(url)
    info = dict(domain=domain, category=None, site=None, lookalike_of=None,
                insecure=(scheme == "http"), shortener=domain in SHORTENERS,
                age_days=None, notes=[])
    if not domain or "." not in domain:
        info["notes"].append("That doesn't look like a valid web address.")
        return info
    if domain in KNOWN_DOMAINS:
        info["category"], info["site"] = KNOWN_DOMAINS[domain]
        info["notes"].append(f"Recognised site: {info['site']}.")
    elif info["shortener"]:
        info["category"] = "unknown_site"
        info["notes"].append("Shortened link: it hides the real website. Ask for the full link.")
    else:
        info["category"] = "unknown_site"
        info["lookalike_of"] = is_lookalike(domain)
        if info["lookalike_of"]:
            info["notes"].append(f"'{domain}' looks like an imitation of {info['lookalike_of']}, "
                                 "but it isn't their real website.")
        else:
            info["notes"].append(f"'{domain}' isn't a ticket site we recognise.")
        if check_age:
            info["age_days"] = domain_age_days(domain)
            if info["age_days"] is not None:
                info["notes"].append(f"Website registered {info['age_days']} days ago.")
    if info["insecure"]:
        info["notes"].append("The link isn't secure (http, not https).")
    return info


# ---------------------------------------------------------------------------
# Turn platform + link into the channel dict used by features.all_flags()
# ---------------------------------------------------------------------------
def channel_info(platform: str | None, link: str = "", check_age: bool = True) -> dict:
    category = PLATFORMS.get(platform) if platform else None
    link_info = analyse_link(link, check_age) if link and link.strip() else None
    # If the link tells us more than the dropdown, trust the link
    # (e.g. user picked 'Twickets' but pasted a lookalike address).
    if link_info and link_info["category"]:
        if category is None or link_info["category"] in ("unknown_site", "social"):
            category = link_info["category"]
    li = link_info or {}
    age = li.get("age_days")
    return dict(
        category=category,                         # official / secondary / social / unknown_site / None
        lookalike=bool(li.get("lookalike_of")),
        new_domain=age is not None and age < 180,
        insecure_or_short=bool(li.get("insecure") or li.get("shortener")),
        platform=platform, link=link_info,
    )


if __name__ == "__main__":
    for p, l in [("Twickets", ""), ("WhatsApp", ""), (None, "https://www.viagogo.co.uk/Concert-Tickets/x"),
                 ("Twickets", "twickets-uk.co.uk/tickets"), (None, "http://bit.ly/abc"),
                 (None, "https://ticketrnaster.co.uk/event/123"), (None, "x.com/someone/status/1")]:
        c = channel_info(p, l, check_age=False)
        print(f"{str(p):12s} {l:45s} -> {c['category']:12s} lookalike={c['lookalike']} "
              f"short/insecure={c['insecure_or_short']}  {c['link']['notes'] if c['link'] else ''}")
