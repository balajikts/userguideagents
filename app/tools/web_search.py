"""Web search for official manuals / support pages (Tavily)."""

from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

# Manufacturer-owned domains. Subdomains match too (helpguide.sony.net → sony.net).
OFFICIAL_DOMAINS: dict[str, list[str]] = {
    "apple": ["apple.com"],
    "samsung": ["samsung.com"],
    "sony": ["sony.com", "sony.net", "sony.co.uk", "playstation.com"],
    "lg": ["lg.com"],
    "google": ["google.com", "googlestore.com"],
    "pixel": ["google.com"],
    "bose": ["bose.com"],
    "jbl": ["jbl.com", "harman.com"],
    "sennheiser": ["sennheiser.com", "sennheiser-hearing.com"],
    "microsoft": ["microsoft.com", "xbox.com"],
    "nintendo": ["nintendo.com"],
    "dell": ["dell.com"],
    "hp": ["hp.com"],
    "lenovo": ["lenovo.com"],
    "asus": ["asus.com"],
    "acer": ["acer.com"],
    "canon": ["canon.com", "usa.canon.com", "canon-europe.com"],
    "nikon": ["nikon.com", "nikonusa.com", "nikonimgsupport.com"],
    "fujifilm": ["fujifilm.com", "fujifilm-x.com"],
    "gopro": ["gopro.com"],
    "netgear": ["netgear.com"],
    "tp-link": ["tp-link.com"],
    "linksys": ["linksys.com"],
    "garmin": ["garmin.com"],
    "fitbit": ["fitbit.com", "google.com"],
    "epson": ["epson.com"],
    "brother": ["brother.com", "brother-usa.com"],
    "roku": ["roku.com"],
    "vizio": ["vizio.com"],
    "tcl": ["tcl.com"],
    "hisense": ["hisense-usa.com", "hisense.com"],
    "panasonic": ["panasonic.com", "panasonic.net"],
    "philips": ["philips.com"],
    "sonos": ["sonos.com"],
    "logitech": ["logitech.com", "logi.com"],
    "anker": ["anker.com", "soundcore.com", "eufy.com"],
    "dyson": ["dyson.com"],
    "whirlpool": ["whirlpool.com"],
    "bosch": ["bosch-home.com", "bosch.com"],
    "dji": ["dji.com"],
    "oneplus": ["oneplus.com"],
    "xiaomi": ["mi.com", "xiaomi.com"],
    "motorola": ["motorola.com"],
    "amazon": ["amazon.com"],
}
# Reputable third-party manual mirrors: useful, but ranked below the manufacturer.
MANUAL_MIRRORS = ["manualslib.com", "manua.ls", "manualsonline.com", "ifixit.com"]


def _norm_brand(brand: str | None) -> str | None:
    if not brand:
        return None
    b = brand.strip().lower().replace(" ", "-")
    return {"tplink": "tp-link", "google-pixel": "google"}.get(b, b)


def official_domains(brand: str | None) -> list[str]:
    b = _norm_brand(brand)
    return OFFICIAL_DOMAINS.get(b, []) if b else []


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def domain_matches(url: str, domains: list[str]) -> bool:
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)


class WebHit(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0


class WebSearchClient(Protocol):
    async def search(self, query: str, *, include_domains: list[str] | None = None, max_results: int = 6) -> list[WebHit]: ...


class TavilySearchClient:
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout: float = 15.0, http: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, *, include_domains: list[str] | None = None, max_results: int = 6) -> list[WebHit]:
        body: dict = {"query": query, "max_results": max_results, "search_depth": "advanced"}
        if include_domains:
            body["include_domains"] = include_domains
        r = await self._http.post(self.URL, json=body, headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        return [
            WebHit(title=h.get("title") or h["url"], url=h["url"], content=h.get("content") or "", score=float(h.get("score") or 0))
            for h in r.json().get("results", [])
            if h.get("url")
        ]

    async def aclose(self) -> None:
        await self._http.aclose()
