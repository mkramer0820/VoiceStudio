"""Read a public Fish Audio voice collection from its web page.

Fish's API has no endpoint for collections, but a public collection page
(``https://fish.audio/collections/<id>``) is server-rendered with its voices.
This is an explicit, user-triggered request to fish.audio's website: no API key
or cookie is sent, only the page the user pasted is fetched, and the response is
size-capped and parsed for ``/m/<voice id>`` links. Private pages (bookmarks)
need a login and cannot be read.
"""
from __future__ import annotations

import html
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

_HOSTS = {"fish.audio", "www.fish.audio"}
_ID = r"[0-9a-f]{32}"
_PAGE_ID_RE = re.compile(rf"^{_ID}$")
_LINK_RE = re.compile(rf'<a[^>]*href="/m/({_ID})/?"[^>]*>(.*?)</a>', re.S | re.I)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_MAX_BYTES = 3_000_000
_TIMEOUT = httpx.Timeout(15.0, connect=8.0)


class CollectionError(ValueError):
    """The URL is not a readable public collection."""


def collection_id(value: str) -> str:
    """The 32-hex collection id from a fish.audio collection URL (or a bare id)."""
    text = (value or "").strip()
    if _PAGE_ID_RE.match(text):
        return text
    parsed = urlparse(text)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _HOSTS:
        raise CollectionError("Paste a link like https://fish.audio/collections/<id>.")
    match = re.match(rf"^/(?:[a-z]{{2}}(?:-[A-Za-z]{{2,4}})?/)?collections/({_ID})/?$", parsed.path)
    if not match:
        raise CollectionError("That is not a Fish Audio collection link.")
    return match.group(1)


def _text(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def parse_collection(page: str) -> dict:
    """``{"title", "voices": [{"id", "title"}]}`` from a collection page's HTML."""
    names: dict[str, str] = {}
    order: list[str] = []
    for voice_id, inner in _LINK_RE.findall(page):
        if voice_id not in names:
            order.append(voice_id)
            names[voice_id] = ""
        label = _text(inner)
        if len(label) > len(names[voice_id]):  # the avatar link is initials; the title link is longer
            names[voice_id] = label
    h1 = _H1_RE.search(page)
    return {"title": _text(h1.group(1)) if h1 else "",
            "voices": [{"id": i, "title": names[i] or i} for i in order]}


def fetch_collection(value: str, *, transport: Optional[httpx.BaseTransport] = None) -> dict:
    """Download and parse one public collection page. Raises ``CollectionError``."""
    cid = collection_id(value)
    url = f"https://fish.audio/collections/{cid}/"  # the site redirects the slashless form
    try:
        with httpx.Client(timeout=_TIMEOUT, transport=transport, follow_redirects=False,
                          headers={"User-Agent": "VoiceStudio collection import"}) as http:
            with http.stream("GET", url) as resp:
                if resp.status_code == 404:
                    raise CollectionError("Collection not found. Is it public?")
                if resp.status_code != 200:
                    raise CollectionError(f"Fish Audio returned HTTP {resp.status_code} for that collection.")
                body = bytearray()
                for part in resp.iter_bytes():
                    body.extend(part)
                    if len(body) > _MAX_BYTES:
                        raise CollectionError("That page is larger than expected.")
    except httpx.HTTPError as exc:
        raise CollectionError(f"Could not reach fish.audio ({type(exc).__name__}).") from exc
    parsed = parse_collection(body.decode("utf-8", errors="replace"))
    if not parsed["voices"]:
        raise CollectionError("No voices found. The collection may be private or empty.")
    return {"id": cid, **parsed}
