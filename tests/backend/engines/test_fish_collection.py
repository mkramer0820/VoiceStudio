"""Public Fish collection import: parsing and URL safety. No network."""
from __future__ import annotations

import httpx
import pytest

from engines.fish_api import collection

A, B = "59aa49a62c9f4608926420443e673e5c", "0c67dc47f4704447a39d436ae677b606"
PAGE = f"""
<html><body><h1 class="x">shadow sl</h1>
<a class="a" href="/m/{A}/" data-x="1"><span>Ni</span></a>
<a class="b" href="/m/{A}/">Nihilister (Goddess &amp; Victory)</a>
<a href="/m/{B}/"><span>Mo</span></a><a href="/m/{B}/">Moon Goddess</a>
<a href="/m/{B}/">Moon Goddess</a>
</body></html>
"""


def test_parses_title_voices_in_order_without_duplicates():
    out = collection.parse_collection(PAGE)
    assert out["title"] == "shadow sl"
    assert out["voices"] == [{"id": A, "title": "Nihilister (Goddess & Victory)"},
                             {"id": B, "title": "Moon Goddess"}]


@pytest.mark.parametrize("value", [
    "https://fish.audio/collections/63bdf88542744737926c783b476c91b7",
    "https://www.fish.audio/collections/63bdf88542744737926c783b476c91b7/",
    "https://fish.audio/en/collections/63bdf88542744737926c783b476c91b7",
    "63bdf88542744737926c783b476c91b7",
])
def test_accepts_collection_links_and_bare_ids(value):
    assert collection.collection_id(value) == "63bdf88542744737926c783b476c91b7"


@pytest.mark.parametrize("value", [
    "http://fish.audio/collections/63bdf88542744737926c783b476c91b7",
    "https://evil.example/collections/63bdf88542744737926c783b476c91b7",
    "https://fish.audio.evil.example/collections/63bdf88542744737926c783b476c91b7",
    "https://fish.audio/m/63bdf88542744737926c783b476c91b7",
    "https://user@fish.audio@evil.example/collections/63bdf88542744737926c783b476c91b7",
    "", "not a link",
])
def test_rejects_other_hosts_schemes_and_pages(value):
    with pytest.raises(collection.CollectionError):
        collection.collection_id(value)


def test_fetch_only_requests_the_canonical_page_and_sends_no_credentials():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, text=PAGE)

    out = collection.fetch_collection("https://www.fish.audio/collections/63bdf88542744737926c783b476c91b7",
                                      transport=httpx.MockTransport(handler))
    assert out["id"] == "63bdf88542744737926c783b476c91b7" and len(out["voices"]) == 2
    (req,) = seen
    assert str(req.url) == "https://fish.audio/collections/63bdf88542744737926c783b476c91b7/"
    assert "authorization" not in req.headers and "cookie" not in req.headers


@pytest.mark.parametrize("status,text", [(404, "no"), (500, "no"), (200, "<html>no voices</html>")])
def test_missing_private_or_empty_collections_give_a_clear_error(status, text):
    with pytest.raises(collection.CollectionError):
        collection.fetch_collection("63bdf88542744737926c783b476c91b7",
                                    transport=httpx.MockTransport(lambda r: httpx.Response(status, text=text)))


def test_redirects_are_not_followed():
    handler = lambda r: httpx.Response(302, headers={"location": "https://evil.example/"})  # noqa: E731
    with pytest.raises(collection.CollectionError):
        collection.fetch_collection("63bdf88542744737926c783b476c91b7", transport=httpx.MockTransport(handler))
