from __future__ import annotations

from dataclasses import dataclass


# JavaScript injected into the search-result page via page.evaluate().
# Kept as a module-level constant so unit tests can verify the JS independently
# of any live browser.
GOOGLE_EXTRACT_JS = """
() => Array.from(
    document.querySelectorAll('div.g, div[data-sokoban-container]')
).map(el => ({
    title:   (el.querySelector('h3')    || {}).innerText || '',
    url:     (el.querySelector('a')     || {}).href     || '',
    snippet: (el.querySelector('.VwiC3b, [data-sncf]') || {}).innerText || ''
}))
"""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class GoogleSearchParser:
    """Transforms raw JS-extracted dicts into validated SearchResult objects.

    Intentionally pure (no Playwright dependency) so it can be unit-tested
    with hand-crafted data without launching a browser.
    """

    def parse(self, raw: list[dict]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in raw:
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if url.startswith("http") and title:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=(item.get("snippet") or "").strip(),
                    )
                )
        return results
