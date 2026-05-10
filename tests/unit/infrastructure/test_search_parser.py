import pytest

from ego_crawler.infrastructure.playwright.search_parser import (
    GoogleSearchParser,
    SearchResult,
)


@pytest.fixture
def parser() -> GoogleSearchParser:
    return GoogleSearchParser()


class TestGoogleSearchParserParse:
    def test_valid_results_become_search_result_objects(self, parser):
        raw = [
            {"title": "Python", "url": "https://python.org", "snippet": "Official site"},
            {"title": "Docs",   "url": "https://docs.python.org", "snippet": "Docs"},
        ]
        results = parser.parse(raw)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_result_fields_are_mapped_correctly(self, parser):
        raw = [{"title": "Example", "url": "https://example.com", "snippet": "A domain"}]
        result = parser.parse(raw)[0]

        assert result.title == "Example"
        assert result.url == "https://example.com"
        assert result.snippet == "A domain"

    def test_filters_out_entries_without_url(self, parser):
        raw = [
            {"title": "No URL",      "url": "",                    "snippet": ""},
            {"title": "Has URL",     "url": "https://example.com", "snippet": ""},
        ]
        results = parser.parse(raw)

        assert len(results) == 1
        assert results[0].title == "Has URL"

    def test_filters_out_entries_without_title(self, parser):
        raw = [
            {"title": "",        "url": "https://example.com", "snippet": ""},
            {"title": "Present", "url": "https://other.com",   "snippet": ""},
        ]
        results = parser.parse(raw)

        assert len(results) == 1
        assert results[0].title == "Present"

    def test_filters_out_non_http_urls(self, parser):
        raw = [
            {"title": "JS redirect", "url": "javascript:void(0)",   "snippet": ""},
            {"title": "Relative",    "url": "/path/to/page",        "snippet": ""},
            {"title": "Valid",       "url": "https://example.com",  "snippet": ""},
        ]
        results = parser.parse(raw)

        assert len(results) == 1
        assert results[0].title == "Valid"

    def test_empty_input_returns_empty_list(self, parser):
        assert parser.parse([]) == []

    def test_strips_whitespace_from_fields(self, parser):
        raw = [{"title": "  Hello  ", "url": "https://example.com  ", "snippet": "  world  "}]
        result = parser.parse(raw)[0]

        assert result.title == "Hello"
        assert result.url == "https://example.com"
        assert result.snippet == "world"

    def test_missing_keys_are_treated_as_empty(self, parser):
        raw = [{"url": "https://example.com"}]  # no title → filtered out
        assert parser.parse(raw) == []

    def test_none_values_are_treated_as_empty(self, parser):
        raw = [{"title": None, "url": "https://example.com", "snippet": None}]
        assert parser.parse(raw) == []

    def test_snippet_is_optional_and_defaults_to_empty_string(self, parser):
        raw = [{"title": "No snippet", "url": "https://example.com"}]
        result = parser.parse(raw)[0]

        assert result.snippet == ""

    def test_search_result_is_frozen(self, parser):
        raw = [{"title": "Test", "url": "https://example.com", "snippet": ""}]
        result = parser.parse(raw)[0]

        with pytest.raises((AttributeError, TypeError)):
            result.title = "mutated"  # type: ignore[misc]

    def test_http_url_also_accepted(self, parser):
        raw = [{"title": "Plain HTTP", "url": "http://example.com", "snippet": ""}]
        results = parser.parse(raw)

        assert len(results) == 1
