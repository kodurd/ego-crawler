"""Integration tests for PlaywrightWebTools.

These tests launch a REAL Chromium browser and make live network requests.
They are skipped by default; run with:

    pytest --integration

Requirements:
    pip install playwright
    playwright install chromium
"""
import pytest

from ego_crawler.infrastructure.playwright.playwright_web_tools import PlaywrightWebTools
from ego_crawler.application.ports.web_tools import ToolResult

# Stable IANA test domain — guaranteed to exist and change very rarely.
_EXAMPLE_URL = "https://example.com"
_EXAMPLE_TITLE = "Example Domain"


# ── module-scoped browser: one launch for all tests ──────────────────────────

@pytest.fixture(scope="module")
def browser():
    """Single headless browser shared across the module to keep tests fast."""
    with PlaywrightWebTools(headless=True, timeout=30_000) as tools:
        yield tools


# ── navigate ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNavigateIntegration:
    def test_navigate_example_com_succeeds(self, browser):
        result = browser.navigate(_EXAMPLE_URL)

        assert result.success is True
        assert isinstance(result.data, dict)

    def test_navigate_returns_final_url(self, browser):
        result = browser.navigate(_EXAMPLE_URL)

        assert "example.com" in result.data["url"]

    def test_navigate_returns_page_title(self, browser):
        result = browser.navigate(_EXAMPLE_URL)

        assert _EXAMPLE_TITLE in result.data["title"]

    def test_navigate_returns_http_status(self, browser):
        result = browser.navigate(_EXAMPLE_URL)

        assert result.data["status_code"] == 200

    def test_navigate_invalid_url_returns_error(self, browser):
        result = browser.navigate("https://this-domain-definitely-does-not-exist-xyz123.com")

        assert result.success is False
        assert result.error_message is not None


# ── extract_text ──────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestExtractTextIntegration:
    @pytest.fixture(autouse=True)
    def go_to_example(self, browser):
        browser.navigate(_EXAMPLE_URL)

    def test_full_body_text_contains_domain_name(self, browser):
        result = browser.extract_text()

        assert result.success is True
        assert "Example Domain" in result.data

    def test_with_h1_selector_returns_heading(self, browser):
        result = browser.extract_text("h1")

        assert result.success is True
        assert len(result.data) > 0

    def test_metadata_contains_length_and_url(self, browser):
        result = browser.extract_text()

        assert result.metadata["length"] == len(result.data)
        assert "example.com" in result.metadata["url"]

    def test_nonexistent_selector_returns_error(self, browser):
        result = browser.extract_text("#element-that-does-not-exist-ever")

        assert result.success is False


# ── extract_images ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestExtractImagesIntegration:
    @pytest.fixture(autouse=True)
    def go_to_example(self, browser):
        browser.navigate(_EXAMPLE_URL)

    def test_returns_list(self, browser):
        result = browser.extract_images()

        assert result.success is True
        assert isinstance(result.data, list)

    def test_metadata_count_matches_data_length(self, browser):
        result = browser.extract_images()

        assert result.metadata["count"] == len(result.data)

    def test_each_image_has_src_and_alt_keys(self, browser):
        result = browser.extract_images()

        for img in result.data:
            assert "src" in img
            assert "alt" in img


# ── scroll ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestScrollIntegration:
    @pytest.fixture(autouse=True)
    def go_to_example(self, browser):
        browser.navigate(_EXAMPLE_URL)

    def test_scroll_down_succeeds(self, browser):
        result = browser.scroll("down")

        assert result.success is True
        assert result.data["direction"] == "down"
        assert result.data["px"] == 500
        assert "position_y" in result.data

    def test_scroll_up_succeeds(self, browser):
        browser.scroll("down")  # ensure we're not at top
        result = browser.scroll("up")

        assert result.success is True
        assert result.data["direction"] == "up"


# ── go_back ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestGoBackIntegration:
    def test_go_back_returns_previous_page(self, browser):
        # Start from a known page
        browser.navigate(_EXAMPLE_URL)
        # Navigate away
        browser.navigate("https://www.iana.org/domains/reserved")

        result = browser.go_back()

        assert result.success is True
        assert "example.com" in result.data["returned_to"]

    def test_go_back_no_history_returns_error(self, browser):
        # Open a new fresh context to ensure empty history.
        # We test this by checking that go_back from the first page fails.
        with PlaywrightWebTools(headless=True) as fresh:
            # Navigate to one page (no history before it)
            fresh.navigate(_EXAMPLE_URL)
            result = fresh.go_back()

        assert result.success is False
        assert result.error_message is not None


# ── search ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSearchIntegration:
    def test_search_returns_tool_result(self, browser):
        """Smoke test: search must return a ToolResult regardless of bot-detection."""
        result = browser.search("Python programming language")

        assert isinstance(result, ToolResult)

    @pytest.mark.xfail(
        strict=False,
        reason="Google may return CAPTCHA or block headless browsers",
    )
    def test_search_returns_non_empty_results(self, browser):
        result = browser.search("Python programming language site:python.org")

        assert result.success is True
        assert len(result.data) > 0
        assert all("title" in r and "url" in r for r in result.data)


# ── end-to-end browse flow ────────────────────────────────────────────────────

@pytest.mark.integration
class TestBrowseFlowIntegration:
    def test_full_browse_session(self):
        """Navigate → extract → scroll → go back — all in one session."""
        with PlaywrightWebTools(headless=True) as tools:
            # 1. Navigate to starting page
            nav = tools.navigate(_EXAMPLE_URL)
            assert nav.success is True

            # 2. Extract page text
            text = tools.extract_text()
            assert text.success is True
            assert len(text.data) > 0

            # 3. Extract images (may be empty on example.com, that's fine)
            images = tools.extract_images()
            assert images.success is True

            # 4. Navigate to a second page
            tools.navigate("https://www.iana.org/domains/reserved")

            # 5. Scroll down
            scroll_down = tools.scroll("down")
            assert scroll_down.success is True

            # 6. Go back to example.com
            back = tools.go_back()
            assert back.success is True
            assert "example.com" in back.data["returned_to"]
