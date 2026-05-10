"""Unit tests for PlaywrightWebTools.

All Playwright internals are mocked — no real browser is launched.
The mock chain mirrors the real call sequence inside __enter__:

    sync_playwright() → manager
    manager.start()   → playwright
    playwright.chromium.launch() → browser
    browser.new_context()        → context
    context.new_page()           → page
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from ego_crawler.infrastructure.playwright.playwright_web_tools import PlaywrightWebTools
from ego_crawler.application.ports.web_tools import ToolResult

_MODULE = "ego_crawler.infrastructure.playwright.playwright_web_tools"


# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mocks() -> dict:
    """Builds the full Playwright mock hierarchy."""
    mock_playwright = MagicMock(name="playwright")
    mock_browser    = MagicMock(name="browser")
    mock_context    = MagicMock(name="context")
    mock_page       = MagicMock(name="page")

    mock_playwright.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value        = mock_context
    mock_context.new_page.return_value           = mock_page

    return {
        "playwright": mock_playwright,
        "browser":    mock_browser,
        "context":    mock_context,
        "page":       mock_page,
    }


@pytest.fixture
def tools(mocks):
    """Returns (PlaywrightWebTools, page_mock) with Playwright fully patched."""
    with patch(f"{_MODULE}.sync_playwright") as mock_sp:
        mock_sp.return_value.start.return_value = mocks["playwright"]
        with PlaywrightWebTools(headless=True, timeout=5_000) as wt:
            yield wt, mocks["page"]


# ── Lifecycle ──────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_enter_launches_browser_and_creates_page(self, mocks):
        with patch(f"{_MODULE}.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mocks["playwright"]
            with PlaywrightWebTools(headless=True) as wt:
                mocks["playwright"].chromium.launch.assert_called_once_with(headless=True)
                mocks["browser"].new_context.assert_called_once()
                mocks["context"].new_page.assert_called_once()
                assert wt._page is mocks["page"]

    def test_exit_closes_page_context_browser_in_order(self, mocks):
        call_order: list[str] = []
        mocks["page"].close.side_effect    = lambda: call_order.append("page")
        mocks["context"].close.side_effect = lambda: call_order.append("context")
        mocks["browser"].close.side_effect = lambda: call_order.append("browser")

        with patch(f"{_MODULE}.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mocks["playwright"]
            with PlaywrightWebTools():
                pass  # __exit__ is called here

        assert call_order == ["page", "context", "browser"]

    def test_exit_stops_playwright(self, mocks):
        with patch(f"{_MODULE}.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mocks["playwright"]
            with PlaywrightWebTools():
                pass
        mocks["playwright"].stop.assert_called_once()

    def test_exit_clears_internal_references(self, mocks):
        with patch(f"{_MODULE}.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mocks["playwright"]
            with PlaywrightWebTools() as wt:
                pass
        assert wt._page is None
        assert wt._browser is None
        assert wt._playwright is None

    def test_exit_swallows_close_exceptions(self, mocks):
        mocks["page"].close.side_effect = RuntimeError("browser crashed")
        with patch(f"{_MODULE}.sync_playwright") as mock_sp:
            mock_sp.return_value.start.return_value = mocks["playwright"]
            with PlaywrightWebTools():
                pass  # must not raise

    def test_methods_raise_when_used_outside_context_manager(self):
        wt = PlaywrightWebTools()
        with pytest.raises(RuntimeError, match="context manager"):
            wt.navigate("https://example.com")


# ── navigate ───────────────────────────────────────────────────────────────────

class TestNavigate:
    def test_success_returns_url_title_status(self, tools):
        wt, page = tools
        mock_response = MagicMock(status=200)
        page.goto.return_value = mock_response
        page.url = "https://example.com"
        page.title.return_value = "Example Domain"

        result = wt.navigate("https://example.com")

        assert result.success is True
        assert result.data["url"] == "https://example.com"
        assert result.data["title"] == "Example Domain"
        assert result.data["status_code"] == 200
        assert result.metadata["requested_url"] == "https://example.com"

    def test_returns_final_url_after_redirect(self, tools):
        wt, page = tools
        page.goto.return_value = MagicMock(status=301)
        page.url = "https://www.example.com"  # redirected
        page.title.return_value = "Redirected"

        result = wt.navigate("https://example.com")

        assert result.data["url"] == "https://www.example.com"

    def test_timeout_returns_error_result(self, tools):
        wt, page = tools
        page.goto.side_effect = PlaywrightTimeoutError("30000ms exceeded")

        result = wt.navigate("https://slow.example.com")

        assert result.success is False
        assert "Timeout" in result.error_message
        assert result.data is None

    def test_generic_exception_returns_error_result(self, tools):
        wt, page = tools
        page.goto.side_effect = Exception("connection refused")

        result = wt.navigate("https://unreachable.example.com")

        assert result.success is False
        assert "connection refused" in result.error_message

    def test_none_response_status_is_handled(self, tools):
        wt, page = tools
        page.goto.return_value = None
        page.url = "https://example.com"
        page.title.return_value = "Example"

        result = wt.navigate("https://example.com")

        assert result.success is True
        assert result.data["status_code"] is None


# ── search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_success_returns_parsed_results(self, tools):
        wt, page = tools
        page.goto.return_value = MagicMock(status=200)
        page.evaluate.return_value = [
            {"title": "Python", "url": "https://python.org",      "snippet": "Official"},
            {"title": "Docs",   "url": "https://docs.python.org", "snippet": "Docs"},
        ]

        result = wt.search("python")

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["title"] == "Python"
        assert result.metadata["query"] == "python"
        assert result.metadata["result_count"] == 2

    def test_parser_filters_invalid_results(self, tools):
        wt, page = tools
        page.goto.return_value = MagicMock()
        page.evaluate.return_value = [
            {"title": "",       "url": "https://bad.com",  "snippet": ""},  # no title
            {"title": "Good",   "url": "https://good.com", "snippet": "ok"},
        ]

        result = wt.search("test")

        assert result.success is True
        assert len(result.data) == 1

    def test_empty_results_returns_empty_list(self, tools):
        wt, page = tools
        page.goto.return_value = MagicMock()
        page.evaluate.return_value = []

        result = wt.search("xyzzy_nonexistent_term")

        assert result.success is True
        assert result.data == []
        assert result.metadata["result_count"] == 0

    def test_timeout_returns_error_result(self, tools):
        wt, page = tools
        page.goto.side_effect = PlaywrightTimeoutError("search timeout")

        result = wt.search("python")

        assert result.success is False
        assert "Timeout" in result.error_message

    def test_navigate_uses_google_url_with_query(self, tools):
        wt, page = tools
        page.goto.return_value = MagicMock()
        page.evaluate.return_value = []

        wt.search("hello world")

        called_url = page.goto.call_args[0][0]
        assert "google.com" in called_url
        assert "hello+world" in called_url or "hello%20world" in called_url or "hello world" in called_url


# ── extract_text ───────────────────────────────────────────────────────────────

class TestExtractText:
    def test_no_selector_reads_full_body(self, tools):
        wt, page = tools
        body_locator = MagicMock()
        body_locator.inner_text.return_value = "Full page text"
        page.locator.return_value = body_locator
        page.url = "https://example.com"

        result = wt.extract_text()

        page.locator.assert_called_with("body")
        assert result.success is True
        assert result.data == "Full page text"
        assert result.metadata["length"] == 14
        assert result.metadata["selector"] is None

    def test_with_selector_reads_matching_element(self, tools):
        wt, page = tools
        locator = MagicMock()
        locator.first.inner_text.return_value = "Heading text"
        page.locator.return_value = locator
        page.url = "https://example.com"

        result = wt.extract_text("h1")

        page.locator.assert_called_with("h1")
        assert result.success is True
        assert result.data == "Heading text"
        assert result.metadata["selector"] == "h1"

    def test_url_included_in_metadata(self, tools):
        wt, page = tools
        page.locator.return_value.inner_text.return_value = "text"
        page.url = "https://current-page.com"

        result = wt.extract_text()

        assert result.metadata["url"] == "https://current-page.com"

    def test_timeout_returns_error_result(self, tools):
        wt, page = tools
        page.locator.return_value.inner_text.side_effect = PlaywrightTimeoutError("timeout")

        result = wt.extract_text()

        assert result.success is False
        assert "Timeout" in result.error_message

    def test_generic_exception_returns_error_result(self, tools):
        wt, page = tools
        page.locator.return_value.first.inner_text.side_effect = Exception("selector error")

        result = wt.extract_text("nonexistent")

        assert result.success is False


# ── extract_images ─────────────────────────────────────────────────────────────

class TestExtractImages:
    def test_returns_image_list(self, tools):
        wt, page = tools
        page.eval_on_selector_all.return_value = [
            {"src": "https://example.com/a.jpg", "alt": "A", "width": 100, "height": 100},
            {"src": "https://example.com/b.png", "alt": "B", "width": 200, "height": 200},
        ]
        page.url = "https://example.com"

        result = wt.extract_images()

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["src"] == "https://example.com/a.jpg"
        assert result.metadata["count"] == 2
        assert result.metadata["selector"] is None

    def test_with_selector_scopes_query(self, tools):
        wt, page = tools
        page.eval_on_selector_all.return_value = []
        page.url = "https://example.com"

        wt.extract_images(".gallery")

        called_selector = page.eval_on_selector_all.call_args[0][0]
        assert called_selector == ".gallery img"

    def test_no_selector_queries_all_images(self, tools):
        wt, page = tools
        page.eval_on_selector_all.return_value = []
        page.url = "https://example.com"

        wt.extract_images()

        called_selector = page.eval_on_selector_all.call_args[0][0]
        assert called_selector == "img"

    def test_empty_page_returns_empty_list(self, tools):
        wt, page = tools
        page.eval_on_selector_all.return_value = []
        page.url = "https://example.com"

        result = wt.extract_images()

        assert result.success is True
        assert result.data == []
        assert result.metadata["count"] == 0

    def test_exception_returns_error_result(self, tools):
        wt, page = tools
        page.eval_on_selector_all.side_effect = Exception("js error")

        result = wt.extract_images()

        assert result.success is False
        assert "js error" in result.error_message


# ── scroll ─────────────────────────────────────────────────────────────────────

class TestScroll:
    def _setup_evaluate(self, page, scroll_position: int = 500):
        """Configures page.evaluate to return the scroll position on second call."""
        call_count = {"n": 0}

        def _side_effect(js: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None  # scrollBy — no return value
            return scroll_position  # scrollY

        page.evaluate.side_effect = _side_effect

    def test_scroll_down_passes_positive_delta(self, tools):
        wt, page = tools
        self._setup_evaluate(page, 500)

        result = wt.scroll("down")

        page.evaluate.assert_any_call("window.scrollBy(0, 500)")
        assert result.success is True
        assert result.data["direction"] == "down"
        assert result.data["px"] == 500

    def test_scroll_up_passes_negative_delta(self, tools):
        wt, page = tools
        self._setup_evaluate(page, 0)

        result = wt.scroll("up")

        page.evaluate.assert_any_call("window.scrollBy(0, -500)")
        assert result.success is True
        assert result.data["direction"] == "up"

    def test_position_y_included_in_data(self, tools):
        wt, page = tools
        self._setup_evaluate(page, 750)

        result = wt.scroll("down")

        assert result.data["position_y"] == 750

    def test_exception_returns_error_result(self, tools):
        wt, page = tools
        page.evaluate.side_effect = Exception("scroll failed")

        result = wt.scroll("down")

        assert result.success is False
        assert "scroll failed" in result.error_message


# ── go_back ────────────────────────────────────────────────────────────────────

class TestGoBack:
    def test_success_returns_previous_url_and_title(self, tools):
        wt, page = tools
        page.go_back.return_value = MagicMock(status=200)
        page.url = "https://previous.com"
        page.title.return_value = "Previous Page"

        result = wt.go_back()

        assert result.success is True
        assert result.data["returned_to"] == "https://previous.com"
        assert result.data["title"] == "Previous Page"

    def test_no_history_returns_error_result(self, tools):
        wt, page = tools
        page.go_back.return_value = None  # Playwright returns None when there's no history

        result = wt.go_back()

        assert result.success is False
        assert "No previous page" in result.error_message

    def test_timeout_returns_error_result(self, tools):
        wt, page = tools
        page.go_back.side_effect = PlaywrightTimeoutError("go_back timeout")

        result = wt.go_back()

        assert result.success is False
        assert "Timeout" in result.error_message

    def test_generic_exception_returns_error_result(self, tools):
        wt, page = tools
        page.go_back.side_effect = Exception("navigation failed")

        result = wt.go_back()

        assert result.success is False
        assert "navigation failed" in result.error_message

    def test_wait_for_networkidle_called_after_go_back(self, tools):
        wt, page = tools
        page.go_back.return_value = MagicMock()
        page.url = "https://prev.com"
        page.title.return_value = "Prev"

        wt.go_back()

        page.wait_for_load_state.assert_called_with("networkidle", timeout=5_000)


# ── ToolResult contract ────────────────────────────────────────────────────────

class TestToolResultContract:
    """Every method must return a ToolResult — never raise — when page is ready."""

    @pytest.fixture(autouse=True)
    def crash_page(self, tools):
        """Page raises unexpectedly on every call."""
        _, page = tools
        page.goto.side_effect                = Exception("unexpected crash")
        page.evaluate.side_effect            = Exception("unexpected crash")
        page.locator.side_effect             = Exception("unexpected crash")
        page.eval_on_selector_all.side_effect = Exception("unexpected crash")
        page.go_back.side_effect             = Exception("unexpected crash")
        self._wt = tools[0]

    def test_navigate_never_raises(self):
        result = self._wt.navigate("https://example.com")
        assert isinstance(result, ToolResult)
        assert result.success is False

    def test_search_never_raises(self):
        result = self._wt.search("test")
        assert isinstance(result, ToolResult)
        assert result.success is False

    def test_extract_text_never_raises(self):
        result = self._wt.extract_text()
        assert isinstance(result, ToolResult)
        assert result.success is False

    def test_extract_images_never_raises(self):
        result = self._wt.extract_images()
        assert isinstance(result, ToolResult)
        assert result.success is False

    def test_scroll_never_raises(self):
        result = self._wt.scroll()
        assert isinstance(result, ToolResult)
        assert result.success is False

    def test_go_back_never_raises(self):
        result = self._wt.go_back()
        assert isinstance(result, ToolResult)
        assert result.success is False
