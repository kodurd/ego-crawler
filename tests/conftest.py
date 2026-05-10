import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / "ego_crawler"
sys.path.insert(0, str(src_path))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require a real browser and network",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="pass --integration to run browser tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
