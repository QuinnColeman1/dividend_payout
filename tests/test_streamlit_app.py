"""End-to-end tests of the Streamlit app with yfinance mocked."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from .conftest import make_yfinance_like_data

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
YF_DATA = make_yfinance_like_data()
CREATED_TICKERS: list = []


def fake_ticker(symbol: str) -> mock.MagicMock:
    ticker = mock.MagicMock()
    CREATED_TICKERS.append(ticker)
    data = YF_DATA.get(symbol)
    if data is None:
        ticker.history.return_value = pd.DataFrame()
        ticker.info = {}
    else:
        ticker.history.return_value = pd.DataFrame({"Close": data["close"], "Dividends": data["dividends"]})
        ticker.info = data["info"]
    return ticker


@pytest.fixture
def app():
    CREATED_TICKERS.clear()
    with mock.patch("yfinance.Ticker", side_effect=fake_ticker):
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        at.sidebar.text_area[0].set_value("\n".join(YF_DATA)).run()
        yield at


def click(at: AppTest, label: str) -> None:
    next(b for b in at.button if label in b.label).click().run()


def test_initialize_loads_tickers(app):
    click(app, "Initialize Optimizer")
    assert not app.exception
    assert app.session_state.optimizer is not None
    assert len(app.session_state.tickers_data) == len(YF_DATA)


def test_requests_daily_unadjusted_prices(app):
    st.cache_data.clear()  # make sure data is fetched, not served from an earlier test's cache
    click(app, "Initialize Optimizer")

    history_calls = [t.history.call_args for t in CREATED_TICKERS if t.history.called]
    assert history_calls
    for call in history_calls:
        assert call.kwargs["interval"] == "1d"
        assert call.kwargs["auto_adjust"] is False


@pytest.mark.parametrize("optimization_type", ["Multi-Objective Balance", "Minimize Capital", "Smooth Cash Flow"])
def test_run_optimization_produces_results(app, optimization_type):
    click(app, "Initialize Optimizer")
    app.selectbox[0].set_value(optimization_type).run()
    click(app, "Run Optimization")

    assert not app.exception
    assert not [e.value for e in app.error], [e.value for e in app.error]
    assert app.session_state.results, "no result was recorded"

    app.sidebar.radio[0].set_value("Optimization Results").run()
    assert not app.exception
    result = list(app.session_state.results.values())[-1]
    assert result["portfolio"]
    assert result["actual_annual_dividend"] >= 10_000 * 0.999


def test_run_without_initializing_shows_error(app):
    click(app, "Run Optimization")
    assert any("initialize the optimizer" in e.value for e in app.error)


def test_efficient_frontier_page(app):
    click(app, "Initialize Optimizer")
    app.sidebar.radio[0].set_value("Efficient Frontier").run()
    click(app, "Generate Efficient Frontier")

    assert not app.exception
    statuses = [row["Status"] for row in app.session_state.frontier_data]
    assert statuses and all(s == "optimal" for s in statuses)


def test_changed_tickers_prompt_reinitialize(app):
    click(app, "Initialize Optimizer")
    app.sidebar.text_area[0].set_value("AAA1\nBBB1").run()
    assert any("Ticker list changed" in w.value for w in app.sidebar.warning)
