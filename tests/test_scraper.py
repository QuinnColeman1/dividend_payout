"""Tests for the S&P 500 dividend scraper (yfinance is mocked)."""

from unittest import mock

import pandas as pd
import pytest

from src.dividend_payout import EnhancedDividendScraper


def fake_history(**kwargs):
    index = pd.DatetimeIndex(["2025-03-14", "2025-03-17", "2025-06-16", "2025-12-30"], tz="America/New_York")
    return pd.DataFrame(
        {
            "Close": [100.0, 101.0, 104.0, 110.0],
            "Dividends": [0.0, 1.0, 1.0, 0.0],
        },
        index=index,
    )


@pytest.fixture
def fake_ticker():
    ticker = mock.MagicMock()
    ticker.info = {"longName": "Example Co", "sector": "Utilities", "marketCap": 5e10, "averageVolume": 1e6}
    ticker.history.side_effect = fake_history
    with mock.patch("src.dividend_payout.yf.Ticker", return_value=ticker):
        yield ticker


def test_requests_unadjusted_prices(fake_ticker):
    EnhancedDividendScraper().get_dividend_events("EX")
    assert fake_ticker.history.call_args.kwargs["auto_adjust"] is False


def test_dividend_events_use_close_on_ex_date(fake_ticker):
    events = EnhancedDividendScraper().get_dividend_events("EX")

    assert [e["ex_dividend_date"] for e in events] == ["2025-03-17", "2025-06-16"]
    assert [e["close_price"] for e in events] == [101.0, 104.0]
    assert events[0]["yield_pct"] == pytest.approx(1.0 / 101.0)
    assert all(e["current_price"] == 110.0 for e in events)
    assert events[0]["company_name"] == "Example Co"


def test_scrape_output_feeds_the_optimizer(fake_ticker):
    from src.dividendportfoliooptimizer import DividendPortfolioOptimizer

    scraper = EnhancedDividendScraper(max_workers=2, batch_size=10)
    with mock.patch.object(scraper, "get_all_tickers", return_value={"sp500": ["EX"]}):
        df = scraper.scrape()

    opt = DividendPortfolioOptimizer(df)
    assert opt.ticker_data["EX"]["current_price"] == 110.0
    assert opt.ticker_data["EX"]["annual_yield"] == pytest.approx(2.0 / 110.0)
