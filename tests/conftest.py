"""Shared fixtures. All data is synthetic, so the suite runs offline and deterministically."""

from typing import Dict, Tuple

import pandas as pd
import pytest

from src.dividendportfoliooptimizer import DividendPortfolioOptimizer

AS_OF = pd.Timestamp("2025-12-31")

# Quarterly payment schedules offset by one month, so a mix can produce even monthly income.
SCHEDULES = {
    "A": [1, 4, 7, 10],
    "B": [2, 5, 8, 11],
    "C": [3, 6, 9, 12],
}

# ticker -> (schedule, dividend per payment, share price)
# The highest yields are all on schedule A, so minimizing capital alone gives lumpy income.
DEFAULT_SPEC: Dict[str, Tuple[str, float, float]] = {
    "AAA1": ("A", 1.50, 50.0),  # 12.0%
    "AAA2": ("A", 1.40, 50.0),  # 11.2%
    "AAA3": ("A", 1.30, 50.0),  # 10.4%
    "BBB1": ("B", 0.80, 50.0),  # 6.4%
    "BBB2": ("B", 0.70, 50.0),  # 5.6%
    "BBB3": ("B", 0.60, 50.0),  # 4.8%
    "CCC1": ("C", 0.75, 50.0),  # 6.0%
    "CCC2": ("C", 0.65, 50.0),  # 5.2%
    "CCC3": ("C", 0.55, 50.0),  # 4.4%
}


def make_dividend_df(
    spec: Dict[str, Tuple[str, float, float]] = DEFAULT_SPEC,
    years: Tuple[int, ...] = (2023, 2024, 2025),
) -> pd.DataFrame:
    rows = []
    for ticker, (schedule, amount, price) in spec.items():
        for year in years:
            for month in SCHEDULES[schedule]:
                rows.append(
                    {
                        "ticker": ticker,
                        "ex_dividend_date": f"{year}-{month:02d}-15",
                        "dividend_amount": amount,
                        "close_price": price,
                        "company_name": f"{ticker} Corp",
                        "sector": f"Sector {schedule}",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def dividend_df() -> pd.DataFrame:
    return make_dividend_df()


@pytest.fixture
def optimizer(dividend_df: pd.DataFrame) -> DividendPortfolioOptimizer:
    return DividendPortfolioOptimizer(dividend_df, as_of=AS_OF)


def make_yfinance_like_data(spec: Dict[str, Tuple[str, float, float]] = DEFAULT_SPEC) -> Dict[str, dict]:
    """Data in the shape the Streamlit app builds from yfinance (tz-aware daily index)."""
    index = pd.date_range("2023-01-02", "2025-12-31", freq="B", tz="America/New_York")
    data = {}
    for ticker, (schedule, amount, price) in spec.items():
        close = pd.Series(price, index=index)
        close.iloc[-1] = price * 1.25  # latest price differs from ex-date prices
        dividends = pd.Series(0.0, index=index)
        for year in (2023, 2024, 2025):
            for month in SCHEDULES[schedule]:
                day = index[(index.year == year) & (index.month == month) & (index.day >= 15)][0]
                dividends[day] = amount
        data[ticker] = {
            "dividends": dividends,
            "close": close,
            "info": {"shortName": f"{ticker} Corp", "sector": f"Sector {schedule}", "currency": "USD"},
        }
    return data
