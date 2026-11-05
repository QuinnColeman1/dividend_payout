"""Tests for OptimizerWrapper."""

import pandas as pd
import pytest

from src.optimizer_wrapper import OptimizerWrapper, create_optimizer_wrapper

from .conftest import AS_OF, make_yfinance_like_data

TARGET = 10_000


@pytest.fixture
def wrapper() -> OptimizerWrapper:
    return OptimizerWrapper(make_yfinance_like_data(), as_of=AS_OF)


def test_prepare_dividend_data_uses_latest_close_as_current_price(wrapper):
    data = wrapper.optimizer.ticker_data["AAA1"]
    assert data["current_price"] == pytest.approx(50.0 * 1.25)
    assert data["annual_yield"] == pytest.approx(6.0 / 62.5)
    assert data["company_name"] == "AAA1 Corp"
    assert data["sector"] == "Sector A"


def test_tickers_without_dividends_are_skipped():
    data = make_yfinance_like_data()
    data["NODIV"] = {
        "dividends": pd.Series(dtype=float),
        "close": pd.Series([10.0], index=pd.DatetimeIndex(["2025-01-02"])),
        "info": {},
    }
    wrapper = OptimizerWrapper(data, as_of=AS_OF)
    assert "NODIV" not in wrapper.optimizer.ticker_data


def test_no_dividend_data_raises():
    with pytest.raises(ValueError, match="No valid dividend data"):
        OptimizerWrapper({"X": {"dividends": pd.Series(dtype=float), "close": pd.Series(dtype=float)}})


def test_minimize_capital(wrapper):
    result = wrapper.minimize_capital(target_annual_dividend=TARGET, max_position_pct=0.2)
    assert result["actual_annual_dividend"] == pytest.approx(TARGET, rel=1e-3)
    assert len(result["portfolio"]) == 5


def test_smooth_cashflow_works(wrapper):
    # Regression: the wrapper used to pass the wrong parameter name, so this always failed.
    result = wrapper.smooth_cashflow(target_annual_dividend=TARGET, max_position_pct=0.15)
    assert result["portfolio"]
    assert result["actual_annual_dividend"] >= TARGET * 0.999


@pytest.mark.parametrize("method", ["minimize_capital", "smooth_cashflow", "multi_objective"])
def test_budget_constraint_is_passed_through(wrapper, method):
    with pytest.raises(RuntimeError, match="infeasible"):
        getattr(wrapper, method)(target_annual_dividend=TARGET, max_position_pct=0.2, budget_constraint=5_000)
    assert "infeasible" in wrapper.get_last_error()


@pytest.mark.parametrize("method", ["smooth_cashflow", "multi_objective"])
def test_min_yield_threshold_is_passed_through(wrapper, method):
    result = getattr(wrapper, method)(target_annual_dividend=TARGET, max_position_pct=0.25, min_yield_threshold=0.05)
    assert all(pos["annual_yield"] >= 0.05 for pos in result["portfolio"])


def test_multi_objective_requires_lambdas_to_sum_to_one(wrapper):
    with pytest.raises(ValueError, match="should equal 1.0"):
        wrapper.multi_objective(target_annual_dividend=TARGET, lambda_capital=0.5, lambda_smoothness=0.2)


def test_last_error_is_cleared_after_success(wrapper):
    with pytest.raises(ValueError):
        wrapper.minimize_capital(target_annual_dividend=TARGET, max_position_pct=0.01)
    assert wrapper.get_last_error()
    wrapper.minimize_capital(target_annual_dividend=TARGET, max_position_pct=0.2)
    assert wrapper.get_last_error() is None


def test_efficient_frontier(wrapper):
    frontier = wrapper.generate_efficient_frontier(TARGET, [0.1, 0.9], max_position_pct=0.15)
    assert [point["status"] for point in frontier] == ["success", "success"]
    assert frontier[0]["cashflow_std"] <= frontier[1]["cashflow_std"] + 1e-6

    failed = wrapper.generate_efficient_frontier(TARGET, [0.5], max_position_pct=0.01)
    assert failed[0]["status"] == "failed"


def test_ticker_summary(wrapper):
    summary = wrapper.get_ticker_summary()
    assert summary["total_tickers"] == 9
    assert summary["sectors"] == {"Sector A": 3, "Sector B": 3, "Sector C": 3}
    assert 0 < summary["min_yield"] < summary["max_yield"] < 1
    assert sum(summary["yield_distribution"].values()) == 9


def test_filter_tickers_by_criteria(wrapper):
    assert set(wrapper.filter_tickers_by_criteria(min_yield=0.08)) == {"AAA1", "AAA2", "AAA3"}
    assert set(wrapper.filter_tickers_by_criteria(sectors=["Sector B"])) == {"BBB1", "BBB2", "BBB3"}
    assert wrapper.filter_tickers_by_criteria(max_price=10) == []


def test_portfolio_metrics(wrapper):
    portfolio = wrapper.minimize_capital(target_annual_dividend=TARGET, max_position_pct=0.2)["portfolio"]
    metrics = wrapper.calculate_portfolio_metrics(portfolio)
    assert sum(metrics["sector_allocation"].values()) == pytest.approx(1.0)
    assert metrics["concentration_metrics"]["effective_positions"] == pytest.approx(5, rel=1e-2)
    assert wrapper.calculate_portfolio_metrics([])["sector_allocation"] == {}


def test_create_optimizer_wrapper():
    assert isinstance(create_optimizer_wrapper(make_yfinance_like_data()), OptimizerWrapper)
