"""Tests for DividendPortfolioOptimizer."""

import numpy as np
import pandas as pd
import pytest

from src.dividendportfoliooptimizer import DividendPortfolioOptimizer, main

from .conftest import AS_OF, make_dividend_df

TARGET = 10_000


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        DividendPortfolioOptimizer("dividend_info.parquet")  # type: ignore[arg-type]


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        DividendPortfolioOptimizer(pd.DataFrame({"ticker": ["X"]}))


def test_annual_yield_is_a_fraction_of_ttm_dividends(optimizer):
    data = optimizer.ticker_data["AAA1"]
    assert data["ttm_dividend"] == pytest.approx(6.0)  # 4 x $1.50
    assert data["annual_yield"] == pytest.approx(0.12)
    assert data["dividend_yield"] == pytest.approx(12.0)


def test_current_price_column_is_preferred():
    df = make_dividend_df({"X": ("A", 1.0, 40.0)})
    df["current_price"] = 80.0
    opt = DividendPortfolioOptimizer(df, as_of=AS_OF)
    assert opt.ticker_data["X"]["current_price"] == 80.0
    assert opt.ticker_data["X"]["annual_yield"] == pytest.approx(4.0 / 80.0)


def test_ttm_is_measured_from_as_of_not_each_tickers_last_payment():
    # Y stopped paying in 2023, so it has no dividends in the trailing twelve months.
    df = pd.concat(
        [
            make_dividend_df({"X": ("A", 1.0, 50.0)}),
            make_dividend_df({"Y": ("B", 1.0, 50.0)}, years=(2022, 2023)),
        ]
    )
    opt = DividendPortfolioOptimizer(df, as_of=AS_OF)
    assert opt.ticker_data["Y"]["annual_yield"] == 0.0
    assert opt.ticker_data["X"]["annual_yield"] > 0


def test_timezone_aware_dates_are_accepted():
    df = make_dividend_df({"X": ("A", 1.0, 50.0)})
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"]).dt.tz_localize("America/New_York")
    opt = DividendPortfolioOptimizer(df, as_of=AS_OF)
    assert opt.ticker_data["X"]["annual_yield"] == pytest.approx(0.08)


def test_monthly_matrix_is_raw_yield_per_dollar(optimizer):
    row = optimizer.monthly_matrix[optimizer.tickers.index("AAA1")]
    expected = np.zeros(12)
    expected[[0, 3, 6, 9]] = 1.5 / 50.0
    np.testing.assert_allclose(row, expected)
    # Not z-scored: sums to the annual yield, not to zero.
    assert row.sum() == pytest.approx(optimizer.ticker_data["AAA1"]["annual_yield"])
    assert optimizer.raw_monthly_matrix is optimizer.monthly_matrix


def test_new_dividend_payer_is_not_penalized():
    # Only one year of history: the missing earlier years must not count as $0 years.
    opt = DividendPortfolioOptimizer(make_dividend_df({"NEW": ("A", 1.0, 50.0)}, years=(2025,)), as_of=AS_OF)
    assert opt.monthly_matrix[0].sum() == pytest.approx(opt.ticker_data["NEW"]["annual_yield"])


def test_two_payments_in_one_month_are_summed():
    df = pd.DataFrame(
        {
            "ticker": ["X", "X"],
            "ex_dividend_date": ["2025-03-01", "2025-03-20"],
            "dividend_amount": [1.0, 0.5],
            "close_price": [50.0, 50.0],
        }
    )
    opt = DividendPortfolioOptimizer(df, as_of=AS_OF)
    assert opt.monthly_matrix[0, 2] == pytest.approx(1.5 / 50.0)


# ----------------------------------------------------------------------
# Minimize capital
# ----------------------------------------------------------------------
def test_minimize_capital_picks_highest_yields_equally(optimizer):
    result = optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.2, min_yield_threshold=0.0)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["actual_annual_dividend"] == pytest.approx(TARGET, rel=1e-3)

    # A 20% cap means 5 equal positions in the 5 highest yields.
    top5 = sorted(optimizer.ticker_data.values(), key=lambda d: d["annual_yield"], reverse=True)[:5]
    expected_capital = TARGET / np.mean([d["annual_yield"] for d in top5])
    assert result["total_capital_used"] == pytest.approx(expected_capital, rel=1e-3)
    assert len(result["portfolio"]) == 5
    for pos in result["portfolio"]:
        assert pos["position_pct"] == pytest.approx(0.2, abs=1e-3)
        assert 0 < pos["annual_yield"] < 1  # a fraction, not a percent


def test_min_yield_threshold_excludes_low_yielders(optimizer):
    result = optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.25, min_yield_threshold=0.06)
    assert {pos["ticker"] for pos in result["portfolio"]} <= {"AAA1", "AAA2", "AAA3", "BBB1", "CCC1"}


def test_too_few_tickers_for_position_cap_gives_clear_error(optimizer):
    with pytest.raises(ValueError, match="needs at least 20 eligible tickers"):
        optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.05)


def test_budget_constraint_is_enforced(optimizer):
    with pytest.raises(RuntimeError, match="infeasible"):
        optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.2, budget_constraint=10_000)


def test_invalid_inputs_are_rejected(optimizer):
    with pytest.raises(ValueError, match="target_annual_dividend"):
        optimizer.minimize_capital_for_target_yield(0)
    with pytest.raises(ValueError, match="max_position_pct"):
        optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=1.5)


# ----------------------------------------------------------------------
# Smoothing and multi-objective
# ----------------------------------------------------------------------
def test_smooth_cashflow_is_smoother_than_min_capital(optimizer):
    min_cap = optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.15, min_yield_threshold=0.0)
    smooth = optimizer.smooth_cashflow_seasonality(TARGET, max_position_pct=0.15)

    assert smooth["actual_annual_dividend"] >= TARGET * 0.999
    assert smooth["cashflow_std"] < min_cap["cashflow_std"] / 10
    assert smooth["total_capital_used"] >= min_cap["total_capital_used"]
    assert smooth["portfolio"], "smoothing must return a portfolio"
    assert len(smooth["monthly_cashflows"]) == 12
    # Reported monthly flows are dollars and add up to roughly the annual dividend.
    assert sum(smooth["monthly_cashflows"]) == pytest.approx(TARGET, rel=0.01)


def test_smooth_cashflow_respects_budget(optimizer):
    # Here, minimum capital is ~$121k and unconstrained smoothing uses ~$154k,
    # so a $130k budget is feasible but binding.
    unconstrained = optimizer.smooth_cashflow_seasonality(TARGET, max_position_pct=0.15)
    limited = optimizer.smooth_cashflow_seasonality(TARGET, max_position_pct=0.15, budget_constraint=130_000)
    assert limited["total_capital_used"] <= 130_000 * 1.001
    assert limited["cashflow_std"] > unconstrained["cashflow_std"]


def test_multi_objective_trades_capital_for_smoothness(optimizer):
    capital_heavy = optimizer.multi_objective_optimization(TARGET, 0.9, 0.1, max_position_pct=0.15)
    smooth_heavy = optimizer.multi_objective_optimization(TARGET, 0.1, 0.9, max_position_pct=0.15)

    assert capital_heavy["total_capital_used"] <= smooth_heavy["total_capital_used"] + 1
    assert smooth_heavy["cashflow_std"] <= capital_heavy["cashflow_std"] + 1e-6
    assert capital_heavy["lambda_capital"] == 0.9


def test_multi_objective_validates_lambdas(optimizer):
    with pytest.raises(ValueError, match="lambda_capital"):
        optimizer.multi_objective_optimization(TARGET, lambda_capital=1.5, lambda_smoothness=0.0)


def test_result_keys_are_consistent(optimizer):
    expected = {
        "status",
        "total_capital_used",
        "target_annual_dividend",
        "actual_annual_dividend",
        "monthly_cashflows",
        "cashflow_std",
        "min_monthly",
        "max_monthly",
        "portfolio",
    }
    for result in (
        optimizer.minimize_capital_for_target_yield(TARGET, max_position_pct=0.2),
        optimizer.smooth_cashflow_seasonality(TARGET, max_position_pct=0.2),
        optimizer.multi_objective_optimization(TARGET, max_position_pct=0.2),
    ):
        assert expected <= result.keys()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_reads_parquet_file(tmp_path, capsys, dividend_df):
    path = tmp_path / "dividend_info.parquet"
    dividend_df.to_parquet(path)

    main(["--data-file", str(path), "--target", "5000", "--max-position", "0.2"])

    out = capsys.readouterr().out
    assert "Error loading data" not in out
    assert out.count("OPTIMIZATION RESULTS") == 3
    assert "FAILED" not in out


def test_cli_reports_missing_file(tmp_path, capsys):
    main(["--data-file", str(tmp_path / "missing.parquet")])
    assert "Could not find data file" in capsys.readouterr().out
