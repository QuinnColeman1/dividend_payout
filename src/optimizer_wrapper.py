"""
Bridge between the Streamlit app and DividendPortfolioOptimizer.

Converts yfinance data into the optimizer's input format, exposes the
optimizations with consistent parameter names, and adds summary helpers.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .dividendportfoliooptimizer import DividendPortfolioOptimizer

RESULT_DEFAULTS: Dict[str, Any] = {
    "status": "unknown",
    "total_capital_used": 0.0,
    "actual_annual_dividend": 0.0,
    "target_annual_dividend": 0.0,
    "portfolio": [],
    "cashflow_std": 0.0,
    "monthly_cashflows": [],
    "min_monthly": 0.0,
    "max_monthly": 0.0,
}


class OptimizerWrapper:
    """Consistent interface between the Streamlit app and the optimizer."""

    def __init__(self, tickers_data: Dict[str, Any], as_of: Optional[Any] = None) -> None:
        """
        Args:
            tickers_data: Ticker symbol -> dict with 'dividends' (Series indexed by ex-date),
                'close' (Series of daily closes) and optional 'info' (yfinance info dict
                with 'shortName', 'sector', 'currency').
            as_of: Optional reference date for yields; defaults to the latest dividend date.
        """
        df = self._prepare_dividend_data(tickers_data)
        self.optimizer = DividendPortfolioOptimizer(df, as_of=as_of)
        self._last_error: Optional[str] = None

    @staticmethod
    def _prepare_dividend_data(tickers_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert yfinance data to the DataFrame format DividendPortfolioOptimizer expects."""
        records = []

        for ticker, data in tickers_data.items():
            dividends = data.get("dividends")
            close = data.get("close")
            if dividends is None or dividends.empty or close is None:
                continue
            close = close.dropna().sort_index()
            if close.empty:
                continue

            info = data.get("info") or {}
            current_price = float(close.iloc[-1])

            for date, amount in dividends.items():
                if amount > 0:  # only actual dividend payments
                    close_price = close.asof(date)
                    records.append(
                        {
                            "ticker": ticker,
                            "ex_dividend_date": date,
                            "dividend_amount": float(amount),
                            "close_price": float(close_price) if pd.notna(close_price) else current_price,
                            "current_price": current_price,
                            "company_name": info.get("shortName", ticker),
                            "sector": info.get("sector", "Unknown"),
                            "currency": info.get("currency", "USD"),
                        }
                    )

        if not records:
            raise ValueError("No valid dividend data found in the provided tickers")

        return pd.DataFrame(records)

    def get_last_error(self) -> Optional[str]:
        """Return the last error message, if any."""
        return self._last_error

    def _run(self, method: Callable[..., Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Call an optimizer method, recording any error message before re-raising."""
        self._last_error = None
        try:
            result = method(**kwargs)
        except Exception as e:
            self._last_error = str(e)
            raise
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Make sure every field the UI reads is present."""
        for key, default in RESULT_DEFAULTS.items():
            result.setdefault(key, default)
        return result

    def minimize_capital(
        self,
        target_annual_dividend: float,
        max_position_pct: float = 0.1,
        min_yield_threshold: float = 0.01,
        budget_constraint: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._run(
            self.optimizer.minimize_capital_for_target_yield,
            target_annual_dividend=target_annual_dividend,
            max_position_pct=max_position_pct,
            min_yield_threshold=min_yield_threshold,
            budget_constraint=budget_constraint,
        )

    def smooth_cashflow(
        self,
        target_annual_dividend: float,
        budget_constraint: Optional[float] = None,
        max_position_pct: float = 0.15,
        min_yield_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        return self._run(
            self.optimizer.smooth_cashflow_seasonality,
            annual_dividend_target=target_annual_dividend,
            budget_constraint=budget_constraint,
            max_position_pct=max_position_pct,
            min_yield_threshold=min_yield_threshold,
        )

    def multi_objective(
        self,
        target_annual_dividend: float,
        lambda_capital: float = 0.5,
        lambda_smoothness: float = 0.5,
        max_position_pct: float = 0.12,
        budget_constraint: Optional[float] = None,
        min_yield_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        if abs((lambda_capital + lambda_smoothness) - 1.0) > 0.01:
            raise ValueError(
                f"lambda_capital ({lambda_capital}) + lambda_smoothness ({lambda_smoothness}) should equal 1.0"
            )
        return self._run(
            self.optimizer.multi_objective_optimization,
            target_annual_dividend=target_annual_dividend,
            lambda_capital=lambda_capital,
            lambda_smoothness=lambda_smoothness,
            max_position_pct=max_position_pct,
            budget_constraint=budget_constraint,
            min_yield_threshold=min_yield_threshold,
        )

    def generate_efficient_frontier(
        self,
        target_annual_dividend: float,
        lambda_values: List[float],
        max_position_pct: float = 0.15,
        budget_constraint: Optional[float] = None,
        min_yield_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Run the multi-objective optimization for each capital weight in lambda_values."""
        frontier_results: List[Dict[str, Any]] = []

        for lambda_capital in lambda_values:
            lambda_smoothness = round(1.0 - lambda_capital, 10)
            try:
                result = self.multi_objective(
                    target_annual_dividend=target_annual_dividend,
                    lambda_capital=lambda_capital,
                    lambda_smoothness=lambda_smoothness,
                    max_position_pct=max_position_pct,
                    budget_constraint=budget_constraint,
                    min_yield_threshold=min_yield_threshold,
                )
                frontier_results.append(
                    {
                        "lambda_capital": lambda_capital,
                        "lambda_smoothness": lambda_smoothness,
                        "total_capital_used": result["total_capital_used"],
                        "cashflow_std": result["cashflow_std"],
                        "portfolio_size": len(result["portfolio"]),
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as e:
                frontier_results.append(
                    {
                        "lambda_capital": lambda_capital,
                        "lambda_smoothness": lambda_smoothness,
                        "total_capital_used": None,
                        "cashflow_std": None,
                        "portfolio_size": 0,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return frontier_results

    def get_ticker_summary(self) -> Dict[str, Any]:
        """Summary statistics about the loaded tickers (yields are fractions)."""
        ticker_data = self.optimizer.ticker_data
        yields = [data["annual_yield"] for data in ticker_data.values()]

        sectors: Dict[str, int] = {}
        for data in ticker_data.values():
            sector = data.get("sector", "Unknown")
            sectors[sector] = sectors.get(sector, 0) + 1

        return {
            "total_tickers": len(ticker_data),
            "average_yield": float(np.mean(yields)) if yields else 0.0,
            "median_yield": float(np.median(yields)) if yields else 0.0,
            "min_yield": float(np.min(yields)) if yields else 0.0,
            "max_yield": float(np.max(yields)) if yields else 0.0,
            "sectors": sectors,
            "yield_distribution": {
                "below_1pct": sum(1 for y in yields if y < 0.01),
                "1_to_3pct": sum(1 for y in yields if 0.01 <= y < 0.03),
                "3_to_5pct": sum(1 for y in yields if 0.03 <= y < 0.05),
                "above_5pct": sum(1 for y in yields if y >= 0.05),
            },
        }

    def filter_tickers_by_criteria(
        self,
        min_yield: float = 0.0,
        max_yield: float = 1.0,
        sectors: Optional[List[str]] = None,
        min_price: float = 0.0,
        max_price: float = float("inf"),
    ) -> List[str]:
        """Ticker symbols meeting the yield (fraction), sector and price criteria."""
        filtered = []
        for ticker, data in self.optimizer.ticker_data.items():
            if not (min_yield <= data["annual_yield"] <= max_yield):
                continue
            if not (min_price <= data["current_price"] <= max_price):
                continue
            if sectors and data.get("sector") not in sectors:
                continue
            filtered.append(ticker)
        return filtered

    def calculate_portfolio_metrics(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sector allocation, concentration and yield statistics for a portfolio."""
        total_capital = sum(pos["capital_allocated"] for pos in portfolio)
        if not portfolio or total_capital <= 0:
            return {"sector_allocation": {}, "concentration_metrics": {}, "yield_stats": {}}

        sector_allocation: Dict[str, float] = {}
        for pos in portfolio:
            sector = self.optimizer.ticker_data[pos["ticker"]].get("sector", "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0.0) + pos["capital_allocated"] / total_capital

        position_sizes = sorted((pos["position_pct"] for pos in portfolio), reverse=True)
        herfindahl = sum(p**2 for p in position_sizes)

        return {
            "sector_allocation": sector_allocation,
            "concentration_metrics": {
                "top5_concentration": sum(position_sizes[:5]),
                "top10_concentration": sum(position_sizes[:10]),
                "herfindahl_index": herfindahl,
                "effective_positions": 1 / herfindahl if herfindahl > 0 else 0,
            },
            "yield_stats": {
                "weighted_average_yield": sum(pos["position_pct"] * pos["annual_yield"] for pos in portfolio),
                "min_yield": min(pos["annual_yield"] for pos in portfolio),
                "max_yield": max(pos["annual_yield"] for pos in portfolio),
            },
        }


def create_optimizer_wrapper(tickers_data: Dict[str, Any]) -> OptimizerWrapper:
    """Create an OptimizerWrapper for the Streamlit app."""
    return OptimizerWrapper(tickers_data)
