"""
Wrapper module to bridge between Streamlit app and DividendPortfolioOptimizer.
This handles parameter translation, error handling, and provides additional functionality.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dividendportfoliooptimizer import DividendPortfolioOptimizer


class OptimizerWrapper:
    """
    Wrapper class that provides a consistent interface between Streamlit and the optimizer.
    Handles parameter translation, error recovery, and additional functionality.
    """

    def __init__(self, tickers_data: Dict[str, Any]) -> None:
        """
        Initialize the wrapper with the optimizer.
        
        Args:
            tickers_data: Dictionary with ticker symbols as keys and their data as values.
                         Each value should be a dict containing 'dividends', 'close', 
                         'company_name', 'sector', and 'currency'.
        """
        # Convert yfinance data to the format expected by DividendPortfolioOptimizer
        df = self._prepare_dividend_data(tickers_data)
        self.optimizer = DividendPortfolioOptimizer(df)
        self._last_error: Optional[str] = None  # Add type annotation here
        self._last_error = None

    def get_last_error(self) -> Optional[str]:
        """Return the last error message if any."""
        return self._last_error

    def _safe_call(self, method_name: str, **kwargs) -> Optional[Dict]:
        """
        Safely call an optimizer method with multiple parameter combinations.
        Tries different parameter sets to handle version differences.
        """
        self._last_error = None
        method = getattr(self.optimizer, method_name)

        # Define parameter combinations to try (in order)
        param_combinations = self._get_param_combinations(method_name, kwargs)

        for params in param_combinations:
            try:
                result = method(**params)
                # Ensure result has all expected fields
                return self._normalize_result(result, method_name)
            except TypeError:
                # Parameter mismatch - try next combination
                continue
            except Exception as e:
                # Other error - save and raise
                self._last_error = str(e)
                raise

        # If we get here, no parameter combination worked
        self._last_error = f"No compatible parameter set found for {method_name}"
        raise TypeError(self._last_error)

    def _get_param_combinations(self, method_name: str, kwargs: Dict) -> List[Dict]:
        """
        Generate parameter combinations to try for each method.
        Orders them from most complete to most basic.
        """
        if method_name == "minimize_capital_for_target_yield":
            # Required parameters
            if "target_annual_dividend" not in kwargs:
                raise ValueError(
                    "minimize_capital_for_target_yield requires 'target_annual_dividend' parameter"
                )

            return [
                {
                    "target_annual_dividend": kwargs["target_annual_dividend"],
                    "max_position_pct": kwargs.get("max_position_pct", 0.1),
                    "min_yield_threshold": kwargs.get("min_yield_threshold", 0.01),
                }
            ]

        elif method_name == "smooth_cashflow_seasonality":
            if "target_annual_dividend" not in kwargs:
                raise ValueError(
                    "smooth_cashflow_seasonality requires 'target_annual_dividend' parameter"
                )

            return [
                {
                    "target_annual_dividend": kwargs["target_annual_dividend"],
                    "budget_constraint": kwargs.get("budget_constraint"),
                    "max_position_pct": kwargs.get("max_position_pct", 0.15),
                }
            ]

        elif method_name == "multi_objective_optimization":
            # Required parameters
            if "target_annual_dividend" not in kwargs:
                raise ValueError(
                    "multi_objective_optimization requires 'target_annual_dividend' parameter"
                )

            # Validate lambda values
            lambda_capital = kwargs.get("lambda_capital", 0.5)
            lambda_smoothness = kwargs.get("lambda_smoothness", 0.5)

            if not (0 <= lambda_capital <= 1):
                raise ValueError(
                    f"lambda_capital must be between 0 and 1, got {lambda_capital}"
                )
            if not (0 <= lambda_smoothness <= 1):
                raise ValueError(
                    f"lambda_smoothness must be between 0 and 1, got {lambda_smoothness}"
                )
            if abs((lambda_capital + lambda_smoothness) - 1.0) > 0.01:
                raise ValueError(
                    f"lambda_capital ({lambda_capital}) + lambda_smoothness ({lambda_smoothness}) should equal 1.0"
                )

            return [
                {
                    "target_annual_dividend": kwargs["target_annual_dividend"],
                    "lambda_capital": lambda_capital,
                    "lambda_smoothness": lambda_smoothness,
                    "max_position_pct": kwargs.get("max_position_pct", 0.12),
                }
            ]

        else:
            raise ValueError(f"Unknown optimization method: {method_name}")

    def _normalize_result(self, result: Dict, method_name: str) -> Dict:
        """
        Normalize results to ensure consistent field names across different methods.
        """
        # Map alternative field names to standard ones
        field_mappings = {
            "total_capital_needed": "total_capital_used",
            "actual_dividend": "actual_annual_dividend",
            "target_dividend": "target_annual_dividend",
            "cashflow_variance": "cashflow_variance",
            "monthly_cashflows": "monthly_cashflows",
        }

        # Apply mappings
        for old_name, new_name in field_mappings.items():
            if old_name in result and new_name not in result:
                result[new_name] = result[old_name]

        # Ensure all expected fields exist
        defaults = {
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

        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value

        # Calculate derived fields if needed
        # Properly handle numpy arrays and check for non-empty arrays
        monthly_cf = result["monthly_cashflows"]
        if isinstance(monthly_cf, np.ndarray):
            if monthly_cf.size > 0:
                if "min_monthly" not in result or result["min_monthly"] == 0:
                    result["min_monthly"] = float(np.min(monthly_cf))
                if "max_monthly" not in result or result["max_monthly"] == 0:
                    result["max_monthly"] = float(np.max(monthly_cf))
                if "cashflow_std" not in result or result["cashflow_std"] == 0:
                    result["cashflow_std"] = float(np.std(monthly_cf))
        elif isinstance(monthly_cf, list) and len(monthly_cf) > 0:
            if "min_monthly" not in result or result["min_monthly"] == 0:
                result["min_monthly"] = float(np.min(monthly_cf))
            if "max_monthly" not in result or result["max_monthly"] == 0:
                result["max_monthly"] = float(np.max(monthly_cf))
            if "cashflow_std" not in result or result["cashflow_std"] == 0:
                result["cashflow_std"] = float(np.std(monthly_cf))

        return result

    def minimize_capital(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for minimize_capital_for_target_yield with parameter handling."""
        result = self._safe_call("minimize_capital_for_target_yield", **kwargs)
        if result is None:
            raise RuntimeError("Optimization returned no result")
        return result

    def smooth_cashflow(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for smooth_cashflow_seasonality with parameter handling."""
        result = self._safe_call("smooth_cashflow_seasonality", **kwargs)
        if result is None:
            raise RuntimeError("Optimization returned no result")
        return result

    def multi_objective(self, **kwargs) -> Dict[str, Any]:
        """Wrapper for multi_objective_optimization with parameter handling."""
        result = self._safe_call("multi_objective_optimization", **kwargs)
        if result is None:
            raise RuntimeError("Optimization returned no result")
        return result

    def generate_efficient_frontier(
        self,
        target_annual_dividend: float,
        lambda_values: List[float],
        max_position_pct: float = 0.15,
        budget_constraint: Optional[float] = None,
    ) -> List[Dict]:
        """
        Generate efficient frontier data points.

        Returns:
            List of dictionaries with frontier results
        """
        frontier_results = []

        for lambda_capital in lambda_values:
            lambda_smoothness = 1.0 - lambda_capital

            try:
                result = self.multi_objective(
                    target_annual_dividend=target_annual_dividend,
                    lambda_capital=lambda_capital,
                    lambda_smoothness=lambda_smoothness,
                    max_position_pct=max_position_pct,
                    budget_constraint=budget_constraint,
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

    def get_ticker_summary(self) -> Dict:
        """
        Get summary statistics about available tickers.

        Returns:
            Dictionary with ticker statistics
        """
        ticker_data = self.optimizer.ticker_data

        yields = [data["annual_yield"] for data in ticker_data.values()]
        sectors = {}
        for ticker, data in ticker_data.items():
            sector = data.get("sector", "Unknown")
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += 1

        return {
            "total_tickers": len(ticker_data),
            "average_yield": np.mean(yields),
            "median_yield": np.median(yields),
            "min_yield": np.min(yields),
            "max_yield": np.max(yields),
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
        """
        Filter tickers based on criteria.

        Returns:
            List of ticker symbols meeting criteria
        """
        filtered = []

        for ticker, data in self.optimizer.ticker_data.items():
            # Yield filter
            if not (min_yield <= data["annual_yield"] <= max_yield):
                continue

            # Price filter
            if not (min_price <= data["current_price"] <= max_price):
                continue

            # Sector filter
            if sectors and data.get("sector") not in sectors:
                continue

            filtered.append(ticker)

        return filtered

    def calculate_portfolio_metrics(self, portfolio: List[Dict]) -> Dict:
        """
        Calculate additional metrics for a portfolio.

        Args:
            portfolio: List of portfolio positions

        Returns:
            Dictionary with portfolio metrics
        """
        if not portfolio:
            return {
                "sector_allocation": {},
                "concentration_metrics": {},
                "yield_stats": {},
            }

        # Sector allocation
        sector_allocation = {}
        total_capital = sum(pos["capital_allocated"] for pos in portfolio)

        for pos in portfolio:
            ticker = pos["ticker"]
            sector = self.optimizer.ticker_data[ticker].get("sector", "Unknown")
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += pos["capital_allocated"] / total_capital

        # Concentration metrics
        position_sizes = [pos["position_pct"] for pos in portfolio]
        top5_concentration = sum(sorted(position_sizes, reverse=True)[:5])
        top10_concentration = sum(sorted(position_sizes, reverse=True)[:10])

        # Yield statistics
        weighted_yields = [
            pos["position_pct"] * pos["annual_yield"] for pos in portfolio
        ]

        return {
            "sector_allocation": sector_allocation,
            "concentration_metrics": {
                "top5_concentration": top5_concentration,
                "top10_concentration": top10_concentration,
                "herfindahl_index": sum(p**2 for p in position_sizes),
                "effective_positions": 1 / sum(p**2 for p in position_sizes)
                if position_sizes
                else 0,
            },
            "yield_stats": {
                "weighted_average_yield": sum(weighted_yields),
                "min_yield": min(pos["annual_yield"] for pos in portfolio),
                "max_yield": max(pos["annual_yield"] for pos in portfolio),
            },
        }


# Convenience function for Streamlit app
    def _prepare_dividend_data(self, tickers_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert yfinance data to the format expected by DividendPortfolioOptimizer."""
        records = []
        
        for ticker, data in tickers_data.items():
            if not data['dividends'].empty:
                for date, amount in data['dividends'].items():
                    if amount > 0:  # Only include actual dividend payments
                        close_price = data['close'].asof(date)
                        records.append({
                            'ticker': ticker,
                            'ex_dividend_date': date,
                            'dividend_amount': amount,
                            'close_price': close_price,
                            'company_name': data.get('info', {}).get('shortName', ticker),
                            'sector': data.get('info', {}).get('sector', 'Unknown'),
                            'currency': data.get('info', {}).get('currency', 'USD')
                        })
        
        if not records:
            raise ValueError("No valid dividend data found in the provided tickers")
            
        df = pd.DataFrame(records)
        df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'])
        return df


def create_optimizer_wrapper(tickers_data: Dict[str, Any]):
    """
    Create and return an optimizer wrapper instance.
    
    Args:
        tickers_data: Dictionary with ticker symbols as keys and their data as values.
                     Each value should be a dict containing 'dividends', 'close', 
                     'info' with 'shortName', 'sector', and 'currency'.
    """
    return OptimizerWrapper(tickers_data)
