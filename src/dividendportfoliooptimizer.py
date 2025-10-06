import warnings
from datetime import timedelta
from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DividendPortfolioOptimizer:
    def __init__(self, dividend_df: pd.DataFrame):
        """
        Initialize the optimizer with dividend data.

        Args:
            dividend_df: DataFrame containing dividend data with columns:
                        - ticker: str
                        - ex_dividend_date: datetime
                        - dividend_amount: float
                        - close_price: float
                        - company_name: str (optional)
                        - sector: str (optional)
                        - currency: str (optional)
        """
        required_columns = ['ticker', 'ex_dividend_date', 'dividend_amount', 'close_price']
        missing_columns = [col for col in required_columns if col not in dividend_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in input DataFrame: {missing_columns}")
            
        self.df = dividend_df.copy()
        self.df["ex_dividend_date"] = pd.to_datetime(self.df["ex_dividend_date"])
        self.df["month"] = self.df["ex_dividend_date"].dt.month
        self.df["year"] = self.df["ex_dividend_date"].dt.year

        # Add default values for optional columns if not present
        if 'company_name' not in self.df.columns:
            self.df['company_name'] = self.df['ticker']
        if 'sector' not in self.df.columns:
            self.df['sector'] = 'Unknown'
        if 'currency' not in self.df.columns:
            self.df['currency'] = 'USD'

        # Prepare data structures
        self.ticker_data = self._prepare_ticker_data()
        self.monthly_matrix = self._build_monthly_dividend_matrix()

        print(f"Loaded data for {len(self.ticker_data)} tickers")
        if not self.df.empty:
            print(
                f"Date range: {self.df['ex_dividend_date'].min()} to {self.df['ex_dividend_date'].max()}"
            )
        else:
            print("Warning: No dividend data found")

    def _prepare_ticker_data(self) -> Dict[str, Dict[str, Any]]:
        """Prepare ticker-level data including current yields and prices."""
        ticker_data: Dict[str, Dict[str, Any]] = {}
        
        if self.df.empty:
            return ticker_data

        for ticker in self.df["ticker"].unique():
            ticker_df = self.df[self.df["ticker"] == ticker].copy()
            
            if ticker_df.empty:
                continue

            # Get most recent data
            recent = ticker_df.loc[ticker_df["ex_dividend_date"].idxmax()]
            
            # Calculate cutoff date for TTM (Trailing Twelve Months) dividend yield
            cutoff_date = recent["ex_dividend_date"] - timedelta(days=365)
            
            # Calculate TTM (Trailing Twelve Months) dividend yield
            ttm_dividends = ticker_df[ticker_df["ex_dividend_date"] >= cutoff_date][
                "dividend_amount"
            ].sum()
            
            # Get company info from the most recent record
            ticker_data[ticker] = {
                "company_name": recent.get("company_name", ticker),
                "sector": recent.get("sector", "Unknown"),
                "currency": recent.get("currency", "USD"),
                "current_price": recent["close_price"],
                "ttm_dividend": ttm_dividends,
                "dividend_yield": (ttm_dividends / recent["close_price"] * 100) if recent["close_price"] > 0 else 0,
            }
        return ticker_data

    def _build_monthly_dividend_matrix(self) -> np.ndarray:
        """Build matrix of monthly dividends per dollar invested with robust normalization."""
        if not self.ticker_data:
            return np.zeros((0, 12))
            
        tickers = list(self.ticker_data.keys())
        monthly_matrix = np.zeros((len(tickers), 12))

        # Use last 3 years of data for monthly patterns
        if self.df.empty:
            return monthly_matrix
            
        recent_data = self.df[
            self.df["ex_dividend_date"]
            >= (self.df["ex_dividend_date"].max() - timedelta(days=1095))
        ]

        for i, ticker in enumerate(tickers):
            ticker_df = recent_data[recent_data["ticker"] == ticker]
            current_price = self.ticker_data[ticker]["current_price"]

            # Calculate weighted average for each month
            for month in range(1, 13):
                # Get dividends by year
                year1_cutoff = self.df["ex_dividend_date"].max() - timedelta(days=365)
                year2_cutoff = self.df["ex_dividend_date"].max() - timedelta(days=730)

                # Replace lines 73-79 with:
                year1_divs = ticker_df[
                    (ticker_df["month"] == month)
                    & (ticker_df["ex_dividend_date"] >= year1_cutoff)
                ]["dividend_amount"].mean()
                year2_divs = ticker_df[
                    (ticker_df["month"] == month)
                    & (ticker_df["ex_dividend_date"] < year1_cutoff)
                    & (ticker_df["ex_dividend_date"] >= year2_cutoff)
                ]["dividend_amount"].mean()
                year3_divs = ticker_df[
                    (ticker_df["month"] == month)
                    & (ticker_df["ex_dividend_date"] < year2_cutoff)
                ]["dividend_amount"].mean()

                # Handle NaN values (months with no dividends)
                year1_divs = 0 if pd.isna(year1_divs) else year1_divs
                year2_divs = 0 if pd.isna(year2_divs) else year2_divs
                year3_divs = 0 if pd.isna(year3_divs) else year3_divs

                # Weighted average: 50% recent year, 33% year 2, 17% year 3
                month_divs = year1_divs * 0.5 + year2_divs * 0.33 + year3_divs * 0.17

                # Normalize by current price to get yield per dollar invested
                monthly_matrix[i, month - 1] = (
                    month_divs / current_price if current_price > 0 else 0
                )

        # Store the raw matrix for calculating actual cash flows later
        self.raw_monthly_matrix = monthly_matrix.copy()

        # Apply normalization for optimization purposes only
        normalized_matrix = np.zeros_like(monthly_matrix)
        for i in range(len(tickers)):
            row = monthly_matrix[i, :]
            row_mean = np.mean(row)
            row_std = np.std(row, ddof=0)

            # Apply minimum standard deviation floor to prevent division issues
            min_std = np.float64(1e-6)
            row_std = np.maximum(row_std.astype(np.float64), min_std).astype(row.dtype)

            # Standardize: (x - mean) / std
            normalized_matrix[i, :] = (row - row_mean) / row_std

        return normalized_matrix

    def minimize_capital_for_target_yield(
        self,
        target_annual_dividend: float,
        max_position_pct: float = 0.1,
        min_yield_threshold: float = 0.01,
    ) -> Dict:
        """
        Solve: Minimize total capital needed to achieve target annual dividend.

        Args:
            target_annual_dividend: Target annual dividend income ($)
            max_position_pct: Maximum position size as fraction of total (0.1 = 10%)
            min_yield_threshold: Minimum yield to consider a stock (0.01 = 1%)

        Returns:
            Dictionary with optimization results
        """
        # Filter tickers with reasonable yields
        valid_tickers = [
            ticker
            for ticker in self.ticker_data.keys()
            if self.ticker_data[ticker]["annual_yield"] >= min_yield_threshold
        ]

        n = len(valid_tickers)
        if n == 0:
            raise ValueError("No tickers meet the minimum yield threshold")

        # Scale to thousands for better numerical stability
        scale = 1000
        target_scaled = target_annual_dividend / scale

        # Decision variables: capital allocation per ticker (in thousands)
        x = cp.Variable(n, nonneg=True)

        # Objective: minimize total capital
        objective = cp.Minimize(cp.sum(x))

        # Constraint: achieve target dividend yield
        yields = np.array(
            [self.ticker_data[ticker].get("annual_yield", 0.0) or 0.0 for ticker in valid_tickers]
        )
        # Ensure no None or NaN values in yields
        yields = np.nan_to_num(yields, nan=0.0, posinf=0.0, neginf=0.0)
        dividend_constraint = cp.sum(cp.multiply(x, yields)) >= target_scaled

        # Position size constraints
        total_capital = cp.sum(x)
        position_constraints = [
            x[i] <= max_position_pct * total_capital for i in range(n)
        ]

        # Solve with adjusted parameters
        constraints = [dividend_constraint] + position_constraints
        problem = cp.Problem(objective, constraints)

        # Try with different solvers
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except (cp.SolverError, ArithmeticError, ValueError, TypeError):
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except (cp.SolverError, ArithmeticError, ValueError, TypeError):
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-3)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")

        # Extract results and ensure proper type handling
        if x.value is None:
            raise RuntimeError("Optimization failed: No solution found")
            
        allocations = np.array(x.value, dtype=np.float64) * scale
        total_capital_needed = float(np.sum(allocations))

        results = {
            "status": problem.status,
            "total_capital_needed": total_capital_needed,
            "target_dividend": target_annual_dividend,
            "actual_dividend": np.sum(allocations * yields),
            "portfolio": [],
        }

        for i, ticker in enumerate(valid_tickers):
            if allocations[i] > 1:  # Only include meaningful positions
                results["portfolio"].append(
                    {
                        "ticker": ticker,
                        "company_name": self.ticker_data[ticker]["company_name"],
                        "capital_allocated": allocations[i],
                        "position_pct": allocations[i] / total_capital_needed,
                        "annual_yield": self.ticker_data[ticker]["annual_yield"],
                        "expected_dividend": allocations[i]
                        * self.ticker_data[ticker]["annual_yield"],
                    }
                )

        # Sort by position size
        results["portfolio"] = sorted(
            results["portfolio"], key=lambda x: x["capital_allocated"], reverse=True
        )

        return results

    def smooth_cashflow_seasonality(
        self,
        annual_dividend_target: float,
        budget_constraint: Optional[float] = None,
        max_position_pct: float = 0.15,
    ) -> Dict:
        """
        Solve: Minimize monthly cash flow variance while achieving dividend target.
        Uses improved scaling and robust solvers.

        Args:
            annual_dividend_target: Target annual dividend income
            budget_constraint: Maximum capital to deploy (if None, no constraint)
            max_position_pct: Maximum position size as fraction of portfolio

        Returns:
            Dictionary with optimization results
        """
        tickers = list(self.ticker_data.keys())
        n = len(tickers)

        # Work in thousands of USD for better scaling
        scale = 1000
        x_scaled = cp.Variable(n, nonneg=True)
        target_scaled = annual_dividend_target / scale

        # Monthly cash flows (matrix is already normalized)
        monthly_cashflows = self.monthly_matrix.T @ x_scaled
        target_monthly = target_scaled / 12

        # Objective: minimize variance of monthly cash flows
        variance = cp.sum_squares(monthly_cashflows - target_monthly)
        objective = cp.Minimize(variance)

        # Constraints
        constraints = []

        # Annual dividend constraint (scaled)
        annual_yields = np.array(
            [self.ticker_data[ticker].get("annual_yield", 0.0) or 0.0 for ticker in tickers]
        )
        annual_yields = np.nan_to_num(annual_yields, nan=0.0, posinf=0.0, neginf=0.0)
        dividend_constraint = (
            cp.sum(cp.multiply(x_scaled, annual_yields)) >= target_scaled
        )
        constraints.append(dividend_constraint)

        # Budget constraint (scaled)
        if budget_constraint:
            constraints.append(cp.sum(x_scaled) <= budget_constraint / scale)

        # Position size constraints (scaled)
        total_capital_scaled = cp.sum(x_scaled)
        for i in range(n):
            constraints.append(x_scaled[i] <= max_position_pct * total_capital_scaled)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")

        # Check for solution
        if x_scaled.value is None:
            raise RuntimeError("Optimization failed: No solution found")
            
        # Extract results and convert back to dollars
        allocations = np.array(x_scaled.value, dtype=np.float64) * scale

        # Calculate actual monthly cash flows using the raw matrix
        monthly_flows = self.raw_monthly_matrix.T @ allocations

        total_capital = np.sum(allocations)

        results = {
            "status": problem.status,
            "total_capital_used": total_capital,
            "target_annual_dividend": annual_dividend_target,
            "actual_annual_dividend": np.sum(allocations * annual_yields),
            "monthly_cashflows": monthly_flows,
            "cashflow_variance": np.var(monthly_flows),
            "cashflow_std": np.std(monthly_flows),
        }

        return results

    def multi_objective_optimization(
        self,
        target_annual_dividend: float,
        lambda_capital: float = 0.5,
        lambda_smoothness: float = 0.5,
        max_position_pct: float = 0.12,
    ) -> Dict:
        """
        Combined optimization: balance capital minimization and cash flow smoothing.

        Args:
            target_annual_dividend: Target annual dividend
            lambda_capital: Weight for capital minimization (0-1)
            lambda_smoothness: Weight for smoothness (0-1)
            max_position_pct: Maximum position size
        """
        tickers = list(self.ticker_data.keys())
        n = len(tickers)

        # Scale for numerical stability
        scale = 1000
        x_scaled = cp.Variable(n, nonneg=True)
        target_scaled = target_annual_dividend / scale

        # Components of objective function
        total_capital = cp.sum(x_scaled)
        monthly_cashflows = self.monthly_matrix.T @ x_scaled
        target_monthly = target_scaled / 12
        smoothness_penalty = cp.sum_squares(monthly_cashflows - target_monthly)

        # Normalize objectives using typical ranges
        typical_capital = target_scaled * 15
        typical_variance = (target_monthly * 0.5) ** 2 * 12

        capital_normalized = total_capital / typical_capital
        smoothness_normalized = smoothness_penalty / typical_variance

        # Combined objective
        objective = cp.Minimize(
            lambda_capital * capital_normalized
            + lambda_smoothness * smoothness_normalized
        )

        # Constraints
        annual_yields = np.array(
            [self.ticker_data[ticker].get("annual_yield", 0.0) or 0.0 for ticker in tickers]
        )
        annual_yields = np.nan_to_num(annual_yields, nan=0.0, posinf=0.0, neginf=0.0)
        constraints = [
            cp.sum(cp.multiply(x_scaled, annual_yields)) >= target_scaled,
            *[x_scaled[i] <= max_position_pct * total_capital for i in range(n)],
        ]

        problem = cp.Problem(objective, constraints)

        # Try different solvers
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except (cp.SolverError, ArithmeticError, ValueError, TypeError):
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except (cp.SolverError, ArithmeticError, ValueError, TypeError):
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-3)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed: {problem.status}")

        # Check for solution and scale back results
        if x_scaled.value is None:
            raise RuntimeError("Optimization failed: No solution found")
            
        allocations = np.array(x_scaled.value, dtype=np.float64) * scale
        monthly_flows = self.raw_monthly_matrix.T @ allocations
        total_cap = np.sum(allocations)

        results = {
            "status": problem.status,
            "lambda_capital": lambda_capital,
            "lambda_smoothness": lambda_smoothness,
            "total_capital_used": total_cap,
            "actual_annual_dividend": np.sum(allocations * annual_yields),
            "monthly_cashflows": monthly_flows,
            "cashflow_std": np.std(monthly_flows),
            "portfolio": [],
        }

        for i, ticker in enumerate(tickers):
            if allocations[i] > 1:
                results["portfolio"].append(
                    {
                        "ticker": ticker,
                        "company_name": self.ticker_data[ticker]["company_name"],
                        "capital_allocated": allocations[i],
                        "position_pct": allocations[i] / total_cap,
                        "annual_yield": annual_yields[i],
                    }
                )

        results["portfolio"] = sorted(
            results["portfolio"], key=lambda x: x["capital_allocated"], reverse=True
        )

        return results

    def print_results(self, results: Dict, optimization_type: str):
        """Pretty print optimization results."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION RESULTS: {optimization_type.upper()}")
        print(f"{'='*60}")

        print(f"Status: {results['status']}")
        print(
            f"Total Capital: ${results.get('total_capital_needed', results.get('total_capital_used', 0)):,.2f}"
        )

        if "target_dividend" in results:
            print(f"Target Dividend: ${results['target_dividend']:,.2f}")
        if "target_annual_dividend" in results:
            print(f"Target Dividend: ${results['target_annual_dividend']:,.2f}")

        actual_div = results.get(
            "actual_dividend", results.get("actual_annual_dividend", 0)
        )
        print(f"Actual Dividend: ${actual_div:,.2f}")

        if "cashflow_std" in results:
            print(f"Monthly Cash Flow Std Dev: ${results['cashflow_std']:.2f}")
            print(
                f"Monthly Range: ${results.get('min_monthly', 0):.2f} - ${results.get('max_monthly', 0):.2f}"
            )

        print(f"\nPORTFOLIO COMPOSITION ({len(results['portfolio'])} positions):")
        print(
            f"{'Ticker':<8} {'Company':<25} {'Capital':<12} {'Weight':<8} {'Yield':<8}"
        )
        print("-" * 70)

        for pos in results["portfolio"][:15]:  # Show top 15 positions
            print(
                f"{pos['ticker']:<8} "
                f"{pos.get('company_name', 'N/A')[:24]:<25} "
                f"${pos['capital_allocated']:>10,.0f} "
                f"{pos['position_pct']*100:>6.1f}% "
                f"{pos['annual_yield']*100:>6.2f}%"
            )

        if "monthly_cashflows" in results:
            print("\nMONTHLY CASH FLOW PATTERN:")
            months = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            for i, month in enumerate(months):
                print(f"{month}: ${results['monthly_cashflows'][i]:>8.2f}", end="  ")
                if (i + 1) % 4 == 0:
                    print()  # New line every 4 months


def main():
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Dividend Portfolio Optimizer")
    parser.add_argument(
        "--target",
        type=float,
        default=10000,
        help="Target annual dividend income (default: $10,000)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum capital budget constraint (optional)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.15,
        help="Maximum position size as percentage (default: 0.15 = 15%)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="dividend_info.parquet",
        help="Path to dividend data parquet file",
    )

    args = parser.parse_args()

    print(f"Available CVXPY solvers: {', '.join(cp.installed_solvers())}")
    print(f"Target Annual Dividend: ${args.target:,.2f}")
    if args.budget:
        print(f"Budget Constraint: ${args.budget:,.2f}")
    print(f"Max Position Size: {args.max_position*100:.1f}%")
    print("=" * 60)

    # Initialize optimizer
    try:
        optimizer = DividendPortfolioOptimizer(args.data_file)
    except FileNotFoundError:
        print(f"Error: Could not find data file '{args.data_file}'")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Example 1: Minimize capital for target dividend
    print(f"\nEXAMPLE 1: Minimize Capital for ${args.target:,.0f} Annual Dividend")
    try:
        results1 = optimizer.minimize_capital_for_target_yield(
            target_annual_dividend=args.target, max_position_pct=args.max_position
        )
        optimizer.print_results(results1, "Capital Minimization")
    except Exception as e:
        print(f"Error in capital minimization: {e}")

    # Example 2: Smooth cash flow seasonality
    print(f"\n\nEXAMPLE 2: Smooth Cash Flow for ${args.target:,.0f} Annual Dividend")
    try:
        budget = (
            args.budget if args.budget else args.target * 20
        )  # Default 20x dividend as budget
        results2 = optimizer.smooth_cashflow_seasonality(
            annual_dividend_target=args.target,
            budget_constraint=budget,
            max_position_pct=args.max_position,
        )
        optimizer.print_results(results2, "Cash Flow Smoothing")
    except Exception as e:
        print(f"Error in cash flow smoothing: {e}")

    # Example 3: Multi-objective optimization
    print("\n\nEXAMPLE 3: Multi-Objective Optimization (Balanced)")
    try:
        results3 = optimizer.multi_objective_optimization(
            target_annual_dividend=args.target,
            lambda_capital=0.3,  # weight on capital minimization
            lambda_smoothness=0.7,  # weight on smoothness
            max_position_pct=args.max_position,
        )
        optimizer.print_results(results3, "Multi-Objective Balance")
    except Exception as e:
        print(f"Error in multi-objective optimization: {e}")

    # Generate efficient frontier
    print(f"\n\nEFFICIENT FRONTIER ANALYSIS (Target: ${args.target:,.0f}):")
    print("Capital vs Smoothness Trade-offs")
    print(
        f"{'λ_cap':<6} {'λ_smooth':<8} {'Capital':<12} {'Std Dev':<10} {'Positions':<10}"
    )
    print("-" * 50)

    for lambda_cap in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lambda_smooth = 1.0 - lambda_cap
        try:
            result = optimizer.multi_objective_optimization(
                target_annual_dividend=args.target,
                lambda_capital=lambda_cap,
                lambda_smoothness=lambda_smooth,
                max_position_pct=args.max_position,
            )
            print(
                f"{lambda_cap:<6.1f} {lambda_smooth:<8.1f} "
                f"${result['total_capital_used']:>10,.0f} "
                f"${result['cashflow_std']:>8.2f} "
                f"{len(result['portfolio']):>8}"
            )
        except (KeyError, TypeError, IndexError):
            print(f"{lambda_cap:<6.1f} {lambda_smooth:<8.1f} {'FAILED':<12}")


if __name__ == "__main__":
    main()
