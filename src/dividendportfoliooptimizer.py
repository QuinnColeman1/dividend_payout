"""Dividend portfolio optimization engine.

Builds per-ticker yield data and a monthly dividend pattern from a table of
dividend events, then solves convex optimization problems with CVXPY:

* minimize the capital needed to reach a target annual dividend,
* smooth monthly dividend cash flow around the target,
* a weighted blend of the two (used for the efficient frontier).
"""

import argparse
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Weights for the most recent, second and third year when building monthly patterns.
YEAR_WEIGHTS = (0.5, 0.33, 0.17)

# Decision variables are expressed in thousands of dollars for numerical stability.
SCALE = 1000.0

# Allocations below this many dollars are treated as solver noise in the portfolio list.
MIN_POSITION_DOLLARS = 1.0

SOLVER_ORDER = ("CLARABEL", "ECOS", "SCS")


def _to_naive_datetime(values: pd.Series) -> pd.Series:
    """Parse dates and drop any timezone so tickers from different exchanges compare cleanly."""
    parsed = pd.to_datetime(values, utc=True)
    return parsed.dt.tz_localize(None)


class DividendPortfolioOptimizer:
    def __init__(self, dividend_df: pd.DataFrame, as_of: Optional[Any] = None):
        """
        Initialize the optimizer with dividend data.

        Args:
            dividend_df: DataFrame of dividend events with columns:
                - ticker: str
                - ex_dividend_date: datetime
                - dividend_amount: float (per share)
                - close_price: float (share price on the ex-dividend date)
                - current_price: float (optional, latest share price; preferred for yields)
                - company_name, sector, currency: str (optional)
            as_of: Reference date for trailing-twelve-month yields and monthly patterns.
                Defaults to the most recent ex-dividend date in the data.
        """
        if not isinstance(dividend_df, pd.DataFrame):
            raise TypeError(
                f"dividend_df must be a pandas DataFrame, got {type(dividend_df).__name__}"
            )

        required_columns = ["ticker", "ex_dividend_date", "dividend_amount", "close_price"]
        missing_columns = [col for col in required_columns if col not in dividend_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in input DataFrame: {missing_columns}")

        df = dividend_df.copy()
        df["ex_dividend_date"] = _to_naive_datetime(df["ex_dividend_date"])
        df["dividend_amount"] = pd.to_numeric(df["dividend_amount"], errors="coerce")
        df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
        df = df[df["dividend_amount"].notna() & (df["dividend_amount"] > 0)]
        df = df.sort_values(["ticker", "ex_dividend_date"]).reset_index(drop=True)
        df["month"] = df["ex_dividend_date"].dt.month

        if "company_name" not in df.columns:
            df["company_name"] = df["ticker"]
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
        if "currency" not in df.columns:
            df["currency"] = "USD"

        self.df = df
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            self.as_of = as_of_ts.tz_convert(None) if as_of_ts.tzinfo else as_of_ts
        elif not df.empty:
            self.as_of = df["ex_dividend_date"].max()
        else:
            self.as_of = pd.Timestamp.now().normalize()

        self.ticker_data = self._prepare_ticker_data()
        self.tickers: List[str] = list(self.ticker_data.keys())
        self.monthly_matrix = self._build_monthly_dividend_matrix()
        # Kept for backwards compatibility: the matrix is no longer normalized.
        self.raw_monthly_matrix = self.monthly_matrix

        print(f"Loaded data for {len(self.ticker_data)} tickers")
        if not df.empty:
            print(f"Date range: {df['ex_dividend_date'].min():%Y-%m-%d} to {df['ex_dividend_date'].max():%Y-%m-%d}")
        else:
            print("Warning: No dividend data found")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _latest_valid(series: pd.Series) -> float:
        valid = series[series.notna() & (series > 0)]
        return float(valid.iloc[-1]) if not valid.empty else 0.0

    def _prepare_ticker_data(self) -> Dict[str, Dict[str, Any]]:
        """Compute current price, trailing-twelve-month dividends and yield per ticker."""
        ticker_data: Dict[str, Dict[str, Any]] = {}
        if self.df.empty:
            return ticker_data

        ttm_start = self.as_of - timedelta(days=365)
        has_current_price = "current_price" in self.df.columns

        for ticker, ticker_df in self.df.groupby("ticker", sort=False):
            recent = ticker_df.iloc[-1]

            price = 0.0
            if has_current_price:
                price = self._latest_valid(pd.to_numeric(ticker_df["current_price"], errors="coerce"))
            if price <= 0:
                price = self._latest_valid(ticker_df["close_price"])

            in_ttm = (ticker_df["ex_dividend_date"] > ttm_start) & (ticker_df["ex_dividend_date"] <= self.as_of)
            ttm_dividends = float(ticker_df.loc[in_ttm, "dividend_amount"].sum())
            annual_yield = ttm_dividends / price if price > 0 else 0.0

            ticker_data[str(ticker)] = {
                "company_name": recent.get("company_name", ticker),
                "sector": recent.get("sector", "Unknown"),
                "currency": recent.get("currency", "USD"),
                "current_price": price,
                "ttm_dividend": ttm_dividends,
                "annual_yield": annual_yield,  # fraction, e.g. 0.035 = 3.5%
                "dividend_yield": annual_yield * 100,  # percent, for display
            }
        return ticker_data

    def _build_monthly_dividend_matrix(self) -> np.ndarray:
        """
        Build a (tickers x 12) matrix of expected dividends per dollar invested in each month.

        Each of the last three years is summed by calendar month and blended with
        YEAR_WEIGHTS. Years before a ticker's first recorded dividend are skipped
        (and the weights renormalized), so newer dividend payers are not penalized.
        """
        matrix = np.zeros((len(self.tickers), 12))

        for i, ticker in enumerate(self.tickers):
            price = self.ticker_data[ticker]["current_price"]
            if price <= 0:
                continue

            ticker_df = self.df[self.df["ticker"] == ticker]
            first_date = ticker_df["ex_dividend_date"].min()

            month_totals = np.zeros(12)
            total_weight = 0.0
            for years_back, weight in enumerate(YEAR_WEIGHTS):
                window_end = self.as_of - timedelta(days=365 * years_back)
                window_start = window_end - timedelta(days=365)
                if years_back > 0 and first_date > window_end:
                    continue  # no history that far back

                in_window = (ticker_df["ex_dividend_date"] > window_start) & (
                    ticker_df["ex_dividend_date"] <= window_end
                )
                by_month = ticker_df.loc[in_window].groupby("month")["dividend_amount"].sum()
                for month, amount in by_month.items():
                    month_totals[int(month) - 1] += weight * float(amount)
                total_weight += weight

            if total_weight > 0:
                matrix[i, :] = month_totals / total_weight / price

        return matrix

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(target_annual_dividend: float, max_position_pct: float) -> None:
        if not target_annual_dividend or target_annual_dividend <= 0:
            raise ValueError("target_annual_dividend must be greater than 0")
        if not 0 < max_position_pct <= 1:
            raise ValueError(f"max_position_pct must be between 0 and 1, got {max_position_pct}")

    def _eligible_tickers(self, min_yield_threshold: float, max_position_pct: float) -> List[str]:
        """Tickers with a positive yield at or above the threshold, checked for feasibility."""
        eligible = [
            ticker
            for ticker in self.tickers
            if self.ticker_data[ticker]["annual_yield"] > 0
            and self.ticker_data[ticker]["annual_yield"] >= min_yield_threshold
        ]
        if not eligible:
            raise ValueError("No tickers meet the minimum yield threshold")

        # Each position is capped at max_position_pct of the total, so at least
        # ceil(1 / max_position_pct) positions are needed for a feasible portfolio.
        needed = math.ceil(1 / max_position_pct - 1e-9)
        if len(eligible) < needed:
            raise ValueError(
                f"A {max_position_pct:.0%} maximum position size needs at least {needed} eligible tickers, "
                f"but only {len(eligible)} meet the criteria. Add tickers, lower the minimum yield, "
                "or raise the maximum position size."
            )
        return eligible

    def _base_constraints(
        self,
        x: cp.Variable,
        yields: np.ndarray,
        target_annual_dividend: float,
        max_position_pct: float,
        budget_constraint: Optional[float],
    ) -> List[Any]:
        total = cp.sum(x)
        constraints = [
            yields @ x >= target_annual_dividend / SCALE,
            x <= max_position_pct * total,
        ]
        if budget_constraint is not None and budget_constraint > 0:
            constraints.append(total <= budget_constraint / SCALE)
        return constraints

    @staticmethod
    def _solve(problem: cp.Problem, variable: cp.Variable) -> None:
        """Solve with the first installed solver that succeeds; raise a clear error otherwise."""
        installed = set(cp.installed_solvers())
        last_status: Optional[str] = None

        for solver in SOLVER_ORDER:
            if solver not in installed:
                continue
            try:
                problem.solve(solver=solver, verbose=False)
            except (cp.SolverError, ArithmeticError, ValueError):
                continue
            last_status = problem.status
            if last_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and variable.value is not None:
                return
            if last_status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                raise RuntimeError(
                    "Optimization failed: infeasible. The budget is too small to reach the target "
                    "dividend with these tickers and position limits."
                )

        raise RuntimeError(f"Optimization failed: {last_status or 'no solver succeeded'}")

    def _build_results(
        self,
        tickers: List[str],
        allocations_scaled: Optional[np.ndarray],
        yields: np.ndarray,
        monthly: np.ndarray,
        status: str,
        target_annual_dividend: float,
    ) -> Dict[str, Any]:
        if allocations_scaled is None:
            raise RuntimeError("Optimization failed: No solution found")
        allocations =np.maximum(np.asarray(allocations_scaled, dtype=np.float64), 0.0) * SCALE
        total_capital = float(np.sum(allocations))
        monthly_flows = monthly.T @ allocations

        portfolio = []
        for i, ticker in enumerate(tickers):
            if allocations[i] > MIN_POSITION_DOLLARS:
                portfolio.append(
                    {
                        "ticker": ticker,
                        "company_name": self.ticker_data[ticker]["company_name"],
                        "sector": self.ticker_data[ticker]["sector"],
                        "capital_allocated": float(allocations[i]),
                        "position_pct": float(allocations[i] / total_capital) if total_capital > 0 else 0.0,
                        "annual_yield": float(yields[i]),
                        "expected_dividend": float(allocations[i] * yields[i]),
                    }
                )
        portfolio.sort(key=lambda pos: pos["capital_allocated"], reverse=True)

        return {
            "status": status,
            "total_capital_used": total_capital,
            "target_annual_dividend": float(target_annual_dividend),
            "actual_annual_dividend": float(np.sum(allocations * yields)),
            "monthly_cashflows": monthly_flows,
            "cashflow_std": float(np.std(monthly_flows)),
            "cashflow_variance": float(np.var(monthly_flows)),
            "min_monthly": float(np.min(monthly_flows)),
            "max_monthly": float(np.max(monthly_flows)),
            "portfolio": portfolio,
        }

    def _problem_data(self, tickers: List[str]) -> tuple:
        idx = [self.tickers.index(t) for t in tickers]
        yields = np.nan_to_num(
            np.array([self.ticker_data[t]["annual_yield"] for t in tickers], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return yields, self.monthly_matrix[idx, :]

    # ------------------------------------------------------------------
    # Optimizations
    # ------------------------------------------------------------------
    def minimize_capital_for_target_yield(
        self,
        target_annual_dividend: float,
        max_position_pct: float = 0.1,
        min_yield_threshold: float = 0.01,
        budget_constraint: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Minimize the total capital needed to reach a target annual dividend.

        Args:
            target_annual_dividend: Target annual dividend income ($)
            max_position_pct: Maximum position size as a fraction of the total (0.1 = 10%)
            min_yield_threshold: Minimum annual yield to consider a stock (0.01 = 1%)
            budget_constraint: Maximum capital to deploy ($); None or 0 for no limit
        """
        self._validate_inputs(target_annual_dividend, max_position_pct)
        tickers = self._eligible_tickers(min_yield_threshold, max_position_pct)
        yields, monthly = self._problem_data(tickers)

        x = cp.Variable(len(tickers), nonneg=True)
        constraints = self._base_constraints(x, yields, target_annual_dividend, max_position_pct, budget_constraint)
        problem = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
        self._solve(problem, x)

        return self._build_results(tickers, x.value, yields, monthly, problem.status, target_annual_dividend)

    def smooth_cashflow_seasonality(
        self,
        annual_dividend_target: float,
        budget_constraint: Optional[float] = None,
        max_position_pct: float = 0.15,
        min_yield_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Minimize the squared deviation of monthly dividend cash flow from target / 12.

        Args:
            annual_dividend_target: Target annual dividend income ($)
            budget_constraint: Maximum capital to deploy ($); None or 0 for no limit
            max_position_pct: Maximum position size as a fraction of the total
            min_yield_threshold: Minimum annual yield to consider a stock
        """
        self._validate_inputs(annual_dividend_target, max_position_pct)
        tickers = self._eligible_tickers(min_yield_threshold, max_position_pct)
        yields, monthly = self._problem_data(tickers)

        x = cp.Variable(len(tickers), nonneg=True)
        target_monthly = annual_dividend_target / SCALE / 12
        objective = cp.Minimize(cp.sum_squares(monthly.T @ x - target_monthly))
        constraints = self._base_constraints(x, yields, annual_dividend_target, max_position_pct, budget_constraint)
        problem = cp.Problem(objective, constraints)
        self._solve(problem, x)

        return self._build_results(tickers, x.value, yields, monthly, problem.status, annual_dividend_target)

    def multi_objective_optimization(
        self,
        target_annual_dividend: float,
        lambda_capital: float = 0.5,
        lambda_smoothness: float = 0.5,
        max_position_pct: float = 0.12,
        budget_constraint: Optional[float] = None,
        min_yield_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Balance capital minimization against cash-flow smoothness.

        Args:
            target_annual_dividend: Target annual dividend ($)
            lambda_capital: Weight for capital minimization (0-1)
            lambda_smoothness: Weight for smoothness (0-1)
            max_position_pct: Maximum position size as a fraction of the total
            budget_constraint: Maximum capital to deploy ($); None or 0 for no limit
            min_yield_threshold: Minimum annual yield to consider a stock
        """
        self._validate_inputs(target_annual_dividend, max_position_pct)
        for name, value in (("lambda_capital", lambda_capital), ("lambda_smoothness", lambda_smoothness)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if lambda_capital + lambda_smoothness <= 0:
            raise ValueError("lambda_capital and lambda_smoothness cannot both be 0")

        tickers = self._eligible_tickers(min_yield_threshold, max_position_pct)
        yields, monthly = self._problem_data(tickers)

        x = cp.Variable(len(tickers), nonneg=True)
        target_scaled = target_annual_dividend / SCALE
        target_monthly = target_scaled / 12

        # Normalize both terms to comparable magnitudes before weighting them.
        typical_capital = target_scaled * 15  # roughly the capital for a ~6.7% yield
        typical_variance = (target_monthly * 0.5) ** 2 * 12
        capital_term = cp.sum(x) / typical_capital
        smoothness_term = cp.sum_squares(monthly.T @ x - target_monthly) / typical_variance

        objective = cp.Minimize(lambda_capital * capital_term + lambda_smoothness * smoothness_term)
        constraints = self._base_constraints(x, yields, target_annual_dividend, max_position_pct, budget_constraint)
        problem = cp.Problem(objective, constraints)
        self._solve(problem, x)

        results = self._build_results(tickers, x.value, yields, monthly, problem.status, target_annual_dividend)
        results["lambda_capital"] = lambda_capital
        results["lambda_smoothness"] = lambda_smoothness
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_results(self, results: Dict[str, Any], optimization_type: str) -> None:
        """Pretty print optimization results."""
        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION RESULTS: {optimization_type.upper()}")
        print(f"{'=' * 60}")
        print(f"Status: {results['status']}")
        print(f"Total Capital: ${results['total_capital_used']:,.2f}")
        print(f"Target Dividend: ${results['target_annual_dividend']:,.2f}")
        print(f"Actual Dividend: ${results['actual_annual_dividend']:,.2f}")
        print(f"Monthly Cash Flow Std Dev: ${results['cashflow_std']:.2f}")
        print(f"Monthly Range: ${results['min_monthly']:.2f} - ${results['max_monthly']:.2f}")

        print(f"\nPORTFOLIO COMPOSITION ({len(results['portfolio'])} positions):")
        print(f"{'Ticker':<8} {'Company':<25} {'Capital':<12} {'Weight':<8} {'Yield':<8}")
        print("-" * 70)
        for pos in results["portfolio"][:15]:
            print(
                f"{pos['ticker']:<8} "
                f"{str(pos.get('company_name', 'N/A'))[:24]:<25} "
                f"${pos['capital_allocated']:>10,.0f} "
                f"{pos['position_pct'] * 100:>6.1f}% "
                f"{pos['annual_yield'] * 100:>6.2f}%"
            )

        print("\nMONTHLY CASH FLOW PATTERN:")
        for i, month in enumerate(MONTHS):
            print(f"{month}: ${results['monthly_cashflows'][i]:>8.2f}", end="  ")
            if (i + 1) % 4 == 0:
                print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Dividend Portfolio Optimizer")
    parser.add_argument("--target", type=float, default=10000, help="Target annual dividend income (default: $10,000)")
    parser.add_argument("--budget", type=float, default=None, help="Maximum capital budget constraint (optional)")
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.15,
        help="Maximum position size as a fraction (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="dividend_info.parquet",
        help="Path to the dividend data parquet file created by dividend_payout.py",
    )
    args = parser.parse_args(argv)

    print(f"Available CVXPY solvers: {', '.join(cp.installed_solvers())}")
    print(f"Target Annual Dividend: ${args.target:,.2f}")
    if args.budget:
        print(f"Budget Constraint: ${args.budget:,.2f}")
    print(f"Max Position Size: {args.max_position * 100:.1f}%")
    print("=" * 60)

    try:
        optimizer = DividendPortfolioOptimizer(pd.read_parquet(args.data_file))
    except FileNotFoundError:
        print(f"Error: Could not find data file '{args.data_file}'. Run dividend_payout.py first to create it.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    examples = [
        (
            f"EXAMPLE 1: Minimize Capital for ${args.target:,.0f} Annual Dividend",
            "Capital Minimization",
            lambda: optimizer.minimize_capital_for_target_yield(
                target_annual_dividend=args.target,
                max_position_pct=args.max_position,
                budget_constraint=args.budget,
            ),
        ),
        (
            f"EXAMPLE 2: Smooth Cash Flow for ${args.target:,.0f} Annual Dividend",
            "Cash Flow Smoothing",
            lambda: optimizer.smooth_cashflow_seasonality(
                annual_dividend_target=args.target,
                budget_constraint=args.budget,
                max_position_pct=args.max_position,
            ),
        ),
        (
            "EXAMPLE 3: Multi-Objective Optimization (Balanced)",
            "Multi-Objective Balance",
            lambda: optimizer.multi_objective_optimization(
                target_annual_dividend=args.target,
                lambda_capital=0.3,
                lambda_smoothness=0.7,
                max_position_pct=args.max_position,
                budget_constraint=args.budget,
            ),
        ),
    ]
    for heading, label, run in examples:
        print(f"\n\n{heading}")
        try:
            optimizer.print_results(run(), label)
        except (RuntimeError, ValueError) as e:
            print(f"Error in {label.lower()}: {e}")

    print(f"\n\nEFFICIENT FRONTIER ANALYSIS (Target: ${args.target:,.0f}):")
    print("Capital vs Smoothness Trade-offs")
    print(f"{'λ_cap':<6} {'λ_smooth':<8} {'Capital':<12} {'Std Dev':<10} {'Positions':<10}")
    print("-" * 50)
    for lambda_cap in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lambda_smooth = round(1.0 - lambda_cap, 2)
        try:
            result = optimizer.multi_objective_optimization(
                target_annual_dividend=args.target,
                lambda_capital=lambda_cap,
                lambda_smoothness=lambda_smooth,
                max_position_pct=args.max_position,
                budget_constraint=args.budget,
            )
            print(
                f"{lambda_cap:<6.1f} {lambda_smooth:<8.1f} "
                f"${result['total_capital_used']:>10,.0f} "
                f"${result['cashflow_std']:>8.2f} "
                f"{len(result['portfolio']):>8}"
            )
        except (RuntimeError, ValueError):
            print(f"{lambda_cap:<6.1f} {lambda_smooth:<8.1f} {'FAILED':<12}")


if __name__ == "__main__":
    main()
