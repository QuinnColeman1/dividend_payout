from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.optimizer_wrapper import create_optimizer_wrapper

# ==============================
# Streamlit Page Configuration
# ==============================
st.set_page_config(
    page_title="Dividend Portfolio Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# Session State Initialization
# ==============================

# Initialize session state with type hints
default_session_items: Dict[str, Any] = {
    "optimizer": None,  # Will hold OptimizerWrapper instance
    "results": {},      # Will hold optimization results
    "frontier_data": None,  # Will hold efficient frontier data
    "last_scrape_timestamp": None,  # Will hold timestamp of last data refresh
    "last_error": None,  # Will hold the last error message
    "tickers_data": None,  # Will hold the tickers data dictionary
}
for k, v in default_session_items.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================
# Data Loading & Processing
# ==============================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_data(tickers: list, years: int = 5) -> dict:
    """
    Fetch dividend data for given tickers using yfinance.
    
    Args:
        tickers: List of stock ticker symbols
        years: Number of years of historical data to fetch
        
    Returns:
        Dictionary with tickers as keys and their data as values
    """
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    # Filter out empty or invalid tickers
    tickers = [t for t in tickers if t and isinstance(t, str) and t.strip()]
    
    if not tickers:
        return {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Fetching data for {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(start=start_date, end=end_date, interval="1mo")
            
            if hist.empty:
                st.warning(f"No data found for {ticker}")
                continue
                
            # Get company info
            info = {}
            try:
                info = stock.info
            except Exception as e:
                st.warning(f"Could not fetch info for {ticker}: {str(e)}")
            
            # Store the data
            data[ticker] = {
                'dividends': hist['Dividends'] if 'Dividends' in hist.columns else pd.Series(dtype=float),
                'close': hist['Close'],
                'info': info  # Store all info in a nested dict
            }
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
    
    status_text.empty()
    progress_bar.empty()
    
    if not data:
        st.error("No valid ticker data could be loaded. Please check your ticker symbols and try again.")
    
    return data


def init_optimizer(tickers: list):
    """
    Initialize the optimizer with data from yfinance.
    
    Args:
        tickers: List of stock ticker symbols to include in the optimization
    """
    if not tickers:
        st.error("Please enter at least one valid ticker symbol.")
        return
        
    # Clear any previous error
    if 'last_error' in st.session_state:
        st.session_state.last_error = None
    
    try:
        with st.spinner("Fetching market data..."):
            # Always fetch fresh data when initializing
            st.session_state.tickers_data = fetch_dividend_data(tickers)
            
            if not st.session_state.tickers_data:
                st.error("No valid ticker data available. Cannot initialize optimizer.")
                return
                
            st.session_state.optimizer = create_optimizer_wrapper(st.session_state.tickers_data)
            
            # Clear any previous errors on successful initialization
            if 'last_error' in st.session_state:
                del st.session_state.last_error
                
    except Exception as e:
        error_msg = f"Failed to initialize optimizer: {str(e)}"
        st.error(error_msg)
        st.session_state.last_error = error_msg


def normalize_result(raw: dict) -> dict:
    """
    Ensure result dict has consistent key names used by the UI.
    """
    if raw is None:
        return {}

    total_capital = raw.get(
        "total_capital_used",
        raw.get("total_capital_needed", raw.get("capital_required", 0.0)),
    )
    actual_div = raw.get(
        "actual_annual_dividend",
        raw.get("actual_dividend", raw.get("achieved_dividend", 0.0)),
    )
    monthly_cf = raw.get(
        "monthly_cashflows",
        raw.get("monthly_cash_flows", raw.get("monthly_dividends", [])),
    )

    if monthly_cf is not None and len(monthly_cf) > 0:
        if isinstance(monthly_cf, np.ndarray):
            monthly_cf = monthly_cf.tolist()
        min_month = raw.get("min_monthly", min(monthly_cf))
        max_month = raw.get("max_monthly", max(monthly_cf))
        std_cf = raw.get("cashflow_std", float(np.std(monthly_cf, ddof=0)))
    else:
        min_month = raw.get("min_monthly", 0.0)
        max_month = raw.get("max_monthly", 0.0)
        std_cf = raw.get("cashflow_std", 0.0)

    portfolio = raw.get("portfolio", [])

    normalized = {
        "status": raw.get("status", "unknown"),
        "total_capital_used": total_capital,
        "actual_annual_dividend": actual_div,
        "cashflow_std": std_cf,
        "monthly_cashflows": monthly_cf,
        "min_monthly": min_month,
        "max_monthly": max_month,
        "portfolio": portfolio,
    }
    return normalized


def record_result(key: str, result: dict):
    st.session_state.results[key] = normalize_result(result)


def get_latest_result():
    if not st.session_state.results:
        return None, None
    key = list(st.session_state.results.keys())[-1]
    return key, st.session_state.results[key]


def run_scraper():
    """
    Refresh market data by clearing the cache and forcing a refetch.
    
    Returns:
        Tuple of (success: bool, message: str, error: str)
    """
    try:
        # Clear the cached data
        if 'tickers_data' in st.session_state:
            st.session_state.tickers_data = None
            
        # Clear the optimizer to force reinitialization
        if 'optimizer' in st.session_state:
            st.session_state.optimizer = None
            
        # Clear any cached data
        st.cache_data.clear()
        
        return True, "Market data refresh initiated. Please wait while we fetch the latest data...", ""
        
    except Exception as e:
        return False, "", f"Error refreshing market data: {str(e)}"


def friendly_status(status: str) -> str:
    return status.upper()


# ==============================
# Default Tickers
# ==============================
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'KO',   # Blue chips
    'T', 'VZ', 'O', 'PEP', 'JPM',        # Dividend stocks
    'SPY', 'VYM', 'SCHD', 'VIG'          # ETFs
]

# ==============================
# Main App Layout
# ==============================
st.sidebar.title("Portfolio Navigator")

# Ticker input
with st.sidebar.expander("Ticker Selection", expanded=True):
    ticker_input = st.text_area(
        "Enter tickers (one per line)",
        value="\n".join(DEFAULT_TICKERS),
        height=150
    )
    
    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    
    # Initialize optimizer when button is clicked
    if st.button("Initialize Optimizer"):
        init_optimizer(tickers)
        
    # Refresh data button
    if st.button(" Refresh Market Data"):
        success, msg, error = run_scraper()
        if success:
            st.success(msg)
            # Reinitialize with current tickers
            init_optimizer(tickers)
        else:
            st.error(f"Error: {error}")

# Display status
if 'tickers_data' in st.session_state and st.session_state.tickers_data:
    st.sidebar.success(f"Loaded data for {len(st.session_state.tickers_data)} tickers")
    
    # Display ticker info
    with st.sidebar.expander("Ticker Info", expanded=False):
        for ticker, data in st.session_state.tickers_data.items():
            st.write(f"**{ticker}**: {data.get('info', {}).get('shortName', 'N/A')}")

# Display error if any
if 'last_error' in st.session_state and st.session_state.last_error:
    st.sidebar.error(st.session_state.last_error)
page = st.sidebar.radio(
    "Go to", ["Input Parameters", "Optimization Results", "Efficient Frontier"]
)

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.info(
    """
**Dividend Portfolio Optimizer**

This app helps you build optimal dividend portfolios from S&P 500 stocks:
- Minimize capital requirements
- Smooth monthly cash flows
- Multi-objective optimization
- Efficient frontier analysis

Built with **Streamlit**, **CVXPY**, and **yfinance**.
"""
)

# ==============================
# PAGE 1: Input Parameters
# ==============================
if page == "Input Parameters":
    st.title("💰 Dividend Portfolio Optimizer")
    st.markdown("### Configure Your Portfolio Optimization Parameters")

    # --- Data Status ---
    st.info("ℹ️ Using yfinance for real-time market data")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        target_dividend = st.number_input(
            "Target Annual Dividend ($)",
            min_value=1_000,
            max_value=2_000_000,
            value=10_000,
            step=500,
            help="Annual dividend income target",
        )
        budget_input = st.number_input(
                "Budget Constraint ($)",
                min_value=0,
                max_value=50_000_000,
                value=0,
                step=25_000,
                help="Maximum investment amount (0 = no limit)",
            )
        budget_constraint: Optional[float] = float(budget_input) if budget_input != 0 else None

    with c2:
        max_position_pct = (
            st.slider(
                "Maximum Position Size (%)",
                min_value=1,
                max_value=50,
                value=15,
                step=1,
                help="Largest percentage any single stock can be",
            )
            / 100
        )

        min_yield_threshold = (
            st.slider(
                "Minimum Yield Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Lowest dividend yield to consider",
            )
            / 100
        )

    with c3:
        optimization_type = st.selectbox(
            "Optimization Type",
            ["Multi-Objective Balance", "Minimize Capital", "Smooth Cash Flow"],
            help="Choose your optimization strategy",
        )

        if optimization_type == "Multi-Objective Balance":
            lambda_capital = st.slider(
                "Capital Weight (λ)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Balance between capital minimization and smoothness",
            )
            lambda_smoothness = 1.0 - lambda_capital
        else:
            lambda_capital = 0.5
            lambda_smoothness = 0.5

    st.divider()

    # --- Run Optimization ---
    center_run = st.container()
    with center_run:
        run_col = st.columns([1, 2, 1])[1]
        with run_col:
            if st.button("🚀 Run Optimization", type="primary", width="stretch"):
                if st.session_state.optimizer is None:
                    st.error("Please initialize the optimizer first!")
                else:
                    with st.spinner("Running optimization..."):
                        try:
                            # Get the list of tickers from the session state
                            if 'tickers' in st.session_state and st.session_state.tickers:
                                init_optimizer(st.session_state.tickers)
                            else:
                                st.error("No tickers available. Please add tickers first.")
                                st.stop()

                            # Run optimization
                            if optimization_type == "Minimize Capital":
                                raw = st.session_state.optimizer.minimize_capital(
                                    target_annual_dividend=target_dividend,
                                    max_position_pct=max_position_pct,
                                    min_yield_threshold=min_yield_threshold,
                                    budget_constraint=budget_constraint,
                                )
                                record_result("minimize_capital", raw)

                            elif optimization_type == "Smooth Cash Flow":
                                raw = st.session_state.optimizer.smooth_cashflow(
                                    target_annual_dividend=target_dividend,
                                    budget_constraint=budget_constraint,
                                    max_position_pct=max_position_pct,
                                )
                                record_result("smooth_cashflow", raw)

                            else:  # Multi-objective
                                raw = st.session_state.optimizer.multi_objective(
                                    target_annual_dividend=target_dividend,
                                    lambda_capital=lambda_capital,
                                    lambda_smoothness=lambda_smoothness,
                                    max_position_pct=max_position_pct,
                                    budget_constraint=budget_constraint,
                                )
                                record_result("multi_objective", raw)

                            st.success("✅ Optimization completed!")
                            st.info(
                                "Navigate to **Optimization Results** tab to view your portfolio."
                            )

                        except Exception as ex:
                            error_msg = f"Optimization failed: {str(ex)}"
                            st.error(error_msg)
                            st.session_state.last_error = error_msg

    # --- Settings Summary Table ---
    st.divider()
    st.markdown("### Current Settings Summary")
    settings_rows = [
        ("Target Annual Dividend", f"${target_dividend:,.0f}"),
        (
            "Budget Constraint",
            f"${budget_constraint:,.0f}" if budget_constraint else "None",
        ),
        ("Max Position Size", f"{max_position_pct*100:.1f}%"),
        ("Min Yield Threshold", f"{min_yield_threshold*100:.2f}%"),
        ("Optimization Type", optimization_type),
    ]
    if optimization_type == "Multi-Objective Balance":
        settings_rows.append(("Capital Weight (λ)", f"{lambda_capital:.2f}"))
        settings_rows.append(("Smoothness Weight (1-λ)", f"{lambda_smoothness:.2f}"))

    settings_df = pd.DataFrame(settings_rows, columns=["Parameter", "Value"])
    st.dataframe(settings_df, width="stretch", hide_index=True)

# ==============================
# PAGE 2: Optimization Results
# ==============================
elif page == "Optimization Results":
    st.title("📊 Optimization Results")

    result_key, results = get_latest_result()
    if results is None:
        st.warning(
            "No optimization results available. Please run an optimization first."
        )
    else:
        opt_type_names = {
            "minimize_capital": "CAPITAL MINIMIZATION",
            "smooth_cashflow": "CASH FLOW SMOOTHING",
            "multi_objective": "MULTI-OBJECTIVE BALANCE",
        }
        st.markdown(f"### {opt_type_names.get(result_key, result_key.upper())}")

        # --- Key Metrics ---
        met1, met2, met3, met4 = st.columns(4)
        with met1:
            st.metric("Status", friendly_status(results["status"]))
        with met2:
            st.metric("Total Capital", f"${results['total_capital_used']:,.0f}")
        with met3:
            st.metric("Actual Dividend", f"${results['actual_annual_dividend']:,.0f}")
        with met4:
            st.metric("Cash Flow Std Dev", f"${results['cashflow_std']:.2f}")

        if results["monthly_cashflows"]:
            st.info(
                f"Monthly Range: ${results['min_monthly']:.2f} - ${results['max_monthly']:.2f}"
            )

        st.divider()

        # --- Portfolio Table ---
        portfolio = results["portfolio"]
        if not portfolio:
            st.warning("No positions returned by optimizer.")
        else:
            st.markdown(f"### Portfolio Composition ({len(portfolio)} positions)")

            # Basic portfolio table
            port_df = pd.DataFrame(portfolio)
            for col in [
                "ticker",
                "company_name",
                "position_pct",
                "annual_yield",
                "capital_allocated",
            ]:
                if col not in port_df.columns:
                    port_df[col] = np.nan

            port_df["Weight (%)"] = port_df["position_pct"] * 100
            port_df["Yield (%)"] = port_df["annual_yield"] * 100
            port_df["Capital"] = port_df["capital_allocated"]

            display_df = (
                port_df.sort_values("Weight (%)", ascending=False)
                .head(15)
                .loc[
                    :, ["ticker", "company_name", "Capital", "Weight (%)", "Yield (%)"]
                ]
                .copy()
            )
            display_df["Capital"] = display_df["Capital"].apply(lambda v: f"${v:,.0f}")
            display_df["Weight (%)"] = display_df["Weight (%)"].apply(
                lambda v: f"{v:.1f}%"
            )
            display_df["Yield (%)"] = display_df["Yield (%)"].apply(
                lambda v: f"{v:.2f}%"
            )

            st.dataframe(display_df, hide_index=True, width="stretch")

            # --- Visualizations ---
            viz_row = st.columns(2)

            # Portfolio allocation pie chart
            with viz_row[0]:
                top10 = port_df.sort_values("position_pct", ascending=False).head(10)
                fig_pie = px.pie(
                    top10,
                    values="position_pct",
                    names="ticker",
                    title="Portfolio Allocation (Top 10)",
                    hole=0.35,
                )
                st.plotly_chart(
                    fig_pie, width="stretch", config={"displayModeBar": True}
                )

            # Yield bar chart
            with viz_row[1]:
                fig_bar = px.bar(
                    port_df.sort_values("annual_yield", ascending=False).head(15),
                    x="ticker",
                    y="annual_yield",
                    title="Dividend Yields by Position",
                )
                fig_bar.update_layout(yaxis=dict(tickformat=".1%"))
                st.plotly_chart(
                    fig_bar, width="stretch", config={"displayModeBar": True}
                )

        # --- Monthly Cash Flow Pattern ---
        if results["monthly_cashflows"]:
            st.divider()
            st.markdown("### Monthly Cash Flow Pattern")
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
            monthly_cf = results["monthly_cashflows"][:12]
            monthly_df = pd.DataFrame({"Month": months, "Cash Flow": monthly_cf})

            mc1, mc2 = st.columns([1, 2])
            with mc1:
                tab_df = monthly_df.copy()
                tab_df["Cash Flow"] = tab_df["Cash Flow"].apply(lambda v: f"${v:,.2f}")
                st.dataframe(tab_df, hide_index=True, width="stretch")

            with mc2:
                avg_month = float(np.mean(monthly_cf))
                fig_line = go.Figure()
                fig_line.add_trace(
                    go.Scatter(
                        x=months,
                        y=monthly_cf,
                        mode="lines+markers",
                        name="Monthly Cash Flow",
                        line=dict(width=3),
                        marker=dict(size=8),
                    )
                )
                fig_line.add_hline(
                    y=avg_month,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: ${avg_month:,.2f}",
                    annotation_position="top left",
                )
                fig_line.update_layout(
                    title="Monthly Dividend Cash Flows",
                    xaxis_title="Month",
                    yaxis_title="Cash Flow ($)",
                    hovermode="x unified",
                )
                st.plotly_chart(
                    fig_line, width="stretch", config={"displayModeBar": True}
                )

# ==============================
# PAGE 3: Efficient Frontier
# ==============================
elif page == "Efficient Frontier":
    st.title("📈 Efficient Frontier Analysis")

    if st.session_state.optimizer is None:
        st.warning("Please initialize the optimizer first!")
    else:
        ef_c1, ef_c2, ef_c3 = st.columns(3)
        with ef_c1:
            ef_target = st.number_input(
                "Target Annual Dividend for Analysis ($)",
                min_value=1000,
                max_value=2_000_000,
                value=15_000,
                step=500,
                help="Annual dividend income to analyze for trade-offs",
            )
        with ef_c2:
            ef_max_position = (
                st.slider(
                    "Max Position Size for Analysis (%)",
                    min_value=1,
                    max_value=50,
                    value=15,
                    help="Maximum allocation to any single stock",
                )
                / 100
            )
        with ef_c3:
            lambda_values_input = st.text_input(
                "Lambda (capital weights) list",
                value="0.1,0.3,0.5,0.7,0.9",
                help="Capital weight values to test (0-1)",
            )

        go_btn_col = st.columns([3, 1, 3])[1]
        with go_btn_col:
            if st.button(
                "🔄 Generate Efficient Frontier", type="primary", width="stretch"
            ):
                with st.spinner("Generating efficient frontier..."):
                    try:
                        lambda_values = []
                        for raw_val in lambda_values_input.split(","):
                            val = raw_val.strip()
                            if val:
                                f = float(val)
                                if 0 <= f <= 1:
                                    lambda_values.append(round(f, 4))
                        if not lambda_values:
                            st.error("No valid lambda values provided.")
                        else:
                            frontier_results = (
                                st.session_state.optimizer.generate_efficient_frontier(
                                    target_annual_dividend=ef_target,
                                    lambda_values=lambda_values,
                                    max_position_pct=ef_max_position,
                                )
                            )

                            frontier_data = []
                            for result in frontier_results:
                                if result["status"] == "success":
                                    frontier_data.append(
                                        {
                                            "λ_cap": result["lambda_capital"],
                                            "λ_smooth": result["lambda_smoothness"],
                                            "Capital": result["total_capital_used"],
                                            "Std Dev": result["cashflow_std"],
                                            "Positions": result["portfolio_size"],
                                            "Status": "optimal",
                                        }
                                    )
                                else:
                                    frontier_data.append(
                                        {
                                            "λ_cap": result["lambda_capital"],
                                            "λ_smooth": result["lambda_smoothness"],
                                            "Capital": None,
                                            "Std Dev": None,
                                            "Positions": 0,
                                            "Status": f"error: {result.get('error', 'unknown')}",
                                        }
                                    )

                            st.session_state.frontier_data = frontier_data
                            st.success("Efficient frontier generation complete.")
                    except Exception:
                        st.error("Failed to generate frontier. Please check your input parameters and try again.")
                        st.session_state.frontier_data = None

        # Display results
        if st.session_state.frontier_data:
            st.divider()
            st.markdown(
                f"### Capital vs Smoothness Trade-offs (Target: ${ef_target:,.0f})"
            )

            frontier_df = pd.DataFrame(st.session_state.frontier_data)
            show_df = frontier_df.copy()
            show_df["Capital"] = show_df["Capital"].apply(
                lambda v: f"${v:,.0f}" if pd.notnull(v) else "FAILED"
            )
            show_df["Std Dev"] = show_df["Std Dev"].apply(
                lambda v: f"${v:,.2f}" if pd.notnull(v) else "N/A"
            )
            st.dataframe(show_df, hide_index=True, width="stretch")

            valid_df = frontier_df.dropna(subset=["Capital", "Std Dev"])
            if not valid_df.empty:
                vc1, vc2 = st.columns(2)
                with vc1:
                    fig_cap = go.Figure()
                    fig_cap.add_trace(
                        go.Scatter(
                            x=valid_df["λ_cap"],
                            y=valid_df["Capital"],
                            mode="lines+markers",
                            line=dict(width=3),
                            marker=dict(size=10),
                            name="Capital",
                        )
                    )
                    fig_cap.update_layout(
                        title="Capital Required vs Capital Weight (λ)",
                        xaxis_title="Capital Weight (λ)",
                        yaxis_title="Total Capital ($)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(
                        fig_cap, width="stretch", config={"displayModeBar": True}
                    )

                with vc2:
                    fig_std = go.Figure()
                    fig_std.add_trace(
                        go.Scatter(
                            x=valid_df["λ_cap"],
                            y=valid_df["Std Dev"],
                            mode="lines+markers",
                            line=dict(width=3),
                            marker=dict(size=10),
                            name="Cash Flow Std Dev",
                        )
                    )
                    fig_std.update_layout(
                        title="Cash Flow Volatility vs Capital Weight (λ)",
                        xaxis_title="Capital Weight (λ)",
                        yaxis_title="Monthly Std Dev ($)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(
                        fig_std, width="stretch", config={"displayModeBar": True}
                    )

                st.divider()
                fig_front = go.Figure()
                fig_front.add_trace(
                    go.Scatter(
                        x=valid_df["Std Dev"],
                        y=valid_df["Capital"],
                        mode="markers+lines",
                        marker=dict(
                            size=valid_df["Positions"] * 2 + 4,
                            color=valid_df["λ_cap"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="λ_cap"),
                        ),
                        text=[
                            f"λ={row['λ_cap']}<br>Pos={row['Positions']}"
                            for _, row in valid_df.iterrows()
                        ],
                        hovertemplate="<b>Capital:</b> $%{y:,.0f}<br>"
                        "<b>Std Dev:</b> $%{x:.2f}<br>"
                        "%{text}<extra></extra>",
                        name="Frontier",
                    )
                )
                fig_front.update_layout(
                    title="Efficient Frontier: Capital vs Cash Flow Smoothness",
                    xaxis_title="Monthly Cash Flow Std Dev ($)",
                    yaxis_title="Total Capital Required ($)",
                    height=600,
                )
                st.plotly_chart(
                    fig_front, width="stretch", config={"displayModeBar": True}
                )

                # Insights
                st.divider()
                st.markdown("### Key Insights")
                min_cap_idx = valid_df["Capital"].idxmin()
                min_vol_idx = valid_df["Std Dev"].idxmin()

                ins1, ins2 = st.columns(2)
                with ins1:
                    st.info(
                        f"""
**Minimum Capital Configuration**
- λ_capital: {valid_df.loc[min_cap_idx, 'λ_cap']}
- Capital: ${valid_df.loc[min_cap_idx, 'Capital']:,.0f}
- Monthly Std Dev: ${valid_df.loc[min_cap_idx, 'Std Dev']:.2f}
- Positions: {valid_df.loc[min_cap_idx, 'Positions']}
"""
                    )
                with ins2:
                    st.info(
                        f"""
**Smoothest Cash Flow Configuration**
- λ_capital: {valid_df.loc[min_vol_idx, 'λ_cap']}
- Capital: ${valid_df.loc[min_vol_idx, 'Capital']:,.0f}
- Monthly Std Dev: ${valid_df.loc[min_vol_idx, 'Std Dev']:.2f}
- Positions: {valid_df.loc[min_vol_idx, 'Positions']}
"""
                    )
            else:
                st.warning("No valid frontier points (all attempts failed).")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:0.75rem;color:gray;'>© "
    f"{datetime.now().year} Dividend Portfolio Optimizer | "
    "Built with Streamlit & CVXPY</p>",
    unsafe_allow_html=True,
)
