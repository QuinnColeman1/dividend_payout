# 📊 Dividend Portfolio Optimizer

Find the dividend portfolio that reaches your target annual income, using as
little capital as possible, with the steadiest month-to-month cash flow, or a
balance of the two.

> ⚠️ **Not financial advice.** Results are based on past dividends, which
> companies can cut or suspend at any time.

## 🌟 Features

- **Three optimization strategies**
  - **Minimize Capital:** the least money needed to reach your dividend target.
  - **Smooth Cash Flow:** spread income as evenly as possible across the 12 months.
  - **Multi-Objective Balance:** weight capital against smoothness with a single slider (λ).
- **Your own ticker list:** any stocks or ETFs on Yahoo Finance.
- **Constraints:** maximum position size, an optional budget, and a minimum dividend yield.
- **Efficient frontier:** see how much extra capital buys a smoother income stream.
- **Charts:** allocation, yield per position, and monthly cash flow.

## 🚀 Quick Start

**Requirements:** Python 3.11+, [Poetry](https://python-poetry.org/docs/) 2.x,
and an internet connection to download market data.

```bash
git clone https://github.com/QuinnColeman1/dividend_payout.git
cd dividend_payout
poetry install
poetry run streamlit run app/streamlit_app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## 📈 Using the App

1. **Choose tickers.** In the sidebar's **Ticker Selection** box, enter one
   ticker per line. A default list of dividend stocks and ETFs is filled in.
2. **Click Initialize Optimizer.** This downloads 5 years of daily prices and
   dividends from Yahoo Finance. Tickers that paid no dividends are skipped. If
   you edit the list afterwards, the sidebar reminds you to initialize again.
3. **Set parameters** on the **Input Parameters** page:

   | Setting | Meaning |
   |---------|---------|
   | Target Annual Dividend | Yearly dividend income to reach ($) |
   | Budget Constraint | Maximum capital to invest; 0 = no limit |
   | Maximum Position Size | Largest share of the portfolio any one ticker can hold |
   | Minimum Yield Threshold | Ignore tickers yielding less than this |
   | Optimization Type | Minimize Capital, Smooth Cash Flow, or Multi-Objective Balance |
   | Capital Weight (λ) | Multi-Objective only: 1 = only minimize capital, 0 = only smooth income |

4. **Click 🚀 Run Optimization**, then open **Optimization Results** to see
   the allocation table, charts and the monthly cash-flow pattern.
5. **Efficient Frontier:** run the multi-objective optimization for several λ
   values and compare capital against monthly volatility.

**Position-size rule:** a maximum position of *p* needs at least 1/*p*
eligible tickers. For example, 15% needs 7 and 5% needs 20. If you have too
few, the app tells you how many are needed.

## 🧮 How It Works

1. **Data:** daily prices are downloaded with `auto_adjust=False`, so they are
   real closing prices, not dividend-adjusted ones.
2. **Yield:** trailing-twelve-month dividends ÷ the latest share price, stored
   as a fraction (0.035 = 3.5%).
3. **Monthly pattern:** for each ticker, dividends per dollar invested in each
   calendar month, blended over the last 3 years (50% / 33% / 17%). Years before
   a ticker started paying are left out, so new dividend payers aren't penalized.
4. **Optimization:** linear and quadratic programs solved with
   [CVXPY](https://www.cvxpy.org/), trying the CLARABEL, ECOS and SCS solvers in
   turn:
   - **Minimize Capital:** minimize total capital, subject to the dividend target,
     position cap and budget. The result spreads money equally across the
     highest yields.
   - **Smooth Cash Flow:** minimize the squared gap between each month's income
     and target ÷ 12.
   - **Multi-Objective:** a normalized λ-weighted sum of both objectives.

## 🗂️ Command-Line Tools

Scrape dividend history for the whole S&P 500 into `dividend_info.parquet`:

```bash
poetry run python src/dividend_payout.py
```

Then run all three optimizations and an efficient frontier on that file:

```bash
poetry run python -m src.dividendportfoliooptimizer --target 10000 --max-position 0.1 --budget 250000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | 10000 | Target annual dividend ($) |
| `--budget` | none | Maximum capital ($) |
| `--max-position` | 0.15 | Maximum position size (fraction) |
| `--data-file` | `dividend_info.parquet` | Data file created by the scraper |

## 🛠️ Development

```bash
poetry install --with dev
make help         # list targets
make test         # pytest with coverage
make lint         # ruff
make check-types  # mypy
make security     # bandit + pip-audit
make all          # all of the above
```

The tests use synthetic data and mock `yfinance`, so they run offline. They
cover the optimizer, the wrapper, the scraper, and the Streamlit app end to end
(via Streamlit's `AppTest`). GitHub Actions runs lint, tests, mypy, bandit and
pip-audit on every push and pull request to `main`.

### Project structure

```
app/streamlit_app.py               Streamlit web interface
src/dividendportfoliooptimizer.py  Optimization engine + CLI
src/optimizer_wrapper.py           Converts yfinance data and exposes the optimizations to the app
src/dividend_payout.py             S&P 500 dividend scraper
tests/                             pytest suite
```

## ⚠️ Limitations

- Yields are based on past dividends; future payouts may differ.
- Taxes, fees and dividend reinvestment are not modeled.
- Prices and dividends come from Yahoo Finance via `yfinance`, an unofficial
  API that can rate-limit or change.
- Positions are fractional dollar amounts, not whole shares.

## 📜 License

MIT. See [LICENSE](LICENSE).
