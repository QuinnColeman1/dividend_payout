import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
class EnhancedDividendScraper:
    def __init__(self, max_workers: int = 30, batch_size: int = 50):
        """
        Initialize the scraper with concurrency settings.
        """
        self.max_workers = max_workers
        self.batch_size = batch_size

    def get_sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 tickers from Wikipedia."""
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-").tolist()
            print(f"Found {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500: {e}")
            # fallback minimal list
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    def get_all_tickers(self) -> Dict[str, List[str]]:
        """Get all tickers organized by index."""
        tickers_by_index = {}

        sp500_tickers = self.get_sp500_tickers()
        tickers_by_index["sp500"] = sp500_tickers

        return tickers_by_index

    def get_dividend_events(self, ticker: str, index_name: str = "sp500") -> List[Dict[str, Any]]:
        """Fetch dividend events with ex-date and closing price, including index info."""
        events: List[Dict[str, Any]] = []
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="10y", actions=True)

            # Get comprehensive info
            market_cap = info.get("marketCap", 0)
            avg_volume = info.get("averageVolume", 0)

            metadata = {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "USD"),
                "index": index_name,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "exchange": info.get("exchange", "Unknown"),
                "quote_type": info.get("quoteType", "Unknown"),
            }

            if "Dividends" not in hist.columns or hist["Dividends"].empty:
                return events

            # filter positive dividends
            divs = hist["Dividends"][hist["Dividends"] > 0]
            for date, amount in divs.items():
                date_str = date.strftime("%Y-%m-%d")
                # get adjusted close if available, else close
                price = None
                if "Adj Close" in hist.columns:
                    price = hist.at[date, "Adj Close"]
                else:
                    price = hist.at[date, "Close"]

                events.append(
                    {
                        **metadata,
                        "ex_dividend_date": date_str,
                        "dividend_amount": float(amount),
                        "close_price": float(price) if price else None,
                        "yield_pct": float(amount / price) if price else None,
                        "data_fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            return events
        except Exception as e:
            print(f"Error {ticker}: {e}")
            return events

    def process_batch(self, tickers_with_index: List[tuple]) -> List[Dict]:
        """Process a batch of tickers with their index labels."""
        all_events = []
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(tickers_with_index))
        ) as executor:
            futures = {
                executor.submit(self.get_dividend_events, t, idx): (t, idx)
                for t, idx in tickers_with_index
            }

            for future in as_completed(futures):
                ticker, index_name = futures[future]
                try:
                    events = future.result(timeout=60)
                    all_events.extend(events)
                    if events:
                        print(
                            f"✅ {ticker} ({index_name}): {len(events)} dividend events"
                        )
                    else:
                        print(f"⚠️  {ticker} ({index_name}): No dividends found")
                except Exception as e:
                    print(f"❌ {ticker} ({index_name}): {e}")
        return all_events

    def scrape(self) -> pd.DataFrame:
        """Main entry: returns a vectorized DataFrame of all dividend events."""
        print("=" * 60)
        print("Enhanced Dividend Scraper - S&P 500")
        print("=" * 60)

        # Get tickers organized by index
        tickers_by_index = self.get_all_tickers()

        # Flatten to list of (ticker, index) tuples
        ticker_index_pairs = []
        for index_name, tickers in tickers_by_index.items():
            ticker_index_pairs.extend([(t, index_name) for t in tickers])

        total_tickers = len(ticker_index_pairs)
        print(f"\nTotal tickers to process: {total_tickers}")
        for index_name, tickers in tickers_by_index.items():
            print(f"  - {index_name}: {len(tickers)} tickers")
        print()

        events = []
        for i in range(0, total_tickers, self.batch_size):
            batch = ticker_index_pairs[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_tickers + self.batch_size - 1) // self.batch_size

            print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} tickers)")
            print("-" * 40)

            batch_events = self.process_batch(batch)
            events.extend(batch_events)

            if i + self.batch_size < total_tickers:
                time.sleep(2)  # Rate limiting

        # Create DataFrame
        df = pd.DataFrame(events)

        if not df.empty:
            # Ensure correct types
            df["dividend_amount"] = df["dividend_amount"].astype(float)
            df["close_price"] = df["close_price"].astype(float)
            df["yield_pct"] = df["yield_pct"].astype(float)
            df["market_cap"] = df["market_cap"].astype(float)
            df["avg_volume"] = df["avg_volume"].astype(float)

        return df


def main():
    scraper = EnhancedDividendScraper(max_workers=20, batch_size=30)

    # Scrape S&P 500
    df = scraper.scrape()

    if not df.empty:
        # Summary statistics
        print("\n" + "=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)
        print(f"Total dividend events: {len(df)}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print("\nBy Index:")
        print(df.groupby("index")["ticker"].nunique())
        print("\nBy Sector:")
        print(df.groupby("sector")["ticker"].nunique().head(10))

        # Market cap distribution
        print("\nMarket Cap Distribution:")
        print(
            f"  Mega Cap (>$200B): {df[df['market_cap'] > 200_000_000_000]['ticker'].nunique()}"
        )
        print(
            f"  Large Cap ($10B-$200B): {df[(df['market_cap'] > 10_000_000_000) & (df['market_cap'] <= 200_000_000_000)]['ticker'].nunique()}"
        )
        print(
            f"  Mid Cap ($2B-$10B): {df[(df['market_cap'] > 2_000_000_000) & (df['market_cap'] <= 10_000_000_000)]['ticker'].nunique()}"
        )
        print(
            f"  Small Cap (<$2B): {df[df['market_cap'] <= 2_000_000_000]['ticker'].nunique()}"
        )

        # Save to parquet
        df.to_parquet("dividend_info.parquet")
        print("\n✅ Saved dataset as dividend_info.parquet")
    else:
        print("\n❌ No data collected!")


if __name__ == "__main__":
    _start_time = time.perf_counter()
    main()
    _end_time = time.perf_counter()
    print(f"\n⏱️ Total runtime: {_end_time - _start_time:.2f} seconds")
