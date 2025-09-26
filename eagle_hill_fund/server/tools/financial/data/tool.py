import pandas as pd
from typing import List, Optional

from eagle_hill_fund.server.integrations.financial.financial_modeling_prep.tool import FinancialModelingPrepTool
from eagle_hill_fund.server.tools.data.parallelization import ParallelizationTool


class FinancialDataCompiler(FinancialModelingPrepTool):
    def __init__(self, max_calls_per_minute: int = 3000, safety_factor: float = 0.8):
        """
        Initialize the Financial Data Compiler with FMP rate limiting and smart parallelization.
        
        Args:
            max_calls_per_minute: Maximum API calls per minute (FMP premium limit)
            safety_factor: Safety factor to avoid hitting rate limits (0.8 = 80% of max)
        """
        super().__init__()
        # FMP-specific rate limiting
        self.max_calls_per_minute = max_calls_per_minute
        self.safety_factor = safety_factor
        
        # Hardware optimization handled by ParallelizationTool
        self.parallel_tool = ParallelizationTool()

    def _calculate_fmp_rate_limit(self, symbol_count: int) -> Optional[float]:
        """Calculate FMP rate limiting based on task size and FMP limits."""
        # FMP allows 3,000 calls per minute, use safety factor
        max_calls_per_minute = self.max_calls_per_minute * self.safety_factor
        
        # For small tasks, no rate limiting needed
        if symbol_count <= 50:
            return None
        
        # For larger tasks, calculate if we need rate limiting
        # Assume we want to complete within 1 minute
        calls_per_second = max_calls_per_minute / 60
        
        if symbol_count > max_calls_per_minute:
            # Need rate limiting
            return calls_per_second
        else:
            return None

    def _chunk_intraday_date_ranges(self, from_date: str, to_date: str, interval: str) -> List[tuple]:
        """
        Chunk the date range into subranges that fit within the FMP intraday API's max window.
        For minutely data, FMP only returns up to 1170 minutes (3 trading days) per call.
        
        Args:
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            interval: Data interval
            
        Returns:
            List of (start_date, end_date) tuples
        """
        import datetime
        
        # Only chunk for minutely intervals
        if interval != "1min":
            return [(from_date, to_date)]
        
        # FMP limit: 1170 minutes (3 trading days) per call
        max_minutes = 1170
        dt_format = "%Y-%m-%d"
        start_dt = datetime.datetime.strptime(from_date, dt_format)
        end_dt = datetime.datetime.strptime(to_date, dt_format)
        
        # Assume 390 minutes per trading day (6.5 hours)
        minutes_per_day = 390
        max_days = max_minutes // minutes_per_day  # 3 days

        # Generate chunked date ranges
        ranges = []
        current_start = start_dt
        while current_start <= end_dt:
            current_end = current_start + datetime.timedelta(days=max_days - 1)
            if current_end > end_dt:
                current_end = end_dt
            ranges.append((
                current_start.strftime(dt_format),
                current_end.strftime(dt_format)
            ))
            current_start = current_end + datetime.timedelta(days=1)
        return ranges

    def _fetch_prices_for_symbol(self, symbol: str, from_date: str, to_date: str, interval: str) -> List[dict]:
        """
        Fetch price data for a single symbol, chunking if necessary.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            interval: Data interval
            
        Returns:
            List of price data dictionaries
        """
        if interval == "1day":
            return self.price_data_tool.get_daily_prices(
                symbol=symbol, 
                from_date=from_date, 
                to_date=to_date
            )
        else:
            # For intraday, chunk the date range if needed
            date_ranges = self._chunk_intraday_date_ranges(from_date, to_date, interval)
            
            # Fetch each chunk sequentially to avoid double parallelization
            all_chunks = []
            for chunk_from, chunk_to in date_ranges:
                chunk_data = self.price_data_tool.get_intraday_prices(
                    symbol=symbol,
                    interval=interval,
                    from_date=chunk_from, 
                    to_date=chunk_to
                )
                chunk_data["symbol"] = symbol
                if chunk_data:
                    all_chunks.extend(chunk_data)
            return all_chunks

    def compile_price_data(
        self, 
        symbols: List[str], 
        from_date: str, 
        to_date: str, 
        interval: str = "1min",
        force_workers: Optional[int] = None,
        force_rate_limit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compile price data for multiple symbols with FMP rate limiting and smart parallelization.
        
        Args:
            symbols: List of stock symbols to fetch data for
            from_date: Start date for price data (YYYY-MM-DD format)
            to_date: End date for price data (YYYY-MM-DD format)
            interval: Data interval (default: "1min", "1hour", "1day")
            force_workers: Override automatic worker calculation
            force_rate_limit: Override FMP rate limiting
            
        Returns:
            DataFrame containing compiled price data for all symbols
        """
        from functools import partial
        
        # Create a partial function with the parameters bound
        fetch_func = partial(
            self._fetch_prices_for_symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )

        # Calculate FMP rate limiting
        fmp_rate_limit = force_rate_limit or self._calculate_fmp_rate_limit(len(symbols))
        
        if fmp_rate_limit:
            print(f"⏱️  FMP rate limited to {fmp_rate_limit:.2f} calls/second")
            # Use rate-limited parallel execution
            results = self.parallel_tool.rate_limited_parallel_map(
                fetch_func, 
                symbols,
                calls_per_second=fmp_rate_limit,
                max_workers=force_workers or self.parallel_tool.get_dynamic_workers_for_task(len(symbols))
            )
        else:
            print(f"🚀 No FMP rate limiting - full speed ahead!")
            # Use smart parallelization - handles hardware optimization
            results = self.parallel_tool.smart_parallel_map(
                fetch_func, 
                symbols,
                force_workers=force_workers
            )

        # Flatten the list of lists and filter out any empty results
        all_prices = [item for sublist in results if sublist for item in sublist]
        
        # Convert to DataFrame
        if all_prices:
            df = pd.DataFrame(all_prices)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
