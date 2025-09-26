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
        def fetch_prices(symbol):
            """Fetch price data for a single symbol."""
            if interval == "1day":
                return self.price_data_tool.get_daily_prices(
                    symbol=symbol, 
                    from_date=from_date, 
                    to_date=to_date
                )
            else:
                return self.price_data_tool.get_intraday_prices(
                    symbol=symbol,
                    interval=interval,
                    from_date=from_date, 
                    to_date=to_date
                )

        # Calculate FMP rate limiting
        fmp_rate_limit = force_rate_limit or self._calculate_fmp_rate_limit(len(symbols))
        
        if fmp_rate_limit:
            print(f"⏱️  FMP rate limited to {fmp_rate_limit:.2f} calls/second")
            # Use rate-limited parallel execution
            results = self.parallel_tool.rate_limited_parallel_map(
                fetch_prices, 
                symbols,
                calls_per_second=fmp_rate_limit,
                max_workers=force_workers or self.parallel_tool.get_dynamic_workers_for_task(len(symbols))
            )
        else:
            print(f"🚀 No FMP rate limiting - full speed ahead!")
            # Use smart parallelization - handles hardware optimization
            results = self.parallel_tool.smart_parallel_map(
                fetch_prices, 
                symbols,
                force_workers=force_workers
            )

        # Flatten the list of lists and filter out any empty results
        all_daily_prices = [item for sublist in results if sublist for item in sublist]
        
        # Convert to DataFrame
        if all_daily_prices:
            df = pd.DataFrame(all_daily_prices)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
