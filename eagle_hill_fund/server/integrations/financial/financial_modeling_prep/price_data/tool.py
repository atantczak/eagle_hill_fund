from typing import Dict, List
from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient
import os
from dotenv import (
    load_dotenv,
)
load_dotenv()


class FMPPriceDataTool(APIClient):
    def __init__(self):
        super().__init__(base_url="https://financialmodelingprep.com/stable")
        self.api_key = os.getenv("FMP_API_KEY")

    def get_stock_price(self, symbol: str) -> Dict:
        """Get current stock price data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict containing current price data
        """
        endpoint = f"/quote/{symbol}"
        response = self.get(endpoint, params={"apikey": self.api_key})
        return response.json()[0] if response.json() else {}

    def get_intraday_prices(
        self,
        symbol: str,
        interval: str = "1min",
        from_date: str = None,
        to_date: str = None,
        nonadjusted: bool = False
    ) -> List[Dict]:
        """Get intraday stock price data.
        
        Args:
            symbol: Stock ticker symbol (e.g. 'AAPL')
            interval: Time interval between prices, defaults to "1min", "1hour"
            from_date: Start date in YYYY-MM-DD format (e.g. '2024-01-01'), optional
            to_date: End date in YYYY-MM-DD format (e.g. '2024-03-01'), optional
            nonadjusted: Whether to return non-adjusted price data, optional
            
        Returns:
            List of dictionaries containing intraday price data
        """
        endpoint = f"/historical-chart/{interval}"
        params = {"apikey": self.api_key, "symbol": symbol}
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if nonadjusted:
            params["nonadjusted"] = ""
            
        response = self.get(endpoint, params=params)
        return response.json()

    def get_daily_prices(
        self,
        symbol: str,
        from_date: str = None,
        to_date: str = None
    ) -> List[Dict]:
        """Get daily (end of day) stock price data.
        
        Args:
            symbol: Stock ticker symbol (e.g. 'AAPL')
            from_date: Start date in YYYY-MM-DD format (e.g. '2024-01-01'), optional
            to_date: End date in YYYY-MM-DD format (e.g. '2024-03-01'), optional
            
        Returns:
            List of dictionaries containing daily price data
        """
        endpoint = f"/historical-price-eod/full"
        params = {
            "apikey": self.api_key,
            "symbol": symbol
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        response = self.get(endpoint, params=params)
        return response.json()