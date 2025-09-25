from typing import Dict, List
from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient
import os
from dotenv import (
    load_dotenv,
)
load_dotenv()

class FMPFinancialsTool(APIClient):
    def __init__(self):
        super().__init__(base_url="https://financialmodelingprep.com/stable")
        self.api_key = os.getenv("FMP_API_KEY")

    def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict containing company profile data
        """
        endpoint = f"/profile"
        response = self.get(endpoint, params={"apikey": self.api_key, "symbol": symbol})
        return response.json()[0] if response.json() else {}
        
    def get_financial_statements(
        self, 
        symbol: str,
        statement: str = "income-statement",
        period: str = "annual",
        limit: int = 5
    ) -> List[Dict]:
        """Get financial statements for a company.
        
        Args:
            symbol: Stock ticker symbol
            statement: One of "income-statement", "balance-sheet", "cash-flow"
            period: "annual" or "quarter" 
            limit: Number of periods to return
            
        Returns:
            List of financial statement dictionaries
            https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey=JxuOcSXvYSQcD22QTupGNdPqjVcSRcFK
        """
        endpoint = f"/{statement}"
        response = self.get(
            endpoint,
            params={
                "apikey": self.api_key,
                "period": period,
                "limit": limit,
                "symbol": symbol
            }
        )
        return response.json()