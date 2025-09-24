from typing import Dict, List
from eagle_hill_fund.server.integrations.financial.financial_modeling_prep.tool import FinancialModelingPrepTool


class FMPFinancialsTool(FinancialModelingPrepTool):
    def __init__(self):
        super().__init__()

    def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict containing company profile data
        """
        endpoint = f"/profile/{symbol}"
        response = self.get(endpoint, params={"apikey": self.api_key})
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
        """
        endpoint = f"/{statement}/{symbol}"
        response = self.get(
            endpoint,
            params={
                "apikey": self.api_key,
                "period": period,
                "limit": limit
            }
        )
        return response.json()