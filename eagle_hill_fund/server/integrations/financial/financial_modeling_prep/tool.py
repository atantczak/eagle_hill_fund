from typing import Dict, List, Optional, Union
from eagle_hill_fund.server.integrations.financial.financial_modeling_prep.price_data.tool import FMPPriceDataTool
from eagle_hill_fund.server.integrations.financial.financial_modeling_prep.financials.tool import FMPFinancialsTool
from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient

import os
from dotenv import (
    load_dotenv,
)

load_dotenv()


class FinancialModelingPrepTool(APIClient):
    def __init__(self):
        """Initialize Financial Modeling Prep API client.
        
        Args:
            api_key: API key for authentication with FMP API
        """
        super().__init__(base_url="https://financialmodelingprep.com/stable")
        self.api_key = os.getenv("FMP_API_KEY")
        self.price_data_tool = FMPPriceDataTool()
        self.financials_tool = FMPFinancialsTool()

    def get_stock_list(self, actively_trading: bool = False) -> List[Dict]:
        """
        Retrieve the list of stocks from Financial Modeling Prep.

        Args:
            actively_trading: If True, retrieve only actively traded stocks. If False, retrieve all stocks.

        Returns:
            List of dictionaries, each representing a stock.
        """
        endpoint = "/actively-trading-list" if actively_trading else "/stock-list"
        response = self.get(endpoint, params={"apikey": self.api_key})
        return response.json()

    def get_available_exchanges(self) -> List[str]:
        """
        Retrieve the list of available stock exchanges.

        Returns:
            List of exchange names as strings.
        """
        endpoint = "/available-exchanges"
        response = self.get(endpoint, params={"apikey": self.api_key})
        return response.json()

    def get_available_sectors(self) -> List[str]:
        """
        Retrieve the list of available sectors.

        Returns:
            List of sector names as strings.
        """
        endpoint = "/available-sectors"
        response = self.get(endpoint, params={"apikey": self.api_key})
        return response.json()

    def get_available_industries(self) -> List[str]:
        """
        Retrieve the list of available industries.

        Returns:
            List of industry names as strings.
        """
        endpoint = "/available-industries"
        response = self.get(endpoint, params={"apikey": self.api_key})
        return response.json()
        