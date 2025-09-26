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

    def screen_companies(
        self,
        market_cap_more_than: Optional[float] = None,
        market_cap_lower_than: Optional[float] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        beta_more_than: Optional[float] = None,
        beta_lower_than: Optional[float] = None,
        price_more_than: Optional[float] = None,
        price_lower_than: Optional[float] = None,
        dividend_more_than: Optional[float] = None,
        dividend_lower_than: Optional[float] = None,
        volume_more_than: Optional[float] = None,
        volume_lower_than: Optional[float] = None,
        exchange: Optional[str] = None,
        country: Optional[str] = None,
        is_etf: Optional[bool] = None,
        is_fund: Optional[bool] = None,
        is_actively_trading: Optional[bool] = None,
        limit: Optional[int] = None,
        include_all_share_classes: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Screen companies using the Financial Modeling Prep stock screener endpoint.

        Args:
            market_cap_more_than: Minimum market capitalization.
            market_cap_lower_than: Maximum market capitalization.
            sector: Sector name.
            industry: Industry name.
            beta_more_than: Minimum beta.
            beta_lower_than: Maximum beta.
            price_more_than: Minimum price.
            price_lower_than: Maximum price.
            dividend_more_than: Minimum dividend.
            dividend_lower_than: Maximum dividend.
            volume_more_than: Minimum volume.
            volume_lower_than: Maximum volume.
            exchange: Exchange name.
            country: Country code.
            is_etf: Whether to include only ETFs.
            is_fund: Whether to include only funds.
            is_actively_trading: Whether to include only actively trading stocks.
            limit: Maximum number of results.
            include_all_share_classes: Whether to include all share classes.

        Returns:
            List of dictionaries, each representing a screened company.
        """
        endpoint = "/company-screener"
        
        # Build params dict, filtering out None values
        param_mapping = {
            "marketCapMoreThan": market_cap_more_than,
            "marketCapLowerThan": market_cap_lower_than,
            "sector": sector,
            "industry": industry,
            "betaMoreThan": beta_more_than,
            "betaLowerThan": beta_lower_than,
            "priceMoreThan": price_more_than,
            "priceLowerThan": price_lower_than,
            "dividendMoreThan": dividend_more_than,
            "dividendLowerThan": dividend_lower_than,
            "volumeMoreThan": volume_more_than,
            "volumeLowerThan": volume_lower_than,
            "exchange": exchange,
            "country": country,
            "isEtf": str(is_etf).lower() if is_etf is not None else None,
            "isFund": str(is_fund).lower() if is_fund is not None else None,
            "isActivelyTrading": str(is_actively_trading).lower() if is_actively_trading is not None else None,
            "limit": 100000 if limit is None else limit,
            "includeAllShareClasses": str(include_all_share_classes).lower() if include_all_share_classes is not None else None,
        }
        
        params = {"apikey": self.api_key}
        params.update({k: v for k, v in param_mapping.items() if v is not None})

        response = self.get(endpoint, params=params)
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
        