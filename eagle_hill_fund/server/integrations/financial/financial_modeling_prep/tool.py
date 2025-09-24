from typing import Dict, List, Optional, Union
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
        