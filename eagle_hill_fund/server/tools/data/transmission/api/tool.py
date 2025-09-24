import io
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
import requests_mock

from eagle_hill_fund.server.tools.data.transmission.tool import DataTransmissionTool


class APIClient(DataTransmissionTool):
    """A flexible client for making API requests and processing responses.
    
    This client provides methods for:
    - Making HTTP requests to APIs
    - Converting API responses to pandas DataFrames
    - Testing API connectivity
    - Handling errors gracefully
    
    Example:
        client = APIClient("https://api.example.com")
        response = client.get("/users")  # Makes GET request
        df = client.to_dataframe(response.json())  # Convert to DataFrame
    """

    def __init__(self, base_url: Optional[str] = None):
        """Initialize the APIClient.

        Args:
            base_url: Base URL to be prepended to all endpoint calls.
                     E.g. "https://api.example.com"
        """
        super().__init__(connection_string=base_url)
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self.make_api_request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self.make_api_request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a PUT request."""
        return self.make_api_request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a DELETE request."""
        return self.make_api_request("DELETE", endpoint, **kwargs)

    def test_connection(self, **kwargs) -> bool:
        """Test API connectivity by mocking a request.

        Returns:
            bool: True if connection test succeeds, False otherwise
        """
        try:
            with requests_mock.Mocker() as m:
                m.request(**kwargs, status_code=200)
                return self.get_status_code(**kwargs) == 200
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False

    def to_dataframe(self, api_response: Union[str, Dict, List[Dict]], **kwargs) -> pd.DataFrame:
        """Convert an API response into a pandas DataFrame.

        Supports responses in formats:
        - Dictionary
        - List of dictionaries  
        - CSV-formatted string
        - XML-formatted string

        Args:
            api_response: API response to convert
            **kwargs: Additional arguments passed to pandas

        Returns:
            pandas.DataFrame: Converted response data

        Raises:
            ValueError: If response format is not supported
        """
        try:
            if isinstance(api_response, str):
                try:
                    return pd.read_csv(io.StringIO(api_response))
                except pd.errors.ParserError:
                    return pd.read_xml(io.StringIO(api_response))

            if isinstance(api_response, dict):
                return pd.DataFrame([api_response])

            if isinstance(api_response, list) and all(isinstance(item, dict) for item in api_response):
                return pd.DataFrame(api_response)

            raise ValueError("Response must be a dict, list of dicts, or CSV/XML string")

        except Exception as e:
            self.logger.error(f"Failed to convert response to DataFrame: {str(e)}")
            raise

    def make_api_request(
        self, 
        method: str,
        endpoint: Optional[str] = None,
        test: bool = False,
        timeout: int = 30,
        **kwargs
    ) -> Union[requests.Response, Dict]:
        """Make an API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE etc)
            endpoint: API endpoint path
            test: If True, test connection instead of making real request
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to requests.request()

        Returns:
            Response object or error dict if request fails

        Example:
            response = client.make_api_request(
                "GET",
                "/users",
                params={"page": 1},
                headers={"Authorization": "Bearer token"}
            )
        """
        url = f"{self.base_url}{endpoint}" if endpoint else self.base_url
        
        try:
            if test:
                return self.test_connection(method=method, url=url, **kwargs)

            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response

        except requests.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "details": str(e)}
