import io
import logging

import pandas as pd
import requests
import requests_mock

logging.basicConfig(level=logging.INFO)


import io
import logging
from abc import ABC, abstractmethod

import pandas as pd
import requests
import requests_mock

logging.basicConfig(level=logging.INFO)


class DataTransmissionTool(ABC):
    def __init__(self, base_url=None):
        """
        Initializes the DataTransmissionTool with a base URL.

        :param base_url: Base URL to be prepended to all endpoint calls (optional).
        """
        self.base_url = base_url

    @abstractmethod
    def get_status_code(self, **kwargs):
        pass

    @abstractmethod
    def test_connection(self, **kwargs):
        pass

    @abstractmethod
    def _process_dataframe(self, api_response, **kwargs):
        pass

    @abstractmethod
    def make_api_request(self, method, endpoint: str = None, test: bool = False, **kwargs):
        pass

    def get_status_code(self, **kwargs):
        """
        Sends a request and returns the status code.

        :param kwargs: Arguments for the request.
        :return: Status code of the response.
        """
        response = requests.request(**kwargs)
        return response.status_code

    def test_connection(self, **kwargs):
        """
        Mocks an API request call to ensure a status code of 200.

        :param kwargs: Arguments for the request.
        :return: True if the status code is 200, False otherwise.
        """
        with requests_mock.Mocker() as m:
            m.request(**kwargs, status_code=200)
            return self.get_status_code(**kwargs) == 200

    def _process_dataframe(self, api_response, **kwargs):
        """
        Converts a given API response into a pandas DataFrame.

        This function supports responses which are in dictionary,
        list of dictionaries, CSV-formatted string, or XML-formatted string format.

        :param api_response: API response to be converted to a DataFrame.
        :type api_response: str | dict | list[dict]
        :return: DataFrame object constructed from the API response.
        :rtype: pd.DataFrame
        :raises ValueError: If the provided api_response is not in one of the supported formats.
        """
        if isinstance(api_response, str):
            try:
                dataframe = pd.read_csv(io.StringIO(api_response))
            except pd.errors.ParserError:
                try:
                    dataframe = pd.read_xml(io.StringIO(api_response))
                except ValueError as e:
                    raise ValueError("When api_response is a string, it should be in CSV or XML format") from e
        elif isinstance(api_response, dict):
            dataframe = pd.DataFrame([api_response])
        elif isinstance(api_response, list) and all(isinstance(item, dict) for item in api_response):
            dataframe = pd.DataFrame(api_response)
        else:
            raise ValueError("api_response should be a dict, a list of dicts, or a CSV or XML-formatted string")

        return dataframe

    def make_api_request(self, method, endpoint: str = None, test: bool = False, **kwargs):
        """
        Makes an API request.

        :param method: HTTP method for the request: ``GET``, ``OPTIONS``, ``HEAD``, ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
        :param endpoint: (optional) URL of the endpoint; base URL will be used if this is not given.
        :param test: (optional; default=False) Used internally to either run a test or run an actual request.
        :param kwargs: Additional arguments for the request.
        :return: Response object or error message.
        :rtype: requests.Response | dict
        """
        url = f"{self.base_url}{endpoint}" if endpoint else self.base_url
        try:
            if test:
                return self.test_connection(method=method, url=url, **kwargs)
            else:
                response = requests.request(method=method, url=url, **kwargs)
                response.raise_for_status()
            return response
        except requests.RequestException as e:
            return {"error": str(e)}
