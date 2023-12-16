import requests
import logging


class DataTransport:
    def __init__(self):
        pass

    def get_data(self):
        """
        Get the data at the source.
        Meant to be overridden by subclasses based on specific source type.
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement method get_data")

    def transmit_data(self, data):
        """
        Put the data at the destination.
        Meant to be overridden by subclasses based on specific destination type.
        Args:
            data: The data to transport.
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement method transmit_data")

    @staticmethod
    def api_request_method(method, url, data=None, headers=None):
        """
        Make an API request.
        Args:
            method (str): The HTTP method (e.g., 'GET', 'POST', etc.)
            url (str): The URL for the API endpoint.
            data (dict, optional): Data to include in the request. Defaults to None.
            headers (dict, optional): Headers to include in the request. Defaults to None.
        """
        try:
            response = requests.request(method, url, json=data, headers=headers)
            if response.status_code == 200:
                logging.info("Success: {}".format(response.json()))
            elif response.status_code == 400:
                logging.warning("Bad Request: {}".format(response.text))
            elif response.status_code == 401:
                logging.warning("Unauthorized: {}".format(response.text))
            elif response.status_code == 403:
                logging.warning("Forbidden: {}".format(response.text))
            elif response.status_code == 404:
                logging.warning("Not Found: {}".format(response.text))
            else:
                logging.error("Unknown error occurred: {}".format(response.text))
        except requests.exceptions.RequestException as e:
            logging.error("Exception Occurred: {}".format(e))
