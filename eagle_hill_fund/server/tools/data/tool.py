import io
import logging

import pandas as pd
import requests
import requests_mock

logging.basicConfig(level=logging.INFO)


class APIClient:
    def __init__(self, base_url=None):
        """
        Initializes the APIHandler with a base URL.

        :param base_url: Base URL to be prepended to all endpoint calls (optional).
        """
        self.base_url = base_url

    def get_status_code(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        response = requests.request(**kwargs)
        return response.status_code

    def test_connection(self, **kwargs):
        """
        Mocks an API request call to ensure a status code of 200.

        :return:
        """
        with requests_mock.Mocker() as m:
            m.request(**kwargs, status_code=200)
            return self.get_status_code(**kwargs) == 200

    def _process_dataframe(self, api_response, **kwargs):
        """
        Converts a given API response into a pandas DataFrame.

        This function supports responses which are in dictionary,
        list of dictionaries, CSV-formatted string, or XML-formatted string format.

        Parameters:
        api_response (str/dict/list of dicts): API response to be converted to a DataFrame

        Returns:
        dataframe: DataFrame object constructed from the API response

        Raises:
        ValueError: If the provided api_response is not in one of the supported formats
        """

        # If the response is a string, try to parse it as CSV or XML
        if isinstance(api_response, str):
            try:
                dataframe = pd.read_csv(io.StringIO(api_response))
            except pd.errors.ParserError:
                try:
                    dataframe = pd.read_xml(io.StringIO(api_response))
                except ValueError as e:
                    raise ValueError("When api_response is a string, it should be in CSV or XML format") from e

        # If the response is a dictionary, convert it into a DataFrame
        elif isinstance(api_response, dict):
            dataframe = pd.DataFrame([api_response])

        # If the response is a list of dictionaries, convert it into a DataFrame
        elif isinstance(api_response, list) and all(isinstance(item, dict) for item in api_response):
            dataframe = pd.DataFrame(api_response)

        else:
            raise ValueError("api_response should be a dict, a list of dicts, or a CSV or XML-formatted string")

        return dataframe

    def make_api_request(self, method, endpoint: str = None, test: bool = False, **kwargs):
        """
        :param method: method for the new :class:`Request` object: ``GET``, ``OPTIONS``, ``HEAD``, ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
        :param endpoint: (optional) URL of the endpoint; base url will be used in case where this is not given
        :param test: (optional; default = False) Used internally to either run a test or run an actual request
        :param params: (optional) Dictionary, list of tuples or bytes to send
            in the query string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the :class:`Request`.
        :param files: (optional) Dictionary of ``'name': file-like-objects`` (or ``{'name': file-tuple}``) for multipart encoding upload.
            ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``, 3-tuple ``('filename', fileobj, 'content_type')``
            or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``, where ``'content-type'`` is a string
            defining the content type of the given file and ``custom_headers`` a dict-like object containing additional headers
            to add for the file.
        :param auth: (optional) Auth tuple to enable Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How many seconds to wait for the server to send data
            before giving up, as a float, or a :ref:`(connect timeout, read
            timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Boolean. Enable/disable GET/OPTIONS/POST/PUT/PATCH/DELETE/HEAD redirection. Defaults to ``True``.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
                the server's TLS certificate, or a string, in which case it must be a path
                to a CA bundle to use. Defaults to ``True``.
        :param stream: (optional) if ``False``, the response content will be immediately downloaded.
        :param cert: (optional) if String, path to ssl client cert file (.pem). If Tuple, ('cert', 'key') pair.
        :return: :class:`Response <Response>` object
        :rtype: requests.Response
        :return:
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
