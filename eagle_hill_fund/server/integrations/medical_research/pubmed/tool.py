from datetime import datetime

from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient


class PubMedTool(APIClient):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, base_url=None):
        super().__init__(base_url=base_url or self.BASE_URL)

    def fetch_articles(self, query, max_results=5):
        endpoint = "esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        return self.make_api_request("GET", endpoint, params=params)

    def fetch_article_details(self, ids):
        endpoint = "esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        return self.make_api_request("GET", endpoint, params=params)

    def fetch_full_text_links(self, ids):
        endpoint = "elink.fcgi"
        params = {"dbfrom": "pubmed", "linkname": "pubmed_pubmed", "id": ",".join(ids), "retmode": "json"}
        return self.make_api_request("GET", endpoint, params=params)

    def fetch_top_articles(self, query, sort_by="relevance", max_results=10):
        """
        Fetches the top 10 articles based on a query and sorts them by the specified parameter.

        :param query: The search query for fetching articles.
        :param sort_by: The parameter to sort articles by (e.g., "relevance", "date").
        :param max_results: The maximum number of articles to fetch (default is 10).
        :return: The response containing the top articles.
        """
        endpoint = "esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": sort_by}
        return self.make_api_request("GET", endpoint, params=params)

    def fetch_top_articles_today(self, max_results=10):
        """
        Fetches the top 10 articles from today and sorts them by relevance.

        :param max_results: The maximum number of articles to fetch (default is 10).
        :return: The response containing the top articles.
        """
        today = datetime.now().strftime("%Y/%m/%d")
        endpoint = "esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f"{today}[PDAT]",
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub+date",
        }
        return self.make_api_request("GET", endpoint, params=params)

    def fetch_article_abstract(self, article_id):
        """
        Fetches the full text of an article given its ID.

        :param article_id: The ID of the article to fetch.
        :return: The response containing the article text.
        """
        endpoint = "efetch.fcgi"
        params = {"db": "pubmed", "id": article_id, "retmode": "text", "rettype": "abstract"}
        return self.make_api_request("GET", endpoint, params=params)
