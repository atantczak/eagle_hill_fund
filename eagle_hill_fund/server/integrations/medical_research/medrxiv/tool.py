import requests
from bs4 import BeautifulSoup

from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient


class MedRxivClient(APIClient):
    def __init__(self, base_url="https://api.medrxiv.org"):
        super().__init__(base_url=base_url)

    def fetch_articles(self, start_date, end_date, category="medrxiv", format="json"):
        url = f"{self.base_url}/details/{category}/{start_date}/{end_date}/0/{format}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("collection", [])
        return articles

    def fetch_article_text(self, doi):
        """
        Scrapes the full-text HTML of an article using its DOI.
        """
        # Convert DOI to a URL-friendly format
        article_url = f"https://www.medrxiv.org/content/{doi}.full"

        try:
            response = requests.get(article_url)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Failed to fetch the article: {e}"

        soup = BeautifulSoup(response.content, "html.parser")
        return soup.text
