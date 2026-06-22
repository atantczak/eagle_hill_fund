import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


class ArxivTool:
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, query="medicine", max_results=5):
        self.query = query
        self.max_results = max_results

    def fetch_articles(self):
        """
        Fetches articles from arXiv and downloads the full-text PDFs.

        :return: List of articles with titles and PDF download links.
        """
        params = {"search_query": f"all:{self.query}", "start": 0, "max_results": self.max_results}

        response = requests.get(self.BASE_URL, params=params)
        root = ET.fromstring(response.content)

        articles = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            pdf_link = ""

            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]
                    break

            articles.append({"title": title, "summary": summary, "pdf_link": pdf_link})

            if pdf_link:
                pdf_response = requests.get(pdf_link)
                pdf_filename = f"{title[:50].replace(' ', '_')}.pdf"
                with open(pdf_filename, "wb") as f:
                    f.write(pdf_response.content)
                print(f"Downloaded: {pdf_filename}")

        return articles

    import requests
    import xml.etree.ElementTree as ET
    from datetime import datetime

    def fetch_top_articles_today(self, max_results=10):
        """
        Fetches the top arXiv articles submitted today.

        :param max_results: Number of articles to fetch.
        :return: List of articles with titles and summaries.
        """
        today = datetime.now().strftime("%Y%m%d")
        last_week = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        query = f"submittedDate:[{last_week}0000 TO {today}2359]"

        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)

        articles = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            pdf_link = ""

            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]
                    break

            articles.append({"title": title, "summary": summary, "pdf_link": pdf_link})

        return articles

    def fetch_article_full_text(self, article_id):
        """
        Fetches the full text of an article given its ID.

        :param article_id: The ID of the article to fetch.
        :return: The full text of the article.
        """
        params = {"id_list": article_id, "start": 0, "max_results": 1}

        response = requests.get(self.BASE_URL, params=params)
        root = ET.fromstring(response.content)

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            full_text = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            return full_text

        return None
