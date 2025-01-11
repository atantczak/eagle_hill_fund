import os
from dotenv import load_dotenv
import requests
from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient


class ThreadsClient(APIClient):
    def __init__(self):
        super().__init__(base_url="https://api.threads.com/v1/")  # Not even sure if this is accurate.
        load_dotenv()
        self.access_token = os.getenv("THREADS_ACCESS_TOKEN")  # Does not exist yet.

        if not self.access_token:
            raise ValueError("Access token is missing. Check your .env file.")

    def post_thread(self, content):
        url = f"{self.base_url}threads"
        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
        data = {"content": content}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            print("Thread posted successfully!")
        else:
            print(f"Failed to post thread: {response.status_code} - {response.text}")

    def get_recent_threads(self, user_id, max_results=5):
        url = f"{self.base_url}users/{user_id}/threads"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"max_results": max_results}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Failed to retrieve threads: {response.status_code} - {response.text}")
            return []
