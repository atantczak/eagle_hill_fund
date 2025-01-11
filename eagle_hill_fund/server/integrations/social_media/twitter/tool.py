import os
import tweepy
from dotenv import load_dotenv

from eagle_hill_fund.server.tools.data.transmission.api.tool import APIClient

load_dotenv()


class TwitterClient(APIClient):
    def __init__(self):
        super().__init__(base_url="https://api.twitter.com/2/")
        self.bearer_token = os.getenv('X_BEARER_TOKEN')
        self.client_id = os.getenv('X_CLIENT_ID')
        self.client_secret = os.getenv('X_CLIENT_SECRET')
        self.access_token = os.getenv('X_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('X_ACCESS_TOKEN_SECRET')

        if not all([self.bearer_token, self.client_id, self.client_secret, self.access_token, self.access_token_secret]):
            raise ValueError("OAuth credentials are missing. Check your .env file.")

        self.client = self.authenticate()

    def authenticate(self):
        try:
            client = tweepy.Client(
                consumer_key=self.client_id,
                consumer_secret=self.client_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )
            print("Authenticated with Twitter API v2.")
            return client
        except Exception as e:
            print(f"Authentication failed: {e}")
            raise

    def post_tweet(self, content):
        try:
            response = self.client.create_tweet(text=content)
            if response.data and 'id' in response.data:
                print(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
            else:
                print("Failed to post tweet.")
        except tweepy.TweepyException as e:
            print(f"Failed to post tweet: {e}")

    def get_recent_tweets(self, user_id, max_results=5):
        try:
            response = self.client.get_users_tweets(id=user_id, max_results=max_results)
            if response.data:
                return [tweet['text'] for tweet in response.data]
            else:
                return []
        except Exception as e:
            print(f"Failed to retrieve tweets: {e}")
            return []

    def delete_tweet(self, tweet_id):
        try:
            response = self.client.delete_tweet(tweet_id)
            if response.data['deleted']:
                print("Tweet deleted successfully!")
            else:
                print("Failed to delete tweet.")
        except Exception as e:
            print(f"Failed to delete tweet: {e}")

