import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "https://bsky.social"

class BlueskyAPI:
    def __init__(self):
        self.auth_token = None

    def login(self, username, password):
        """
        Authenticate with Bluesky API and retrieve an access token.
        """
        url = f"{BASE_URL}/xrpc/com.atproto.server.createSession"
        payload = {"identifier": username, "password": password}
        headers = {"Content-Type": "application/json"}

        print("payload---", payload)

        response = requests.post(url, json=payload, headers=headers)
        print("login---res---", response)
        print("login---res---", response.text)
        if response.status_code == 200:
            self.auth_token = response.json().get("accessJwt")
            return {"success": True, "auth_token": self.auth_token}
        return {"success": False, "error": response.json()}

    def fetch_user_data(self, handle):
        """
        Fetch user profile data by handle.
        """
        # Remove '@' if it exists at the start of the handle
        sanitized_handle = handle.lstrip('@')

        url = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
        params = {"actor": sanitized_handle}

        try:
            # Make the GET request
            response = requests.get(url, params=params)
            
            # Print debugging information
            print("Request URL:", response.url)
            print("Response Status Code:", response.status_code)
            
            # Check for a successful response
            if response.status_code == 200:
                print("Response JSON:", response.json())
                return response.json()
            else:
                print("Error Response:", response.text)
                return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            print("Exception Occurred:", str(e))
            return {"error": str(e)}

    def fetch_user_timeline(self):
        """
        Fetch authenticated user's timeline.
        """
        url = f"{BASE_URL}/xrpc/app.bsky.feed.getTimeline"
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        response = requests.get(url, headers=headers)
        print("response---", response)
        return response.json() if response.status_code == 200 else response.json()
    

    def fetch_user_posts(self, handle, limit=50, cursor=None, filter_type="posts_with_replies", include_pins=False):
        """
        Fetch a user's author feed (posts and reposts).
        
        Args:
            handle (str): The user's handle (at-identifier)
            limit (int, optional): Number of posts to return (1-100). Defaults to 50.
            cursor (str, optional): Pagination cursor. Defaults to None.
            filter_type (str, optional): Type of posts to include. 
                Options: posts_with_replies, posts_no_replies, posts_with_media, posts_and_author_threads
                Defaults to "posts_with_replies".
            include_pins (bool, optional): Whether to include pinned posts. Defaults to False.
        
        Returns:
            dict: Response containing feed items and optional cursor for pagination
        """
        # Remove '@' if it exists at the start of the handle
        sanitized_handle = handle.lstrip('@')
        
        # Validate limit
        if not 1 <= limit <= 100:
            limit = 50  # Reset to default if invalid
            
        # Build URL and parameters
        url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
        params = {
            "actor": sanitized_handle,
            "limit": limit,
            "filter": filter_type,
            "includePins": str(include_pins).lower()
        }
        
        # Add cursor if provided
        if cursor:
            params["cursor"] = cursor
            
        try:
            response = requests.get(url, params=params)
            
            # Print debugging information
            print("Request URL:", response.url)
            print("Response Status Code:", response.status_code)
            
            if response.status_code == 200:
                return response.json()
            else:
                print("Error Response:", response.text)
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            print("Exception Occurred:", str(e))
            return {"error": str(e)}
