import requests
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from pathlib import Path
import json

# Load environment variables
load_dotenv()

BASE_URL = "https://bsky.social"

class BlueskyAPI:
    def __init__(self):
        self.auth_token = None
        self.refresh_token = None
        self.token_file = Path("tokens.json")
        self._load_tokens()

    def _load_tokens(self):
        """Load tokens from file if they exist and are valid"""
        try:
            if self.token_file.exists():
                with open(self.token_file) as f:
                    data = json.load(f)
                    access_token = data.get('access_token')
                    refresh_token = data.get('refresh_token')
                    
                    # Check if tokens are still valid
                    if access_token and refresh_token:
                        access_exp = jwt.decode(access_token, options={"verify_signature": False})['exp']
                        refresh_exp = jwt.decode(refresh_token, options={"verify_signature": False})['exp']
                        
                        current_time = datetime.now().timestamp()
                        
                        if current_time < access_exp:
                            self.auth_token = access_token
                        elif current_time < refresh_exp:
                            # Access token expired but refresh token valid - try refreshing
                            self._refresh_session(refresh_token)
        except Exception as e:
            print(f"Error loading tokens: {e}")
            self._clear_tokens()

    def _save_tokens(self, access_token, refresh_token):
        """Save tokens to file"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    'access_token': access_token,
                    'refresh_token': refresh_token
                }, f)
        except Exception as e:
            print(f"Error saving tokens: {e}")

    def _clear_tokens(self):
        """Clear stored tokens"""
        self.auth_token = None
        self.refresh_token = None
        if self.token_file.exists():
            self.token_file.unlink()

    def _refresh_session(self, refresh_token):
        """Refresh the session using the refresh token"""
        url = f"{BASE_URL}/xrpc/com.atproto.server.refreshSession"
        headers = {
            "Authorization": f"Bearer {refresh_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data['accessJwt']
                self.refresh_token = data['refreshJwt']
                self._save_tokens(self.auth_token, self.refresh_token)
                return True
        except Exception as e:
            print(f"Error refreshing session: {e}")
        
        return False

    def login(self, username, password):
        """
        Authenticate with Bluesky API and retrieve access tokens.
        """
        url = f"{BASE_URL}/xrpc/com.atproto.server.createSession"
        payload = {"identifier": username, "password": password}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            self.auth_token = data['accessJwt']
            self.refresh_token = data['refreshJwt']
            self._save_tokens(self.auth_token, self.refresh_token)
            return {"success": True, "auth_token": self.auth_token}
        return {"success": False, "error": response.json()}

    def fetch_user_data(self, handle):
        """
        Fetch user profile data by handle.
        """
        sanitized_handle = handle.lstrip('@')
        url = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
        params = {"actor": sanitized_handle}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}

    def fetch_user_timeline(self):
        """
        Fetch authenticated user's timeline.
        """
        if not self.auth_token:
            self._load_tokens()  # Try loading tokens if not available
            
        if not self.auth_token:
            return {"error": "No valid authentication token"}
            
        url = f"{BASE_URL}/xrpc/app.bsky.feed.getTimeline"
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 401 and self.refresh_token:
                # Try refreshing the token
                if self._refresh_session(self.refresh_token):
                    # Retry with new token
                    headers = {"Authorization": f"Bearer {self.auth_token}"}
                    response = requests.get(url, headers=headers)
            
            return response.json() if response.status_code == 200 else response.json()
        except Exception as e:
            return {"error": str(e)}

    def fetch_user_posts(self, handle, limit=50, cursor=None, filter_type="posts_with_replies", include_pins=False):
        """
        Fetch a user's author feed (posts and reposts).
        """
        sanitized_handle = handle.lstrip('@')
        url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
        params = {
            "actor": sanitized_handle,
            "limit": min(max(1, limit), 100),
            "filter": filter_type,
            "includePins": str(include_pins).lower()
        }
        
        if cursor:
            params["cursor"] = cursor
            
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}