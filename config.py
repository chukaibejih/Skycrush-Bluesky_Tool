import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")
    BLUESKY_USERNAME = os.getenv("USERNAME")
    BLUESKY_PASSWORD = os.getenv("PASSWORD")

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SESSION_TYPE = 'filesystem'

    

