import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT') or 'MovieSentimentAnalyzer/1.0'
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH') or './models/bert_sentiment_model'
    CACHE_DIR = os.environ.get('CACHE_DIR') or './cache'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    
    # Application settings
    MAX_POSTS_PER_REQUEST = 100
    REDDIT_SEARCH_LIMIT = 50
    MODEL_MAX_LENGTH = 512

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Production-specific settings
    WORKERS = int(os.environ.get('WORKERS', 4))
    TIMEOUT = int(os.environ.get('TIMEOUT', 120))

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}