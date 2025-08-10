import praw
import pandas as pd
import re
import time
from typing import List, Dict, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta
import os
from config import Config

class RedditMovieScraper:
    """
    Reddit scraper for movie-related posts and comments
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        """
        Initialize Reddit scraper
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
        """
        self.client_id = client_id or Config.REDDIT_CLIENT_ID
        self.client_secret = client_secret or Config.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or Config.REDDIT_USER_AGENT
        
        self.reddit = None
        self.initialize_reddit()
        
        # Movie-related subreddits
        self.movie_subreddits = [
            'movies',
            'MovieReviews',
            'flicks',
            'TrueFilm',
            'boxoffice',
            'MovieSuggestions',
            'horror',
            'scifi',
            'Marvel',
            'DC_Cinematic'
        ]
        
        logger.info("Reddit scraper initialized")
    
    def initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            if not all([self.client_id, self.client_secret, self.user_agent]):
                logger.warning("Reddit credentials not provided. Some features may not work.")
                return False
            
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit API connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.+?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'\^(.+?)', r'\1', text)        # Superscript
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def search_movie_posts(self, movie_name: str, limit: int = 50, 
                          time_filter: str = 'month') -> List[Dict]:
        """
        Search for posts about a specific movie
        
        Args:
            movie_name: Name of the movie to search for
            limit: Maximum number of posts to retrieve
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
            
        Returns:
            List of post dictionaries
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
        
        posts = []
        search_query = f'"{movie_name}" OR {movie_name.replace(" ", " AND ")}'
        
        try:
            # Search across movie subreddits
            for subreddit_name in self.movie_subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search posts
                    search_results = subreddit.search(
                        search_query,
                        sort='relevance',
                        time_filter=time_filter,
                        limit=limit // len(self.movie_subreddits)
                    )
                    
                    for post in search_results:
                        if self.is_movie_related(post.title + " " + post.selftext, movie_name):
                            post_data = {
                                'id': post.id,
                                'title': post.title,
                                'text': self.clean_text(post.selftext),
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc,
                                'subreddit': post.subreddit.display_name,
                                'author': str(post.author) if post.author else '[deleted]',
                                'url': post.url,
                                'type': 'post'
                            }
                            posts.append(post_data)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error searching subreddit {subreddit_name}: {e}")
                    continue
            
            logger.info(f"Found {len(posts)} posts for movie: {movie_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error searching for movie posts: {e}")
            return []
    
    def get_post_comments(self, post_id: str, limit: int = 100) -> List[Dict]:
        """
        Get comments for a specific post
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comment dictionaries
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
        
        comments = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            
            comment_count = 0
            for comment in submission.comments.list():
                if comment_count >= limit:
                    break
                
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comment_data = {
                        'id': comment.id,
                        'text': self.clean_text(comment.body),
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'parent_id': post_id,
                        'type': 'comment'
                    }
                    comments.append(comment_data)
                    comment_count += 1
            
            logger.info(f"Retrieved {len(comments)} comments for post {post_id}")
            return comments
            
        except Exception as e:
            logger.error(f"Error getting comments for post {post_id}: {e}")
            return []
    
    def is_movie_related(self, text: str, movie_name: str) -> bool:
        """
        Check if text is related to the specified movie
        
        Args:
            text: Text to check
            movie_name: Movie name to look for
            
        Returns:
            Boolean indicating if text is movie-related
        """
        text_lower = text.lower()
        movie_lower = movie_name.lower()
        
        # Direct match
        if movie_lower in text_lower:
            return True
        
        # Check for individual words (for multi-word movie titles)
        movie_words = movie_lower.split()
        if len(movie_words) > 1:
            word_matches = sum(1 for word in movie_words if word in text_lower)
            if word_matches >= len(movie_words) * 0.7:  # At least 70% of words match
                return True
        
        return False
    
    def scrape_movie_data(self, movie_name: str, max_posts: int = 50, 
                         max_comments_per_post: int = 50) -> Dict[str, List[Dict]]:
        """
        Comprehensive movie data scraping
        
        Args:
            movie_name: Name of the movie
            max_posts: Maximum number of posts to retrieve
            max_comments_per_post: Maximum comments per post
            
        Returns:
            Dictionary containing posts and comments
        """
        logger.info(f"Starting comprehensive scrape for movie: {movie_name}")
        
        # Get posts
        posts = self.search_movie_posts(movie_name, limit=max_posts)
        
        # Get comments for each post
        all_comments = []
        for post in posts[:10]:  # Limit to top 10 posts to avoid rate limiting
            comments = self.get_post_comments(post['id'], limit=max_comments_per_post)
            all_comments.extend(comments)
            time.sleep(1)  # Rate limiting
        
        result = {
            'movie_name': movie_name,
            'posts': posts,
            'comments': all_comments,
            'total_posts': len(posts),
            'total_comments': len(all_comments),
            'scraped_at': datetime.now().isoformat()
        }
        
        logger.info(f"Scraping completed. Posts: {len(posts)}, Comments: {len(all_comments)}")
        return result
    
    def get_all_texts(self, scraped_data: Dict) -> List[str]:
        """
        Extract all text content from scraped data
        
        Args:
            scraped_data: Data returned from scrape_movie_data
            
        Returns:
            List of all text content
        """
        texts = []
        
        # Add post titles and content
        for post in scraped_data.get('posts', []):
            if post.get('title'):
                texts.append(post['title'])
            if post.get('text') and post['text'].strip():
                texts.append(post['text'])
        
        # Add comments
        for comment in scraped_data.get('comments', []):
            if comment.get('text') and comment['text'].strip():
                texts.append(comment['text'])
        
        # Filter out very short texts
        texts = [text for text in texts if len(text.split()) >= 3]
        
        return texts
    
    def save_scraped_data(self, scraped_data: Dict, filename: str = None):
        """Save scraped data to file"""
        if not filename:
            movie_name = scraped_data.get('movie_name', 'unknown').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scraped_data_{movie_name}_{timestamp}.json"
        
        os.makedirs('./data/scraped', exist_ok=True)
        filepath = f"./data/scraped/{filename}"
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraped data saved to {filepath}")
        return filepath

def main():
    """Test the scraper"""
    scraper = RedditMovieScraper()
    
    if scraper.reddit:
        # Test with a popular movie
        movie_name = "Avengers Endgame"
        data = scraper.scrape_movie_data(movie_name, max_posts=10, max_comments_per_post=20)
        
        print(f"Scraped data for '{movie_name}':")
        print(f"Posts: {data['total_posts']}")
        print(f"Comments: {data['total_comments']}")
        
        # Get all texts
        texts = scraper.get_all_texts(data)
        print(f"Total text samples: {len(texts)}")
        
        # Save data
        scraper.save_scraped_data(data)
        
    else:
        print("Reddit API not available. Please check your credentials.")

if __name__ == "__main__":
    main()