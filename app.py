from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
from loguru import logger
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.model.bert_predictor import MovieSentimentPredictor
from src.data.reddit_scraper import RedditMovieScraper

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Enable CORS
    CORS(app)
    
    # Setup logging
    logger.add("./logs/app.log", rotation="10 MB", level=app.config.get('LOG_LEVEL', 'INFO'))
    
    # Initialize components
    predictor = MovieSentimentPredictor(app.config.get('MODEL_PATH', './models/bert_sentiment_model'))
    scraper = RedditMovieScraper()
    
    # Load model on startup
    model_loaded = predictor.load_model()
    if not model_loaded:
        logger.warning("BERT model not loaded. Training may be required.")
    
    @app.route('/')
    def index():
        """Serve the main page"""
        return render_template('index.html')
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_loaded,
            'reddit_available': scraper.reddit is not None
        })
    
    @app.route('/api/predict', methods=['POST'])
    def predict_sentiment():
        """Predict sentiment for given text"""
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'Text is required'}), 400
            
            text = data['text']
            if not text.strip():
                return jsonify({'error': 'Text cannot be empty'}), 400
            
            if not model_loaded:
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
            
            # Predict sentiment
            result = predictor.predict_single(text)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in predict_sentiment: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/api/analyze-movie', methods=['POST'])
    def analyze_movie():
        """Analyze sentiment for a movie using Reddit data"""
        try:
            data = request.get_json()
            
            if not data or 'movie_name' not in data:
                return jsonify({'error': 'Movie name is required'}), 400
            
            movie_name = data['movie_name'].strip()
            if not movie_name:
                return jsonify({'error': 'Movie name cannot be empty'}), 400
            
            if not model_loaded:
                return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
            
            if not scraper.reddit:
                return jsonify({'error': 'Reddit API not available. Please check credentials.'}), 503
            
            # Get parameters
            max_posts = min(data.get('max_posts', 20), 50)  # Limit to prevent overload
            max_comments = min(data.get('max_comments_per_post', 30), 100)
            
            logger.info(f"Starting movie analysis for: {movie_name}")
            
            # Scrape Reddit data
            scraped_data = scraper.scrape_movie_data(
                movie_name, 
                max_posts=max_posts, 
                max_comments_per_post=max_comments
            )
            
            # Extract texts
            texts = scraper.get_all_texts(scraped_data)
            
            if not texts:
                return jsonify({
                    'success': True,
                    'movie_name': movie_name,
                    'message': 'No relevant discussions found for this movie',
                    'total_texts': 0,
                    'sentiment_analysis': None
                })
            
            # Analyze sentiment
            sentiment_analysis = predictor.analyze_sentiment_distribution(texts)
            
            # Prepare response
            response = {
                'success': True,
                'movie_name': movie_name,
                'total_texts': len(texts),
                'scraped_posts': scraped_data['total_posts'],
                'scraped_comments': scraped_data['total_comments'],
                'sentiment_analysis': {
                    'overall_sentiment': sentiment_analysis.get('overall_sentiment'),
                    'sentiment_distribution': sentiment_analysis.get('sentiment_percentages', {}),
                    'average_confidence': sentiment_analysis.get('average_confidence', 0),
                    'sentiment_counts': sentiment_analysis.get('sentiment_counts', {})
                },
                'sample_texts': texts[:5],  # Include first 5 texts as samples
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Movie analysis completed for: {movie_name}")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in analyze_movie: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/api/train-model', methods=['POST'])
    def train_model():
        """Train the BERT model (for development/testing)"""
        try:
            if not app.config.get('DEBUG', False):
                return jsonify({'error': 'Training endpoint only available in debug mode'}), 403
            
            logger.info("Starting model training...")
            
            # Import trainer
            from src.model.bert_trainer import MovieSentimentBERTTrainer
            
            trainer = MovieSentimentBERTTrainer()
            trainer.full_training_pipeline()
            
            # Reload the predictor
            global model_loaded
            model_loaded = predictor.load_model()
            
            return jsonify({
                'success': True,
                'message': 'Model training completed successfully',
                'model_loaded': model_loaded,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            return jsonify({'error': 'Training failed'}), 500
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        """Get model information"""
        try:
            if not model_loaded:
                return jsonify({'error': 'Model not loaded'}), 503
            
            info = predictor.get_model_info()
            return jsonify({
                'success': True,
                'model_info': info,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in model_info: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

# Create the application
app = create_app(os.environ.get('FLASK_ENV', 'development'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)