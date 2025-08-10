# Movie Sentiment Analyzer - Usage Guide

## Quick Start

### 1. Setup and Installation

```bash
# Clone or navigate to the project directory
cd movie-sentiment-analyzer

# Run the automated setup script
./start.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Train the BERT model
- Start the application

### 2. Manual Setup (Alternative)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Reddit API credentials

# Train the model
python3 src/model/bert_trainer.py

# Start the application
python3 app.py
```

### 3. Docker Setup

```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or using Docker directly
docker build -t movie-sentiment-analyzer .
docker run -p 5000:5000 movie-sentiment-analyzer
```

## Using the Application

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. You'll see two main features:

#### Movie Sentiment Analysis
- Enter a movie name (e.g., "Avengers Endgame", "The Dark Knight")
- Select the number of posts to analyze
- Click "Analyze Movie Sentiment"
- The system will:
  - Fetch discussions from Reddit
  - Analyze sentiment using BERT
  - Show overall sentiment and distribution

#### Direct Text Analysis
- Enter any text about a movie
- Click "Analyze Text Sentiment"
- Get immediate sentiment classification

### API Usage

#### Health Check
```bash
curl http://localhost:5000/api/health
```

#### Analyze Text
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'
```

#### Analyze Movie
```bash
curl -X POST http://localhost:5000/api/analyze-movie \
  -H "Content-Type: application/json" \
  -d '{"movie_name": "Avengers Endgame", "max_posts": 20}'
```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Reddit API (optional but recommended)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=MovieSentimentAnalyzer/1.0

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your_secret_key

# Model settings
MODEL_PATH=./models/bert_sentiment_model
CACHE_DIR=./cache
LOG_LEVEL=INFO
```

### Reddit API Setup

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App"
3. Choose "script" type
4. Copy client ID and secret to `.env`

## Features

### BERT Model
- Fine-tuned `bert-base-uncased` for movie sentiment
- 3-class classification: Positive, Neutral, Negative
- Trained on movie review data
- High accuracy sentiment prediction

### Reddit Integration
- Searches multiple movie subreddits
- Fetches posts and comments
- Filters relevant discussions
- Rate-limited to respect API limits

### Web Interface
- Responsive design
- Real-time analysis
- Visual sentiment distribution
- Sample text display
- System status monitoring

### Production Features
- Docker containerization
- Gunicorn WSGI server
- Comprehensive logging
- Error handling
- Health monitoring
- Configurable scaling

## Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   python3 src/model/bert_trainer.py
   ```

2. **Reddit API errors**
   - Check credentials in `.env`
   - Ensure Reddit app is configured correctly

3. **Memory issues**
   - Reduce batch size in training
   - Use smaller model if needed

4. **Port conflicts**
   - Change PORT in environment variables
   - Use different port: `PORT=8000 python3 app.py`

### Logs

Check application logs:
```bash
tail -f logs/app.log
```

Check training logs:
```bash
tail -f logs/training.log
```

## Performance Tips

1. **Model Loading**: Model loads once at startup for better performance
2. **Caching**: Results can be cached for repeated queries
3. **Batch Processing**: Use batch endpoints for multiple texts
4. **Rate Limiting**: Built-in rate limiting for Reddit API

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Structure
```
src/
├── model/
│   ├── bert_trainer.py    # Model training
│   └── bert_predictor.py  # Model inference
└── data/
    └── reddit_scraper.py  # Reddit data fetching
```

### Adding Custom Data

Modify `create_sample_dataset()` in `bert_trainer.py` or provide your own DataFrame with `text` and `label` columns.

## Deployment

### Production Deployment

1. **Using Gunicorn**
   ```bash
   gunicorn --config gunicorn.conf.py app:app
   ```

2. **Using Docker**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Environment Variables for Production**
   ```env
   FLASK_ENV=production
   WORKERS=4
   TIMEOUT=120
   ```

### Scaling

- Increase `WORKERS` for more concurrent requests
- Use load balancer for multiple instances
- Add Redis for caching (uncomment in docker-compose.yml)
- Add PostgreSQL for data persistence

## Support

- Check logs for detailed error information
- Review API documentation in README.md
- Create issues for bugs or feature requests
- Ensure all dependencies are installed correctly

## Examples

### Successful Movie Analysis
```json
{
  "success": true,
  "movie_name": "Avengers Endgame",
  "total_texts": 45,
  "sentiment_analysis": {
    "overall_sentiment": "positive",
    "sentiment_distribution": {
      "positive": 67.8,
      "neutral": 22.2,
      "negative": 10.0
    },
    "average_confidence": 0.89
  }
}
```

### Text Analysis Result
```json
{
  "success": true,
  "result": {
    "text": "This movie was absolutely amazing!",
    "predicted_sentiment": "positive",
    "confidence": 0.95,
    "probabilities": {
      "positive": 0.95,
      "neutral": 0.04,
      "negative": 0.01
    }
  }
}