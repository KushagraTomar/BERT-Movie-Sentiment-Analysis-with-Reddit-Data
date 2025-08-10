# Movie Sentiment Analyzer

A production-ready movie sentiment analysis system using BERT and Reddit data. This application fetches movie discussions from Reddit and analyzes sentiment using a fine-tuned BERT model.

## Features

- **BERT-based Sentiment Analysis**: Fine-tuned BERT model for accurate movie sentiment classification
- **Reddit Data Integration**: Automatically fetches movie discussions from multiple movie-related subreddits
- **Web Interface**: Clean, responsive web interface for easy interaction
- **REST API**: Complete API for programmatic access
- **Production Ready**: Docker support, logging, error handling, and monitoring
- **Real-time Analysis**: Analyze both custom text and live Reddit discussions

## Quick Start

### Prerequisites

- Python 3.8+
- Reddit API credentials (optional, for Reddit data fetching)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd movie-sentiment-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials
   ```

4. **Train the BERT model**
   ```bash
   python src/model/bert_trainer.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Web Interface: http://localhost:5000
   - API Health Check: http://localhost:5000/api/health

## Reddit API Setup

To enable Reddit data fetching, you need to create a Reddit application:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Note down your `client_id` and `client_secret`
5. Add these to your `.env` file:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=MovieSentimentAnalyzer/1.0
   ```

## API Endpoints

### Health Check
```http
GET /api/health
```
Returns system status and component availability.

### Text Sentiment Analysis
```http
POST /api/predict
Content-Type: application/json

{
  "text": "This movie was absolutely amazing!"
}
```

### Movie Sentiment Analysis
```http
POST /api/analyze-movie
Content-Type: application/json

{
  "movie_name": "Avengers Endgame",
  "max_posts": 20,
  "max_comments_per_post": 30
}
```

### Model Information
```http
GET /api/model-info
```

### Train Model (Development Only)
```http
POST /api/train-model
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and run**
   ```bash
   docker-compose up --build
   ```

2. **Run in background**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image**
   ```bash
   docker build -t movie-sentiment-analyzer .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 \
     -e REDDIT_CLIENT_ID=your_client_id \
     -e REDDIT_CLIENT_SECRET=your_client_secret \
     movie-sentiment-analyzer
   ```

## Production Deployment

### Using Gunicorn

```bash
gunicorn --config gunicorn.conf.py app:app
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `PORT` | Server port | `5000` |
| `WORKERS` | Number of worker processes | `4` |
| `TIMEOUT` | Request timeout in seconds | `120` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `REDDIT_CLIENT_ID` | Reddit API client ID | - |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret | - |
| `REDDIT_USER_AGENT` | Reddit API user agent | `MovieSentimentAnalyzer/1.0` |
| `MODEL_PATH` | Path to BERT model | `./models/bert_sentiment_model` |
| `CACHE_DIR` | Cache directory | `./cache` |

## Project Structure

```
movie-sentiment-analyzer/
├── src/
│   ├── model/
│   │   ├── bert_trainer.py      # BERT model training
│   │   ├── bert_predictor.py    # BERT model inference
│   │   └── __init__.py
│   ├── data/
│   │   ├── reddit_scraper.py    # Reddit data fetching
│   │   └── __init__.py
│   └── __init__.py
├── templates/
│   └── index.html               # Web interface
├── models/                      # Trained models directory
├── data/                        # Data storage
├── logs/                        # Application logs
├── cache/                       # Cache directory
├── app.py                       # Flask application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── gunicorn.conf.py            # Gunicorn configuration
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Model Training

The system uses a fine-tuned BERT model for sentiment analysis. The training process:

1. **Data Preparation**: Uses the IMDB movie review dataset (50,000 reviews) for training
2. **Model Fine-tuning**: Fine-tunes `bert-base-uncased` for 3-class sentiment classification
3. **Label Mapping**: 0 = Negative, 1 = Neutral, 2 = Positive
4. **Evaluation**: Provides accuracy metrics and saves the trained model
5. **Inference**: Loads the trained model for real-time predictions

### Dataset Details

- **Primary Dataset**: IMDB Movie Review Dataset (automatically downloaded)
- **Size**: 10,000 samples (subset for faster training, configurable)
- **Classes**: 3-class sentiment (negative, neutral, positive)
- **Fallback**: Sample dataset if IMDB loading fails

### Custom Training Data

To train with your own data, modify the `load_imdb_dataset()` method in `bert_trainer.py` or provide a pandas DataFrame with `text` and `label` columns to the `full_training_pipeline()` method.

## Monitoring and Logging

- **Application Logs**: Stored in `./logs/app.log`
- **Training Logs**: Stored in `./logs/training.log`
- **Health Check**: Available at `/api/health`
- **Metrics**: Response times, error rates, and model performance

## Error Handling

The application includes comprehensive error handling:

- **API Errors**: Proper HTTP status codes and error messages
- **Model Errors**: Graceful handling of model loading and prediction failures
- **Reddit API Errors**: Fallback behavior when Reddit API is unavailable
- **Validation**: Input validation for all endpoints

## Performance Considerations

- **Model Loading**: Model is loaded once at startup for better performance
- **Caching**: Results can be cached to improve response times
- **Rate Limiting**: Built-in rate limiting for Reddit API calls
- **Batch Processing**: Support for batch sentiment analysis
- **Memory Management**: Efficient memory usage with proper cleanup

## Security

- **Input Validation**: All inputs are validated and sanitized
- **Environment Variables**: Sensitive data stored in environment variables
- **CORS**: Configurable CORS settings
- **Non-root User**: Docker container runs as non-root user
- **Health Checks**: Built-in health monitoring

## Troubleshooting

### Common Issues

1. **Model not found**: Run the training script first
   ```bash
   python src/model/bert_trainer.py
   ```

2. **Reddit API errors**: Check your API credentials in `.env`

3. **Memory issues**: Reduce batch size or use a smaller model

4. **Port conflicts**: Change the port in environment variables

### Logs

Check the logs for detailed error information:
```bash
tail -f logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the logs for error details
- Review the API documentation

## Changelog

### v1.0.0
- Initial release
- BERT-based sentiment analysis
- Reddit data integration
- Web interface
- Docker support
- Production-ready configuration