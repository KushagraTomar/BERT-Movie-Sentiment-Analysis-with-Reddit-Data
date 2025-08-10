import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
from typing import List, Dict, Union, Tuple
import warnings
warnings.filterwarnings("ignore")

class MovieSentimentPredictor:
    """
    BERT-based sentiment predictor for movie reviews
    """
    
    def __init__(self, model_path: str = "./models/bert_sentiment_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model path {self.model_path} does not exist!")
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r") as f:
                    self.label_mapping = json.load(f)
            else:
                # Default mapping
                self.label_mapping = {0: "negative", 1: "positive"}
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction"""
        # Basic text cleaning
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded! Call load_model() first.")
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Get all probabilities
            probabilities = {
                self.label_mapping[str(i)]: float(predictions[0][i].item())
                for i in range(len(self.label_mapping))
            }
            
            result = {
                "text": text,
                "predicted_sentiment": self.label_mapping[str(predicted_class)],
                "confidence": float(confidence),
                "probabilities": probabilities
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict]]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded! Call load_model() first.")
            return [{"error": "Model not loaded"} for _ in texts]
        
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def analyze_sentiment_distribution(self, texts: List[str]) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze sentiment distribution across multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary containing distribution analysis
        """
        predictions = self.predict_batch(texts)
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0}
        total_confidence = 0
        valid_predictions = 0
        
        for pred in predictions:
            if "error" not in pred:
                sentiment = pred["predicted_sentiment"]
                sentiment_counts[sentiment] += 1
                total_confidence += pred["confidence"]
                valid_predictions += 1
        
        if valid_predictions == 0:
            return {"error": "No valid predictions"}
        
        # Calculate percentages
        total_texts = len(texts)
        sentiment_percentages = {
            sentiment: (count / total_texts) * 100
            for sentiment, count in sentiment_counts.items()
        }
        
        # Overall sentiment
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        avg_confidence = total_confidence / valid_predictions if valid_predictions > 0 else 0
        
        return {
            "total_texts": total_texts,
            "valid_predictions": valid_predictions,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "overall_sentiment": max_sentiment,
            "average_confidence": avg_confidence,
            "detailed_predictions": predictions
        }
    
    def get_model_info(self) -> Dict[str, Union[str, Dict]]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "label_mapping": self.label_mapping,
            "model_type": self.model.__class__.__name__,
            "tokenizer_type": self.tokenizer.__class__.__name__
        }

# Utility function for quick predictions
def quick_predict(text: str, model_path: str = "./models/bert_sentiment_model") -> Dict:
    """
    Quick prediction function for single text
    
    Args:
        text: Input text
        model_path: Path to the trained model
        
    Returns:
        Prediction result
    """
    predictor = MovieSentimentPredictor(model_path)
    if predictor.load_model():
        return predictor.predict_single(text)
    else:
        return {"error": "Failed to load model"}

def main():
    """Test the predictor"""
    # Test texts
    test_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film, waste of time and money.",
        "It was okay, nothing special but watchable.",
        "One of the best movies I've ever seen!",
        "Boring and predictable storyline."
    ]
    
    # Initialize predictor
    predictor = MovieSentimentPredictor()
    
    if predictor.load_model():
        print("Model loaded successfully!")
        
        # Test single prediction
        print("\n=== Single Predictions ===")
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result.get('predicted_sentiment', 'Error')}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            print("-" * 50)
        
        # Test batch analysis
        print("\n=== Sentiment Distribution Analysis ===")
        analysis = predictor.analyze_sentiment_distribution(test_texts)
        print(f"Overall Sentiment: {analysis.get('overall_sentiment', 'Unknown')}")
        print(f"Sentiment Distribution: {analysis.get('sentiment_percentages', {})}")
        print(f"Average Confidence: {analysis.get('average_confidence', 0):.3f}")
        
    else:
        print("Failed to load model. Please train the model first.")

if __name__ == "__main__":
    main()