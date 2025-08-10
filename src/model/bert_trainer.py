import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from loguru import logger
import json
import gc
from typing import Dict, List, Tuple, Optional

class MovieSentimentBERTTrainer:
    """
    BERT-based sentiment analysis trainer for movie reviews
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # GPU memory management and device selection
        self.device = self._setup_device()
        
        # Create necessary directories
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        logger.add("./logs/training.log", rotation="10 MB")
    
    def _setup_device(self):
        """Setup device with proper GPU memory management"""
        if torch.cuda.is_available():
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                gc.collect()
                
                # Test GPU availability
                device = torch.device("cuda")
                test_tensor = torch.tensor([1.0]).to(device)
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"GPU available: {torch.cuda.get_device_name()}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                return device
                
            except Exception as e:
                logger.warning(f"GPU test failed: {e}")
                logger.info("Falling back to CPU")
                return torch.device("cpu")
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    def load_tokenizer_and_model(self, num_labels: int = 2):
        """Load tokenizer and model"""
        try:
            # Clear memory before loading
            self._clear_gpu_memory()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Try to move model to device with error handling
            try:
                self.model.to(self.device)
                logger.info(f"Loaded {self.model_name} with {num_labels} labels on {self.device}")
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning(f"GPU error: {e}")
                    logger.info("Falling back to CPU")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_imdb_dataset(self) -> pd.DataFrame:
        """
        Load the IMDB movie review dataset for training
        """
        try:
            
            logger.info("Loading IMDB dataset...")
            
            # Load IMDB dataset
            dataset = load_dataset("imdb")
            
            train_data = dataset['train']
            test_data = dataset['test']
            
            # Combine train and test for more data (we'll split again later)
            texts = list(train_data['text']) + list(test_data['text'])
            labels = list(train_data['label']) + list(test_data['label'])
            
            # Create DataFrame
            df = pd.DataFrame({
                'text': texts,
                'label': labels  # 0 = negative, 1 = positive
            })
            
            # Take a subset for faster training (you can increase this)
            df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
            
            # Shuffle the dataset 
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Loaded IMDB dataset with {len(df)} samples")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading IMDB dataset: {e}")
            raise RuntimeError("Failed to load IMDB dataset. Cannot proceed with training.") from e
    
    def tokenize_data(self, texts: List[str]) -> Dict:
        """Tokenize the input texts"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label'] # same label distribution in train and val
        )
        
        # Tokenize
        train_encodings = self.tokenize_data(train_texts)
        val_encodings = self.tokenize_data(val_texts)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # Print text report
        print(classification_report(labels, predictions))
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset, 
                   output_dir: str = "./models/bert_sentiment_model"):
        """Train the BERT model"""
        
        # Training arguments - GPU optimized with memory efficiency
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            
            # Memory optimization: smaller batch size + gradient accumulation
            per_device_train_batch_size=1,  # Very small batch size for GPU memory
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch
            
            # FP16 training for memory efficiency
            fp16=True,  # Use half precision to reduce memory usage
            fp16_opt_level="O1",  # Conservative FP16 optimization
            
            # Memory management
            dataloader_pin_memory=False,  # Reduce memory usage
            dataloader_num_workers=0,  # Avoid multiprocessing memory overhead
            
            # Training settings
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=2e-5,  # Standard BERT learning rate
            
            # Logging and evaluation
            logging_dir='./logs',
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            
            # Additional memory optimizations
            remove_unused_columns=True,
            prediction_loss_only=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting model training...")
        
        # Clear memory before training
        self._clear_gpu_memory()
        
        try:
            # Train the model
            self.trainer.train()
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.error(f"GPU training failed: {e}")
                logger.info("Try reducing batch size or using CPU")
                raise RuntimeError("GPU training failed. Consider using CPU or reducing batch size.") from e
            else:
                raise
        
        # Save the model
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        label_mapping = {0: "negative", 1: "positive"}
        with open(f"{output_dir}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f)
        
        logger.info(f"Model saved to {output_dir}")
        
        return self.trainer
    
    def evaluate_model(self, val_dataset: Dataset):
        """Evaluate the trained model"""
        if self.trainer is None:
            logger.error("Model not trained yet!")
            return None
        
        # Evaluate
        eval_results = self.trainer.evaluate(val_dataset)
        logger.info(f"Evaluation results: {eval_results}")
        
        return eval_results
    
    def full_training_pipeline(self, custom_data: Optional[pd.DataFrame] = None):
        """Complete training pipeline"""
        try:
            # Load model and tokenizer
            self.load_tokenizer_and_model(num_labels=2)
            
            # Try to load IMDB dataset 
            df = self.load_imdb_dataset()
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_dataset(df)
            
            # Train model
            trainer = self.train_model(train_dataset, val_dataset)
            
            # Evaluate model
            eval_results = self.evaluate_model(val_dataset)
            
            logger.info("Training pipeline completed successfully!")
            return trainer, eval_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main function to run training"""
    trainer = MovieSentimentBERTTrainer()
    trainer.full_training_pipeline()

if __name__ == "__main__":
    main()