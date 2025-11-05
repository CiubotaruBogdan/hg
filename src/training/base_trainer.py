"""
Base trainer class for LLM fine-tuning.
Provides common functionality for all model trainers.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMTrainer(ABC):
    """Base class for LLM trainers."""
    
    def __init__(self, 
                 model_name: str,
                 model_path: str,
                 output_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config or self._get_default_config()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training state
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'total_time': None,
            'epochs': 0,
            'steps': 0,
            'best_loss': float('inf'),
            'final_loss': None
        }
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the pre-trained model."""
        pass
    
    @abstractmethod
    def prepare_data(self, train_file: str, eval_file: str):
        """Prepare training and evaluation data."""
        pass
    
    @abstractmethod
    def train(self, train_data, eval_data) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def save_model(self):
        """Save the trained model."""
        pass
    
    @abstractmethod
    def evaluate_model(self, eval_data) -> Dict[str, Any]:
        """Evaluate the model."""
        pass
    
    def run_training_pipeline(self, train_file: str, eval_file: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            train_file (str): Path to training data file
            eval_file (str): Path to evaluation data file
            
        Returns:
            Dict containing training results and statistics
        """
        logger.info(f"Starting training pipeline for {self.model_name}")
        
        try:
            # Record start time
            self.training_stats['start_time'] = time.time()
            
            # Step 1: Load model
            logger.info("Loading model...")
            self.load_model()
            
            # Step 2: Prepare data
            logger.info("Preparing data...")
            train_data, eval_data = self.prepare_data(train_file, eval_file)
            
            # Step 3: Train model
            logger.info("Training model...")
            training_results = self.train(train_data, eval_data)
            
            # Step 4: Evaluate model
            logger.info("Evaluating model...")
            eval_results = self.evaluate_model(eval_data)
            
            # Step 5: Save model
            logger.info("Saving model...")
            self.save_model()
            
            # Record end time
            self.training_stats['end_time'] = time.time()
            self.training_stats['total_time'] = self.training_stats['end_time'] - self.training_stats['start_time']
            
            # Compile results
            results = {
                'model_name': self.model_name,
                'training_config': self.config,
                'training_stats': self.training_stats,
                'training_results': training_results,
                'evaluation_results': eval_results,
                'output_directory': self.output_dir
            }
            
            # Save results
            self._save_results(results)
            
            logger.info(f"Training pipeline completed for {self.model_name}")
            logger.info(f"Total training time: {self.training_stats['total_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed for {self.model_name}: {str(e)}")
            raise
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_file = os.path.join(self.output_dir, f"{self.model_name}_training_results.json")
        
        # Convert any non-serializable objects to strings
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training results saved to: {results_file}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def get_training_summary(self) -> str:
        """Get a summary of training results."""
        summary = f"""
Training Summary for {self.model_name}:
- Total training time: {self.training_stats.get('total_time', 0):.2f} seconds
- Epochs completed: {self.training_stats.get('epochs', 0)}
- Training steps: {self.training_stats.get('steps', 0)}
- Best loss: {self.training_stats.get('best_loss', 'N/A')}
- Final loss: {self.training_stats.get('final_loss', 'N/A')}
- Output directory: {self.output_dir}
"""
        return summary.strip()


class MockLLMTrainer(BaseLLMTrainer):
    """Mock trainer for testing purposes when actual models are not available."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'learning_rate': 2e-5,
            'batch_size': 4,
            'epochs': 3,
            'max_length': 512,
            'warmup_steps': 100
        }
    
    def load_model(self):
        """Mock model loading."""
        logger.info(f"Mock: Loading {self.model_name} model")
        time.sleep(1)  # Simulate loading time
        self.model = f"mock_{self.model_name}_model"
        self.tokenizer = f"mock_{self.model_name}_tokenizer"
    
    def prepare_data(self, train_file: str, eval_file: str):
        """Mock data preparation."""
        logger.info("Mock: Preparing training data")
        train_data = self.load_jsonl_data(train_file)
        eval_data = self.load_jsonl_data(eval_file)
        
        logger.info(f"Loaded {len(train_data)} training examples")
        logger.info(f"Loaded {len(eval_data)} evaluation examples")
        
        return train_data, eval_data
    
    def train(self, train_data, eval_data) -> Dict[str, Any]:
        """Mock training."""
        logger.info("Mock: Training model...")
        
        # Simulate training
        epochs = self.config['epochs']
        for epoch in range(epochs):
            logger.info(f"Mock: Epoch {epoch + 1}/{epochs}")
            time.sleep(2)  # Simulate training time
            
            # Mock loss values
            loss = 2.5 - (epoch * 0.3) + (0.1 * (epoch % 2))
            self.training_stats['best_loss'] = min(self.training_stats['best_loss'], loss)
            
        self.training_stats['epochs'] = epochs
        self.training_stats['steps'] = epochs * len(train_data) // self.config['batch_size']
        self.training_stats['final_loss'] = loss
        
        return {
            'final_loss': loss,
            'best_loss': self.training_stats['best_loss'],
            'epochs_completed': epochs
        }
    
    def save_model(self):
        """Mock model saving."""
        logger.info(f"Mock: Saving model to {self.output_dir}")
        
        # Create mock model files
        model_file = os.path.join(self.output_dir, "pytorch_model.bin")
        config_file = os.path.join(self.output_dir, "config.json")
        
        with open(model_file, 'w') as f:
            f.write(f"Mock model weights for {self.model_name}")
        
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def evaluate_model(self, eval_data) -> Dict[str, Any]:
        """Mock evaluation."""
        logger.info("Mock: Evaluating model...")
        time.sleep(1)
        
        # Mock evaluation metrics
        return {
            'accuracy': 0.75 + (0.1 * hash(self.model_name) % 10) / 100,
            'bleu_score': 0.65 + (0.1 * hash(self.model_name) % 15) / 100,
            'rouge_l': 0.70 + (0.1 * hash(self.model_name) % 12) / 100,
            'perplexity': 15.0 - (hash(self.model_name) % 5),
            'eval_loss': 1.2 + (0.1 * hash(self.model_name) % 8) / 100
        }


def main():
    """Test the base trainer."""
    # Test with mock trainer
    trainer = MockLLMTrainer(
        model_name="test_model",
        model_path="mock/path",
        output_dir="/home/ubuntu/llm-evaluation/models/test"
    )
    
    print("Testing base trainer functionality...")
    print(trainer.get_training_summary())


if __name__ == "__main__":
    main()

