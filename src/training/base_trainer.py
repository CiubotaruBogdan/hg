"""
Base trainer class for LLM fine-tuning.
Provides common functionality for all model trainers.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLMTrainer(ABC):
    """Base class for LLM trainers with GPU optimization."""
    
    def __init__(self, 
                 model_name: str,
                 model_path: str,
                 output_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config or self._get_default_config()
        
        # GPU setup
        self.device = self._setup_device()
        self.gpu_config = self._get_gpu_config()
        
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
            'final_loss': None,
            'device_used': str(self.device),
            'gpu_memory_used': None
        }
        
        logger.info(f"Trainer initialized for {model_name}")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _setup_device(self) -> torch.device:
        """Setup and optimize device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
            # GPU optimization settings
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def _get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU-optimized configuration."""
        if not torch.cuda.is_available():
            return {
                'mixed_precision': False,
                'gradient_checkpointing': False,
                'max_batch_size': 1,
                'dataloader_num_workers': 0
            }
        
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Optimize based on GPU memory
        if gpu_memory_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
            config = {
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'max_batch_size': 8,
                'dataloader_num_workers': 4,
                'pin_memory': True
            }
        elif gpu_memory_gb >= 12:  # Mid-range GPU (RTX 4070 Ti, etc.)
            config = {
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'max_batch_size': 4,
                'dataloader_num_workers': 2,
                'pin_memory': True
            }
        elif gpu_memory_gb >= 8:  # Entry-level GPU (RTX 4060 Ti, etc.)
            config = {
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'max_batch_size': 2,
                'dataloader_num_workers': 1,
                'pin_memory': True
            }
        else:  # Low VRAM GPU
            config = {
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'max_batch_size': 1,
                'dataloader_num_workers': 0,
                'pin_memory': False
            }
        
        logger.info(f"GPU Config: {config}")
        return config
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        return {
            'allocated': round(allocated, 2),
            'reserved': round(reserved, 2),
            'free': round(free, 2),
            'total': round(total, 2)
        }
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def log_gpu_stats(self):
        """Log current GPU statistics."""
        if torch.cuda.is_available():
            memory = self.get_gpu_memory_usage()
            logger.info(f"GPU Memory - Allocated: {memory['allocated']:.2f}GB, "
                       f"Reserved: {memory['reserved']:.2f}GB, "
                       f"Free: {memory['free']:.2f}GB")
            
            # Log temperature if available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temp = result.stdout.strip()
                    logger.info(f"GPU Temperature: {temp}°C")
            except Exception:
                pass  # Temperature monitoring not critical
    
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
    
    def export_trained_model(self, export_path: str = None) -> bool:
        """
        Export the trained model for future use.
        
        Args:
            export_path: Optional custom export path
            
        Returns:
            bool: True if export successful
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                logger.error("No trained model available for export")
                return False
            
            # Default export path
            if export_path is None:
                export_path = f"models/{self.model_name}_trained_export"
            
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Exporting trained model to: {export_path}")
            
            # For mock trainer, create mock export
            if isinstance(self, MockLLMTrainer):
                return self._mock_export(export_dir)
            
            # Save the model and tokenizer (for real trainers)
            try:
                self.model.save_pretrained(export_dir / "model")
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    self.tokenizer.save_pretrained(export_dir / "tokenizer")
                
                # Save LoRA adapters if available
                if hasattr(self.model, 'peft_config'):
                    logger.info("Saving LoRA adapters...")
                    self.model.save_pretrained(export_dir / "lora_adapters")
            except Exception as e:
                logger.warning(f"Could not save model files: {e}")
                return self._mock_export(export_dir)
            
            # Save training metadata
            export_metadata = {
                "model_name": self.model_name,
                "base_model": getattr(self, 'model_id', 'unknown'),
                "export_date": str(datetime.now()),
                "training_completed": True,
                "export_type": "full_model_with_adapters",
                "lora_config": getattr(self, 'lora_config', {}),
                "training_args": getattr(self, 'training_args_dict', {}),
                "usage_instructions": {
                    "load_model": f"AutoModelForCausalLM.from_pretrained('{export_dir / 'model'}')",
                    "load_tokenizer": f"AutoTokenizer.from_pretrained('{export_dir / 'tokenizer'}')",
                    "load_lora": f"PeftModel.from_pretrained(base_model, '{export_dir / 'lora_adapters'}')"
                }
            }
            
            with open(export_dir / "export_metadata.json", "w") as f:
                json.dump(export_metadata, f, indent=2)
            
            # Create usage example script
            self._create_usage_script(export_dir)
            
            # Calculate export size
            export_size = self._calculate_directory_size(export_dir)
            
            logger.info(f"✅ Model export completed successfully!")
            logger.info(f"📁 Export location: {export_dir}")
            logger.info(f"💾 Export size: {export_size:.2f} GB")
            logger.info(f"📝 Usage example: {export_dir / 'usage_example.py'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model export failed: {e}")
            return False
    
    def _mock_export(self, export_dir: Path) -> bool:
        """Create mock export for testing."""
        logger.info("Creating mock model export...")
        
        # Create mock model files
        (export_dir / "model").mkdir(exist_ok=True)
        (export_dir / "tokenizer").mkdir(exist_ok=True)
        (export_dir / "lora_adapters").mkdir(exist_ok=True)
        
        with open(export_dir / "model" / "pytorch_model.bin", "w") as f:
            f.write(f"Mock trained model weights for {self.model_name}")
        
        with open(export_dir / "tokenizer" / "tokenizer.json", "w") as f:
            f.write(f"Mock tokenizer for {self.model_name}")
        
        with open(export_dir / "lora_adapters" / "adapter_model.bin", "w") as f:
            f.write(f"Mock LoRA adapters for {self.model_name}")
        
        # Create metadata
        export_metadata = {
            "model_name": self.model_name,
            "export_date": str(datetime.now()),
            "export_type": "mock_export",
            "note": "This is a mock export for testing purposes"
        }
        
        with open(export_dir / "export_metadata.json", "w") as f:
            json.dump(export_metadata, f, indent=2)
        
        self._create_usage_script(export_dir)
        
        return True
    
    def _create_usage_script(self, export_dir: Path):
        """Create usage example script."""
        usage_script = f'''#!/usr/bin/env python3
"""
Usage example for exported {self.model_name} model
Generated on {datetime.now()}
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_trained_model():
    """Load the exported trained model."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("{export_dir / 'tokenizer'}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "{export_dir / 'model'}",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    # Load LoRA adapters if available
    try:
        model = PeftModel.from_pretrained(model, "{export_dir / 'lora_adapters'}")
        print("✅ LoRA adapters loaded successfully")
    except Exception as e:
        print(f"⚠️  LoRA adapters not found or failed to load: {{e}}")
    
    return model, tokenizer

def generate_response(prompt: str, max_length: int = 512):
    """Generate response using the trained model."""
    
    model, tokenizer = load_trained_model()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Example usage
    prompt = "What are the classification levels mentioned in HG 585?"
    response = generate_response(prompt)
    print(f"Prompt: {{prompt}}")
    print(f"Response: {{response}}")
'''
        
        with open(export_dir / "usage_example.py", "w") as f:
            f.write(usage_script)
    
    def _calculate_directory_size(self, path: Path) -> float:
        """Calculate directory size in GB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue
            return total_size / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
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

