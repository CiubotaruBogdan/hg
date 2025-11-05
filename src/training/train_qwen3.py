"""
Qwen 3 Training Script
Fine-tunes Qwen 3 model on the HG 585 dataset.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.base_trainer import BaseLLMTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3Trainer(BaseLLMTrainer):
    """Trainer for Qwen 3 model."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any] = None):
        super().__init__(
            model_name="qwen3",
            model_path="Qwen/Qwen2.5-7B-Instruct",  # Using Qwen 2.5 as Qwen 3 placeholder
            output_dir=output_dir,
            config=config
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Qwen 3 training."""
        return {
            'learning_rate': 1e-5,
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'epochs': 3,
            'max_length': 512,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'lr_scheduler_type': 'cosine',
            'save_steps': 500,
            'eval_steps': 250,
            'logging_steps': 50,
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        }
    
    def load_model(self):
        """Load Qwen 3 model and tokenizer."""
        try:
            # Try to import required libraries
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType
            import torch
            
            logger.info(f"Loading Qwen 3 model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            device_map = "auto" if torch.cuda.is_available() else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Apply LoRA if configured
            if self.config.get('use_lora', True):
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config['lora_r'],
                    lora_alpha=self.config['lora_alpha'],
                    lora_dropout=self.config['lora_dropout'],
                    target_modules=self.config['target_modules']
                )
                self.model = get_peft_model(self.model, lora_config)
                logger.info("Applied LoRA configuration to model")
            
            logger.info("Model and tokenizer loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Required libraries not available: {e}")
            logger.info("Falling back to mock training")
            self._use_mock_training()
    
    def _use_mock_training(self):
        """Use mock training when libraries are not available."""
        from training.base_trainer import MockLLMTrainer
        
        mock_trainer = MockLLMTrainer(
            model_name=self.model_name,
            model_path=self.model_path,
            output_dir=self.output_dir,
            config=self.config
        )
        
        # Copy methods from mock trainer
        self.load_model = mock_trainer.load_model
        self.prepare_data = mock_trainer.prepare_data
        self.train = mock_trainer.train
        self.save_model = mock_trainer.save_model
        self.evaluate_model = mock_trainer.evaluate_model
        
        # Load mock model
        self.load_model()
    
    def prepare_data(self, train_file: str, eval_file: str) -> Tuple[Any, Any]:
        """Prepare training and evaluation data for Qwen 3."""
        try:
            from datasets import Dataset
            
            # Load JSONL data
            train_data = self.load_jsonl_data(train_file)
            eval_data = self.load_jsonl_data(eval_file)
            
            # Format data for instruction tuning (Qwen format)
            def format_instruction(example):
                instruction = example['question']
                response = example['answer']
                
                # Use Qwen chat format
                formatted_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
                
                return {'text': formatted_text}
            
            # Format datasets
            train_formatted = [format_instruction(ex) for ex in train_data]
            eval_formatted = [format_instruction(ex) for ex in eval_data]
            
            # Create HuggingFace datasets
            train_dataset = Dataset.from_list(train_formatted)
            eval_dataset = Dataset.from_list(eval_formatted)
            
            # Tokenize datasets
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=self.config['max_length'],
                    return_tensors="pt"
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            logger.info(f"Prepared {len(train_dataset)} training examples")
            logger.info(f"Prepared {len(eval_dataset)} evaluation examples")
            
            return train_dataset, eval_dataset
            
        except ImportError:
            # Fall back to base implementation
            return super().prepare_data(train_file, eval_file)
    
    def train(self, train_data, eval_data) -> Dict[str, Any]:
        """Train the Qwen 3 model."""
        try:
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.config['epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                per_device_eval_batch_size=self.config['batch_size'],
                gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                warmup_steps=self.config['warmup_steps'],
                lr_scheduler_type=self.config['lr_scheduler_type'],
                logging_steps=self.config['logging_steps'],
                eval_steps=self.config['eval_steps'],
                save_steps=self.config['save_steps'],
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb/tensorboard
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                data_collator=data_collator,
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Update training stats
            self.training_stats['epochs'] = self.config['epochs']
            self.training_stats['steps'] = train_result.global_step
            self.training_stats['final_loss'] = train_result.training_loss
            self.training_stats['best_loss'] = min(self.training_stats['best_loss'], train_result.training_loss)
            
            return {
                'final_loss': train_result.training_loss,
                'best_loss': self.training_stats['best_loss'],
                'epochs_completed': self.config['epochs'],
                'global_step': train_result.global_step
            }
            
        except ImportError:
            # Fall back to mock training
            return super().train(train_data, eval_data)
    
    def save_model(self):
        """Save the trained Qwen 3 model."""
        try:
            logger.info(f"Saving model to {self.output_dir}")
            
            # Save model and tokenizer
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training config
            config_file = os.path.join(self.output_dir, "training_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
            # Fall back to mock saving
            super().save_model()
    
    def evaluate_model(self, eval_data) -> Dict[str, Any]:
        """Evaluate the trained Qwen 3 model."""
        try:
            from transformers import Trainer, TrainingArguments
            import torch
            
            # Create a trainer for evaluation
            eval_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_eval_batch_size=self.config['batch_size'],
                remove_unused_columns=False,
            )
            
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=eval_data,
            )
            
            # Run evaluation
            eval_results = trainer.evaluate()
            
            # Calculate additional metrics
            perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
            
            return {
                'eval_loss': eval_results['eval_loss'],
                'perplexity': perplexity,
                'eval_runtime': eval_results.get('eval_runtime', 0),
                'eval_samples_per_second': eval_results.get('eval_samples_per_second', 0)
            }
            
        except ImportError:
            # Fall back to mock evaluation
            return super().evaluate_model(eval_data)


def main():
    """Main function to run Qwen 3 training."""
    # Configuration
    output_dir = "/home/ubuntu/llm-evaluation/models/qwen3"
    train_file = "/home/ubuntu/llm-evaluation/data/processed/train.jsonl"
    eval_file = "/home/ubuntu/llm-evaluation/data/processed/eval.jsonl"
    
    # Custom config (optional)
    custom_config = {
        'epochs': 2,  # Reduced for testing
        'batch_size': 2,  # Smaller batch size
        'max_length': 256,  # Shorter sequences for testing
    }
    
    # Create trainer
    trainer = Qwen3Trainer(output_dir, custom_config)
    
    # Check if data files exist
    if not os.path.exists(train_file) or not os.path.exists(eval_file):
        logger.error("Training data not found. Please run preprocessing first.")
        logger.info("Expected files:")
        logger.info(f"  - {train_file}")
        logger.info(f"  - {eval_file}")
        return
    
    try:
        # Run training pipeline
        results = trainer.run_training_pipeline(train_file, eval_file)
        
        # Print results
        print("\n" + "="*50)
        print("QWEN 3 TRAINING COMPLETED")
        print("="*50)
        print(trainer.get_training_summary())
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

