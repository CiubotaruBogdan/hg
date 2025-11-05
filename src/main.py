"""
LLM Training and Evaluation Main Application
Command-line interface for the complete LLM training and evaluation pipeline.
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import PreprocessingPipeline
from training import (
    Llama3Trainer, Qwen3Trainer, DeepSeekTrainer, 
    Gemma3Trainer, GPTOSSTrainer
)
from evaluation import LLMEvaluator
from evaluation.visualizer import ResultsVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMEvaluationApp:
    """Main application class for LLM training and evaluation."""
    
    def __init__(self, base_dir: str = "/home/ubuntu/llm-evaluation"):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.models_dir = os.path.join(base_dir, "models")
        self.results_dir = os.path.join(base_dir, "results")
        
        # Model configurations
        self.models_config = {
            "llama3": {
                "trainer_class": Llama3Trainer,
                "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "description": "Meta's Llama 3.1 - General purpose flagship model"
            },
            "qwen3": {
                "trainer_class": Qwen3Trainer,
                "model_path": "Qwen/Qwen2.5-7B-Instruct",
                "description": "Alibaba's Qwen 3 - Multilingual powerhouse"
            },
            "deepseek": {
                "trainer_class": DeepSeekTrainer,
                "model_path": "deepseek-ai/deepseek-llm-7b-chat",
                "description": "DeepSeek-V2 - Specialized reasoning model"
            },
            "gemma3": {
                "trainer_class": Gemma3Trainer,
                "model_path": "google/gemma-2-9b-it",
                "description": "Google's Gemma 3 - Latest open-source model"
            },
            "gpt_oss": {
                "trainer_class": GPTOSSTrainer,
                "model_path": "gpt-oss:20b",
                "description": "GPT-OSS 20B - Open-source GPT model via Ollama"
            }
        }
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def run_preprocessing(self, input_document: str, force: bool = False) -> bool:
        """
        Run the preprocessing pipeline.
        
        Args:
            input_document (str): Path to input document
            force (bool): Force reprocessing even if data exists
            
        Returns:
            bool: Success status
        """
        logger.info("Starting preprocessing pipeline...")
        
        processed_dir = os.path.join(self.data_dir, "processed")
        train_file = os.path.join(processed_dir, "train.jsonl")
        eval_file = os.path.join(processed_dir, "eval.jsonl")
        
        # Check if preprocessing already done
        if not force and os.path.exists(train_file) and os.path.exists(eval_file):
            logger.info("Preprocessed data already exists. Use --force to reprocess.")
            return True
        
        try:
            # Create preprocessing pipeline
            pipeline = PreprocessingPipeline(
                max_chunk_size=512,
                overlap_size=50,
                train_ratio=0.8,
                qa_pairs_per_chunk=2
            )
            
            # Run preprocessing
            results = pipeline.process_document(input_document, processed_dir)
            
            logger.info("Preprocessing completed successfully!")
            logger.info(f"Created {results['statistics']['train_examples']} training examples")
            logger.info(f"Created {results['statistics']['eval_examples']} evaluation examples")
            
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False
    
    def run_training(self, models: Optional[List[str]] = None, config_overrides: Optional[Dict] = None) -> bool:
        """
        Run training for specified models.
        
        Args:
            models (List[str], optional): List of model names to train
            config_overrides (Dict, optional): Configuration overrides
            
        Returns:
            bool: Success status
        """
        if models is None:
            models = list(self.models_config.keys())
        
        logger.info(f"Starting training for models: {', '.join(models)}")
        
        # Check if training data exists
        train_file = os.path.join(self.data_dir, "processed", "train.jsonl")
        eval_file = os.path.join(self.data_dir, "processed", "eval.jsonl")
        
        if not os.path.exists(train_file) or not os.path.exists(eval_file):
            logger.error("Training data not found. Please run preprocessing first.")
            return False
        
        success_count = 0
        
        for model_name in models:
            if model_name not in self.models_config:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                logger.info(f"Training {model_name}...")
                
                # Get model configuration
                model_config = self.models_config[model_name]
                trainer_class = model_config["trainer_class"]
                
                # Create output directory
                model_output_dir = os.path.join(self.models_dir, model_name)
                
                # Apply configuration overrides
                training_config = config_overrides or {}
                
                # Create trainer
                trainer = trainer_class(model_output_dir, training_config)
                
                # Run training
                results = trainer.run_training_pipeline(train_file, eval_file)
                
                logger.info(f"Training completed for {model_name}")
                logger.info(trainer.get_training_summary())
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                continue
        
        logger.info(f"Training completed for {success_count}/{len(models)} models")
        return success_count > 0
    
    def run_evaluation(self, 
                      models: Optional[List[str]] = None,
                      stages: Optional[List[str]] = None,
                      create_visualizations: bool = True) -> bool:
        """
        Run evaluation for specified models and stages.
        
        Args:
            models (List[str], optional): List of model names to evaluate
            stages (List[str], optional): Evaluation stages
            create_visualizations (bool): Whether to create visualizations
            
        Returns:
            bool: Success status
        """
        if models is None:
            models = list(self.models_config.keys())
        
        if stages is None:
            stages = ["before_training", "after_training"]
        
        logger.info(f"Starting evaluation for models: {', '.join(models)}")
        logger.info(f"Evaluation stages: {', '.join(stages)}")
        
        # Check if evaluation data exists
        eval_file = os.path.join(self.data_dir, "processed", "eval.jsonl")
        if not os.path.exists(eval_file):
            logger.error("Evaluation data not found. Please run preprocessing first.")
            return False
        
        try:
            # Create evaluator
            evaluator = LLMEvaluator(self.results_dir)
            
            # Prepare models configuration for evaluation
            eval_models_config = {}
            for model_name in models:
                if model_name in self.models_config:
                    eval_models_config[model_name] = self.models_config[model_name]["model_path"]
            
            # Run evaluation
            results = evaluator.evaluate_all_models(
                eval_models_config,
                eval_file,
                stages
            )
            
            logger.info("Evaluation completed successfully!")
            
            # Find best model
            best_model, best_score = evaluator.get_best_model()
            if best_model:
                logger.info(f"Best performing model: {best_model} (F1: {best_score:.4f})")
            
            # Create visualizations
            if create_visualizations:
                self.create_visualizations()
            
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return False
    
    def create_visualizations(self) -> bool:
        """Create visualization charts from evaluation results."""
        logger.info("Creating visualizations...")
        
        results_file = os.path.join(self.results_dir, "evaluation_results.json")
        if not os.path.exists(results_file):
            logger.error("Evaluation results not found. Please run evaluation first.")
            return False
        
        try:
            viz_dir = os.path.join(self.results_dir, "visualizations")
            visualizer = ResultsVisualizer(results_file, viz_dir)
            
            visualizer.create_all_visualizations()
            visualizer.generate_summary_report()
            
            logger.info(f"Visualizations created in: {viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return False
    
    def run_full_pipeline(self, 
                         input_document: str,
                         models: Optional[List[str]] = None,
                         config_overrides: Optional[Dict] = None) -> bool:
        """
        Run the complete pipeline: preprocessing, training, and evaluation.
        
        Args:
            input_document (str): Path to input document
            models (List[str], optional): List of model names
            config_overrides (Dict, optional): Configuration overrides
            
        Returns:
            bool: Success status
        """
        logger.info("Starting full LLM evaluation pipeline...")
        start_time = time.time()
        
        # Step 1: Preprocessing
        if not self.run_preprocessing(input_document):
            logger.error("Pipeline failed at preprocessing stage")
            return False
        
        # Step 2: Training
        if not self.run_training(models, config_overrides):
            logger.error("Pipeline failed at training stage")
            return False
        
        # Step 3: Evaluation
        if not self.run_evaluation(models):
            logger.error("Pipeline failed at evaluation stage")
            return False
        
        total_time = time.time() - start_time
        logger.info(f"Full pipeline completed successfully in {total_time:.2f} seconds!")
        
        return True
    
    def list_models(self):
        """List available models with descriptions."""
        print("Available Models:")
        print("=" * 50)
        for name, config in self.models_config.items():
            print(f"• {name}: {config['description']}")
        print()
    
    def get_status(self):
        """Get current status of the application."""
        print("LLM Evaluation Application Status:")
        print("=" * 40)
        
        # Check preprocessing
        train_file = os.path.join(self.data_dir, "processed", "train.jsonl")
        eval_file = os.path.join(self.data_dir, "processed", "eval.jsonl")
        preprocessing_done = os.path.exists(train_file) and os.path.exists(eval_file)
        print(f"Preprocessing: {'✓ Complete' if preprocessing_done else '✗ Not done'}")
        
        # Check training
        trained_models = []
        for model_name in self.models_config.keys():
            model_dir = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_dir) and os.listdir(model_dir):
                trained_models.append(model_name)
        
        print(f"Training: {len(trained_models)}/{len(self.models_config)} models trained")
        if trained_models:
            print(f"  Trained models: {', '.join(trained_models)}")
        
        # Check evaluation
        results_file = os.path.join(self.results_dir, "evaluation_results.json")
        evaluation_done = os.path.exists(results_file)
        print(f"Evaluation: {'✓ Complete' if evaluation_done else '✗ Not done'}")
        
        # Check visualizations
        viz_dir = os.path.join(self.results_dir, "visualizations")
        viz_done = os.path.exists(viz_dir) and os.listdir(viz_dir)
        print(f"Visualizations: {'✓ Complete' if viz_done else '✗ Not done'}")
        
        print()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="LLM Training and Evaluation Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (uses data/raw/source.pdf by default)
  python main.py full
  
  # Run with custom document
  python main.py full --input /path/to/document.pdf
  
  # Run only preprocessing
  python main.py preprocess
  
  # Train specific models
  python main.py train --models llama3 gpt_oss
  
  # Evaluate all models
  python main.py evaluate
  
  # Create visualizations
  python main.py visualize
  
  # Check status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--input', default='data/raw/source.pdf', help='Input document path (default: data/raw/source.pdf)')
    full_parser.add_argument('--models', nargs='+', help='Models to train/evaluate')
    full_parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    full_parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing only')
    preprocess_parser.add_argument('--input', default='data/raw/source.pdf', help='Input document path (default: data/raw/source.pdf)')
    preprocess_parser.add_argument('--force', action='store_true', help='Force reprocessing')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training only')
    train_parser.add_argument('--models', nargs='+', help='Models to train')
    train_parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation only')
    eval_parser.add_argument('--models', nargs='+', help='Models to evaluate')
    eval_parser.add_argument('--stages', nargs='+', choices=['before_training', 'after_training'],
                           help='Evaluation stages')
    eval_parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations only')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show application status')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create application instance
    app = LLMEvaluationApp()
    
    try:
        if args.command == 'full':
            config_overrides = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'max_length': 256  # Shorter for testing
            }
            success = app.run_full_pipeline(args.input, args.models, config_overrides)
            
        elif args.command == 'preprocess':
            success = app.run_preprocessing(args.input, args.force)
            
        elif args.command == 'train':
            config_overrides = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'max_length': 256
            }
            success = app.run_training(args.models, config_overrides)
            
        elif args.command == 'evaluate':
            success = app.run_evaluation(args.models, args.stages, not args.no_viz)
            
        elif args.command == 'visualize':
            success = app.create_visualizations()
            
        elif args.command == 'status':
            app.get_status()
            success = True
            
        elif args.command == 'list':
            app.list_models()
            success = True
            
        else:
            parser.print_help()
            success = False
        
        if success:
            logger.info("Command completed successfully!")
        else:
            logger.error("Command failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

