#!/usr/bin/env python3
"""
HG 585 LLM Evaluation System
Interactive Menu Interface

Double-click this file to run the interactive menu system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import PreprocessingPipeline
from training import (
    Llama3Trainer, Qwen3Trainer, DeepSeekTrainer, 
    Gemma3Trainer, GPTOSSTrainer
)
from evaluation import LLMEvaluator
from evaluation.visualizer import ResultsVisualizer
from model_manager import ModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InteractiveLLMEvaluationApp:
    """Interactive menu-driven LLM evaluation application."""
    
    def __init__(self):
        # Set up directories
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.preprocessor = PreprocessingPipeline()
        self.evaluator = LLMEvaluator()
        self.visualizer = ResultsVisualizer(
            results_file=str(self.results_dir / "evaluation_results.json"),
            output_dir=str(self.results_dir / "visualizations")
        )
        self.model_manager = ModelManager(str(self.models_dir))
        
        # Model configurations
        self.models_config = {
            "llama3": {
                "trainer": Llama3Trainer,
                "description": "Meta's Llama 3.1 - General purpose flagship model"
            },
            "qwen3": {
                "trainer": Qwen3Trainer,
                "description": "Alibaba's Qwen 3 - Multilingual powerhouse"
            },
            "deepseek": {
                "trainer": DeepSeekTrainer,
                "description": "DeepSeek-V2 - Specialized reasoning model"
            },
            "gemma3": {
                "trainer": Gemma3Trainer,
                "description": "Google's Gemma 3 - Latest open-source model"
            },
            "gpt_oss": {
                "trainer": GPTOSSTrainer,
                "description": "GPT-OSS 20B - Open-source GPT model via Ollama"
            }
        }
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the application header."""
        print("=" * 70)
        print("           HG 585 LLM EVALUATION SYSTEM")
        print("         Interactive Menu Interface")
        print("=" * 70)
        print()
    
    def print_main_menu(self):
        """Print the main menu options."""
        print("Please select an option:")
        print()
        print("  1. Preprocess Data Source (HG585.pdf)")
        print("  2. Download Models")
        print("  3. Check Models Status")
        print("  4. Evaluate Initial Models (Before Training)")
        print("  5. Train Models")
        print("  6. Evaluate Models Post Training")
        print()
        print("  7. Run Complete Workflow (Steps 1-6)")
        print("  8. Create Visualizations")
        print("  9. View System Status")
        print("  0. Exit")
        print()
    
    def get_user_choice(self) -> str:
        """Get user menu choice."""
        while True:
            try:
                choice = input("Enter your choice (0-9): ").strip()
                if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return choice
                else:
                    print("Invalid choice. Please enter a number between 0-9.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            except Exception:
                print("Invalid input. Please try again.")
    
    def pause(self):
        """Pause and wait for user input."""
        print()
        input("Press Enter to continue...")
    
    def step1_preprocess_data(self):
        """Step 1: Preprocess data source."""
        self.clear_screen()
        self.print_header()
        print("STEP 1: PREPROCESS DATA SOURCE")
        print("-" * 40)
        
        input_file = self.data_dir / "raw" / "source.pdf"
        
        if not input_file.exists():
            print(f"❌ Source document not found: {input_file}")
            print("Please place your HG585.pdf file as data/raw/source.pdf")
            self.pause()
            return False
        
        print(f"📄 Processing document: {input_file}")
        print("This will extract text and create training datasets...")
        print()
        
        try:
            # Run preprocessing
            processed_dir = self.data_dir / "processed"
            results = self.preprocessor.process_document(str(input_file), str(processed_dir))
            
            print("✅ Preprocessing completed successfully!")
            print()
            print("Results:")
            print(f"  • Training examples: {results['statistics']['train_examples']}")
            print(f"  • Evaluation examples: {results['statistics']['eval_examples']}")
            print(f"  • Original words: {results['statistics']['original_word_count']:,}")
            print(f"  • Text chunks: {results['statistics']['text_chunks']}")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            self.pause()
            return False
    
    def step2_download_models(self):
        """Step 2: Download models."""
        self.clear_screen()
        self.print_header()
        print("STEP 2: DOWNLOAD MODELS")
        print("-" * 40)
        
        print("Available models to download:")
        print()
        for i, (model_name, config) in enumerate(self.models_config.items(), 1):
            auth_required = "🔒" if model_name in ["llama3", "gemma3"] else ""
            print(f"  {i}. {model_name.upper()}: {config['description']} {auth_required}")
        
        print()
        print("Options:")
        print("  0. HuggingFace Authentication (Login with token)")
        print("  A. Download all models")
        print("  S. Select specific models")
        print("  C. Cancel")
        print()
        
        choice = input("Enter your choice (0/A/S/C): ").strip().upper()
        
        if choice == 'C':
            return False
        elif choice == '0':
            # HuggingFace authentication
            print()
            success = self.model_manager.authenticate_huggingface()
            if success:
                print()
                print("Authentication completed! You can now download protected models.")
            else:
                print()
                print("Authentication failed. You can still download public models.")
            self.pause()
            return self.step2_download_models()  # Return to menu
        elif choice == 'A':
            models_to_download = list(self.models_config.keys())
        elif choice == 'S':
            models_to_download = self.select_models("download")
            if not models_to_download:
                return False
        else:
            print("Invalid choice.")
            self.pause()
            return False
        
        print()
        print(f"📥 Downloading {len(models_to_download)} models...")
        print("This may take several minutes depending on your internet connection.")
        print()
        
        try:
            results = {}
            for model_name in models_to_download:
                print(f"Downloading {model_name}...")
                results[model_name] = self.model_manager.download_model(model_name)
            
            # Show results
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            print()
            print(f"Download completed: {successful}/{total} models successful")
            print()
            
            for model_name, success in results.items():
                status = "✅" if success else "❌"
                print(f"{status} {model_name.upper()}")
            
            self.pause()
            return successful > 0
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            self.pause()
            return False
    
    def step3_check_status(self):
        """Step 3: Check models status."""
        self.clear_screen()
        self.print_header()
        print("STEP 3: CHECK MODELS STATUS")
        print("-" * 40)
        
        try:
            # Get system info
            system_info = self.model_manager.get_system_info()
            
            print("SYSTEM INFORMATION:")
            print(f"  GPU Available: {'✅' if system_info['gpu_available'] else '❌'}")
            print(f"  Ollama Available: {'✅' if system_info['ollama_available'] else '❌'}")
            print(f"  HuggingFace Auth: {'✅' if system_info['hf_auth'] else '❌'}")
            print(f"  Downloaded Models: {system_info['downloaded_models']}/{system_info['total_models']}")
            
            if system_info['gpu_available'] and 'gpu_memory' in system_info:
                print(f"  GPU Memory: {system_info['gpu_memory']} GB")
            
            # Check HuggingFace authentication status
            try:
                import subprocess
                result = subprocess.run(['huggingface-cli', 'whoami'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    username = result.stdout.strip()
                    print(f"  HuggingFace User: {username}")
                else:
                    print(f"  HuggingFace User: Not logged in")
            except:
                print(f"  HuggingFace User: CLI not available")
            
            print()
            print("MODELS STATUS:")
            print()
            
            # Get model status
            status_report = self.model_manager.check_all_models_status()
            
            for model_name, status in status_report.items():
                downloaded = "✅" if status["downloaded"] else "❌"
                auth_required = "🔒" if model_name in ["llama3", "gemma3"] else ""
                print(f"{downloaded} {model_name.upper()} {auth_required}")
                print(f"     Description: {status['description']}")
                print(f"     Size: {status['size_gb']}GB")
                print(f"     Type: {status['type']}")
                
                if status["downloaded"]:
                    print(f"     Status: Ready for training")
                else:
                    if model_name in ["llama3", "gemma3"]:
                        print(f"     Status: Not downloaded (requires HuggingFace auth)")
                    else:
                        print(f"     Status: Not downloaded")
                print()
            
            print("💡 Tip: Use option 0 in Download Models to authenticate with HuggingFace")
            print("    for accessing protected models (Llama 3.1, Gemma 3)")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Status check failed: {str(e)}")
            self.pause()
            return False
    
    def step4_evaluate_initial(self):
        """Step 4: Evaluate initial models."""
        self.clear_screen()
        self.print_header()
        print("STEP 4: EVALUATE INITIAL MODELS")
        print("-" * 40)
        
        # Check if evaluation data exists
        eval_file = self.data_dir / "processed" / "eval.jsonl"
        if not eval_file.exists():
            print("❌ Evaluation data not found.")
            print("Please run Step 1 (Preprocess Data) first.")
            self.pause()
            return False
        
        # Get available models
        available_models = self.model_manager.get_available_models()
        if not available_models:
            print("❌ No models are downloaded.")
            print("Please run Step 2 (Download Models) first.")
            self.pause()
            return False
        
        print("Available models for evaluation:")
        for model in available_models:
            print(f"  • {model.upper()}")
        
        print()
        print("Options:")
        print("  A. Evaluate all available models")
        print("  S. Select specific models")
        print("  C. Cancel")
        print()
        
        choice = input("Enter your choice (A/S/C): ").strip().upper()
        
        if choice == 'C':
            return False
        elif choice == 'A':
            models_to_evaluate = available_models
        elif choice == 'S':
            models_to_evaluate = self.select_models("evaluate", available_models)
            if not models_to_evaluate:
                return False
        else:
            print("Invalid choice.")
            self.pause()
            return False
        
        print()
        print(f"🔍 Evaluating {len(models_to_evaluate)} models (before training)...")
        print("This will test the models' baseline performance on HG585 questions.")
        print()
        
        try:
            # Run evaluation
            results = self.evaluator.evaluate_models(
                models_to_evaluate,
                str(eval_file),
                stages=["before_training"],
                output_dir=str(self.results_dir)
            )
            
            # Create visualizations
            print("Creating visualizations...")
            viz_dir = self.results_dir / "visualizations"
            self.visualizer.create_all_visualizations(results, str(viz_dir))
            
            print("✅ Initial evaluation completed!")
            print(f"Results saved to: {self.results_dir}")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            self.pause()
            return False
    
    def step5_train_models(self):
        """Step 5: Train models."""
        self.clear_screen()
        self.print_header()
        print("STEP 5: TRAIN MODELS")
        print("-" * 40)
        
        # Check prerequisites
        train_file = self.data_dir / "processed" / "train.jsonl"
        if not train_file.exists():
            print("❌ Training data not found.")
            print("Please run Step 1 (Preprocess Data) first.")
            self.pause()
            return False
        
        available_models = self.model_manager.get_available_models()
        if not available_models:
            print("❌ No models are downloaded.")
            print("Please run Step 2 (Download Models) first.")
            self.pause()
            return False
        
        print("Available models for training:")
        for model in available_models:
            print(f"  • {model.upper()}")
        
        print()
        print("Options:")
        print("  A. Train all available models")
        print("  S. Select specific models")
        print("  C. Cancel")
        print()
        
        choice = input("Enter your choice (A/S/C): ").strip().upper()
        
        if choice == 'C':
            return False
        elif choice == 'A':
            models_to_train = available_models
        elif choice == 'S':
            models_to_train = self.select_models("train", available_models)
            if not models_to_train:
                return False
        else:
            print("Invalid choice.")
            self.pause()
            return False
        
        # Get training parameters
        print()
        print("Training Configuration:")
        try:
            epochs = int(input("Number of epochs (default 3): ") or "3")
            batch_size = int(input("Batch size (default 2): ") or "2")
        except ValueError:
            epochs, batch_size = 3, 2
        
        print()
        print(f"🏋️ Training {len(models_to_train)} models...")
        print(f"Configuration: {epochs} epochs, batch size {batch_size}")
        print("This may take several hours depending on your hardware.")
        print()
        
        try:
            results = {}
            for model_name in models_to_train:
                print(f"Training {model_name.upper()}...")
                
                # Get trainer class
                trainer_class = self.models_config[model_name]["trainer"]
                trainer = trainer_class()
                
                # Train model
                config = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': 1e-5,
                    'max_length': 512
                }
                
                success = trainer.train(
                    str(train_file),
                    str(self.models_dir / model_name),
                    config
                )
                
                results[model_name] = success
                status = "✅" if success else "❌"
                print(f"{status} {model_name.upper()} training completed")
                print()
            
            # Show results
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            print(f"Training completed: {successful}/{total} models successful")
            
            self.pause()
            return successful > 0
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            self.pause()
            return False
    
    def step6_evaluate_final(self):
        """Step 6: Evaluate models post training."""
        self.clear_screen()
        self.print_header()
        print("STEP 6: EVALUATE MODELS POST TRAINING")
        print("-" * 40)
        
        # Check prerequisites
        eval_file = self.data_dir / "processed" / "eval.jsonl"
        if not eval_file.exists():
            print("❌ Evaluation data not found.")
            print("Please run Step 1 (Preprocess Data) first.")
            self.pause()
            return False
        
        # Check for trained models
        trained_models = []
        for model_name in self.models_config.keys():
            model_dir = self.models_dir / model_name
            if model_dir.exists() and any(model_dir.iterdir()):
                trained_models.append(model_name)
        
        if not trained_models:
            print("❌ No trained models found.")
            print("Please run Step 5 (Train Models) first.")
            self.pause()
            return False
        
        print("Available trained models:")
        for model in trained_models:
            print(f"  • {model.upper()}")
        
        print()
        print("Options:")
        print("  A. Evaluate all trained models")
        print("  S. Select specific models")
        print("  C. Cancel")
        print()
        
        choice = input("Enter your choice (A/S/C): ").strip().upper()
        
        if choice == 'C':
            return False
        elif choice == 'A':
            models_to_evaluate = trained_models
        elif choice == 'S':
            models_to_evaluate = self.select_models("evaluate", trained_models)
            if not models_to_evaluate:
                return False
        else:
            print("Invalid choice.")
            self.pause()
            return False
        
        print()
        print(f"🔍 Evaluating {len(models_to_evaluate)} models (after training)...")
        print("This will compare before/after training performance.")
        print()
        
        try:
            # Run evaluation
            results = self.evaluator.evaluate_models(
                models_to_evaluate,
                str(eval_file),
                stages=["before_training", "after_training"],
                output_dir=str(self.results_dir)
            )
            
            # Create visualizations
            print("Creating comparison visualizations...")
            viz_dir = self.results_dir / "visualizations"
            self.visualizer.create_all_visualizations(results, str(viz_dir))
            
            print("✅ Final evaluation completed!")
            print("📊 Comparison charts and reports generated!")
            print(f"Results saved to: {self.results_dir}")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Final evaluation failed: {str(e)}")
            self.pause()
            return False
    
    def step7_complete_workflow(self):
        """Step 7: Run complete workflow."""
        self.clear_screen()
        self.print_header()
        print("STEP 7: COMPLETE WORKFLOW")
        print("-" * 40)
        
        print("This will run all 6 steps automatically:")
        print("  1. Preprocess Data")
        print("  2. Download Models")
        print("  3. Check Status")
        print("  4. Evaluate Initial Models")
        print("  5. Train Models")
        print("  6. Evaluate Final Models")
        print()
        
        confirm = input("Continue with complete workflow? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
        
        print()
        print("🚀 Starting complete workflow...")
        print()
        
        # Step 1: Preprocess
        print("Step 1/6: Preprocessing data...")
        if not self.step1_preprocess_data():
            print("❌ Workflow failed at preprocessing step")
            return False
        
        # Step 2: Download
        print("Step 2/6: Downloading models...")
        if not self.step2_download_models():
            print("❌ Workflow failed at download step")
            return False
        
        # Step 3: Status (always succeeds)
        print("Step 3/6: Checking status...")
        self.step3_check_status()
        
        # Step 4: Initial evaluation
        print("Step 4/6: Initial evaluation...")
        if not self.step4_evaluate_initial():
            print("❌ Workflow failed at initial evaluation step")
            return False
        
        # Step 5: Training
        print("Step 5/6: Training models...")
        if not self.step5_train_models():
            print("❌ Workflow failed at training step")
            return False
        
        # Step 6: Final evaluation
        print("Step 6/6: Final evaluation...")
        if not self.step6_evaluate_final():
            print("❌ Workflow failed at final evaluation step")
            return False
        
        print()
        print("🎉 Complete workflow finished successfully!")
        print("All models have been trained and evaluated on HG585 data.")
        
        self.pause()
        return True
    
    def step8_create_visualizations(self):
        """Step 8: Create visualizations."""
        self.clear_screen()
        self.print_header()
        print("STEP 8: CREATE VISUALIZATIONS")
        print("-" * 40)
        
        results_file = self.results_dir / "evaluation_results.json"
        if not results_file.exists():
            print("❌ No evaluation results found.")
            print("Please run evaluation steps first.")
            self.pause()
            return False
        
        try:
            print("📊 Creating visualizations...")
            
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Create visualizations
            viz_dir = self.results_dir / "visualizations"
            self.visualizer.create_all_visualizations(results, str(viz_dir))
            
            print("✅ Visualizations created successfully!")
            print(f"Charts saved to: {viz_dir}")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {str(e)}")
            self.pause()
            return False
    
    def step9_system_status(self):
        """Step 9: View system status."""
        self.clear_screen()
        self.print_header()
        print("STEP 9: SYSTEM STATUS")
        print("-" * 40)
        
        try:
            # Check data files
            print("DATA FILES:")
            source_file = self.data_dir / "raw" / "source.pdf"
            train_file = self.data_dir / "processed" / "train.jsonl"
            eval_file = self.data_dir / "processed" / "eval.jsonl"
            
            print(f"  Source document: {'✅' if source_file.exists() else '❌'}")
            print(f"  Training data: {'✅' if train_file.exists() else '❌'}")
            print(f"  Evaluation data: {'✅' if eval_file.exists() else '❌'}")
            
            # Check models
            print()
            print("MODELS:")
            available_models = self.model_manager.get_available_models()
            for model_name in self.models_config.keys():
                downloaded = "✅" if model_name in available_models else "❌"
                
                # Check if trained
                model_dir = self.models_dir / model_name
                trained = "✅" if model_dir.exists() and any(model_dir.iterdir()) else "❌"
                
                print(f"  {model_name.upper()}: Downloaded {downloaded}, Trained {trained}")
            
            # Check results
            print()
            print("RESULTS:")
            results_file = self.results_dir / "evaluation_results.json"
            viz_dir = self.results_dir / "visualizations"
            
            print(f"  Evaluation results: {'✅' if results_file.exists() else '❌'}")
            print(f"  Visualizations: {'✅' if viz_dir.exists() and any(viz_dir.iterdir()) else '❌'}")
            
            self.pause()
            return True
            
        except Exception as e:
            print(f"❌ Status check failed: {str(e)}")
            self.pause()
            return False
    
    def select_models(self, action: str, available_models: List[str] = None) -> List[str]:
        """Helper method to select specific models."""
        if available_models is None:
            available_models = list(self.models_config.keys())
        
        print()
        print(f"Select models to {action}:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model.upper()}")
        
        print()
        print("Enter model numbers separated by spaces (e.g., 1 3 5):")
        
        try:
            choices = input("Your selection: ").strip().split()
            selected_models = []
            
            for choice in choices:
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    selected_models.append(available_models[idx])
                else:
                    print(f"Invalid choice: {choice}")
            
            return selected_models
            
        except (ValueError, KeyboardInterrupt):
            return []
    
    def run(self):
        """Run the interactive menu system."""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_main_menu()
            
            choice = self.get_user_choice()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                self.step1_preprocess_data()
            elif choice == '2':
                self.step2_download_models()
            elif choice == '3':
                self.step3_check_status()
            elif choice == '4':
                self.step4_evaluate_initial()
            elif choice == '5':
                self.step5_train_models()
            elif choice == '6':
                self.step6_evaluate_final()
            elif choice == '7':
                self.step7_complete_workflow()
            elif choice == '8':
                self.step8_create_visualizations()
            elif choice == '9':
                self.step9_system_status()


def main():
    """Main entry point."""
    try:
        app = InteractiveLLMEvaluationApp()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
