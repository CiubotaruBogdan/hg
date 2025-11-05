"""
LLM Evaluator
Main evaluation class for assessing LLM performance before and after training.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import logging

from .metrics import EvaluationMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Main evaluator for LLM performance assessment."""
    
    def __init__(self, output_dir: str = "/home/ubuntu/llm-evaluation/results"):
        self.output_dir = output_dir
        self.metrics_calculator = EvaluationMetrics()
        self.evaluation_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, 
                      model_name: str,
                      model_path: str,
                      eval_data_path: str,
                      stage: str = "before_training") -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_name (str): Name of the model
            model_path (str): Path to the model
            eval_data_path (str): Path to evaluation data
            stage (str): Evaluation stage ("before_training" or "after_training")
            
        Returns:
            Dict containing evaluation results
        """
        logger.info(f"Evaluating {model_name} ({stage})")
        
        start_time = time.time()
        
        try:
            # Load evaluation data
            eval_data = self._load_evaluation_data(eval_data_path)
            
            # Generate predictions
            predictions = self._generate_predictions(model_name, model_path, eval_data)
            
            # Extract references and questions
            references = [item['answer'] for item in eval_data]
            questions = [item['question'] for item in eval_data]
            
            # Compute metrics
            metrics = self.metrics_calculator.compute_all_metrics(
                predictions, references, questions
            )
            
            # Add timing information
            evaluation_time = time.time() - start_time
            metrics['evaluation_time'] = evaluation_time
            metrics['num_examples'] = len(eval_data)
            
            # Store results
            result_key = f"{model_name}_{stage}"
            self.evaluation_results[result_key] = {
                'model_name': model_name,
                'model_path': model_path,
                'stage': stage,
                'metrics': metrics,
                'evaluation_time': evaluation_time,
                'num_examples': len(eval_data),
                'timestamp': time.time()
            }
            
            logger.info(f"Evaluation completed for {model_name} ({stage})")
            logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
            
            return self.evaluation_results[result_key]
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name} ({stage}): {str(e)}")
            raise
    
    def evaluate_models(self, 
                       model_names: List[str],
                       eval_data_path: str,
                       stages: List[str] = ["before_training"],
                       output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate specified models at given stages.
        
        Args:
            model_names (List[str]): Names of models to evaluate
            eval_data_path (str): Path to evaluation data
            stages (List[str]): Evaluation stages
            output_dir (str): Output directory for results
            
        Returns:
            Dict containing evaluation results
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting evaluation of {len(model_names)} models at {len(stages)} stages")
        
        # Load evaluation data
        eval_data = self._load_evaluation_data(eval_data_path)
        logger.info(f"Loaded {len(eval_data)} evaluation examples")
        
        all_results = {}
        
        for model_name in model_names:
            for stage in stages:
                try:
                    logger.info(f"Evaluating {model_name} ({stage})")
                    
                    start_time = time.time()
                    
                    # Extract questions for prediction
                    questions = [item['question'] for item in eval_data]
                    
                    # Generate predictions
                    predictions = self._generate_predictions(model_name, questions)
                    
                    # Extract references
                    references = [item['answer'] for item in eval_data]
                    
                    # Compute metrics
                    metrics = self.metrics_calculator.compute_all_metrics(
                        predictions, references, questions
                    )
                    
                    # Add timing and metadata
                    evaluation_time = time.time() - start_time
                    
                    result = {
                        'model_name': model_name,
                        'stage': stage,
                        'metrics': metrics,
                        'evaluation_time': evaluation_time,
                        'num_examples': len(eval_data),
                        'timestamp': time.time()
                    }
                    
                    result_key = f"{model_name}_{stage}"
                    all_results[result_key] = result
                    
                    logger.info(f"Evaluation completed for {model_name} ({stage})")
                    logger.info(f"Evaluation time: {evaluation_time:.2f} seconds")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} at {stage}: {str(e)}")
                    continue
        
        # Save results
        self._save_results(all_results)
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        logger.info("Completed evaluation of all models")
        return all_results
    
    def _load_evaluation_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load evaluation data from JSONL file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Evaluation data not found: {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} evaluation examples")
        return data
    
    def _generate_predictions(self, model_name: str, questions: List[str]) -> List[str]:
        """
        Generate predictions using the actual model.
        
        Args:
            model_name (str): Name of the model
            questions (List[str]): List of questions
            
        Returns:
            List[str]: Generated predictions
        """
        try:
            # Model configurations
            model_configs = {
                "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "qwen3": "Qwen/Qwen2.5-7B-Instruct",
                "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
                "gemma3": "google/gemma-2-9b-it",
                "gpt_oss": "ollama:gpt-oss:20b"  # Special handling for Ollama
            }
            
            if model_name not in model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            predictions = []
            
            # Handle Ollama models separately
            if model_name == "gpt_oss":
                predictions = self._generate_ollama_predictions(questions)
            else:
                # Handle HuggingFace models
                predictions = self._generate_hf_predictions(model_name, model_configs[model_name], questions)
            
            logger.info(f"Generated {len(predictions)} predictions for {model_name}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for {model_name}: {e}")
            # Return placeholder predictions to avoid complete failure
            return [f"Unable to generate prediction for question {i+1}" for i in range(len(questions))]
    
    def _generate_hf_predictions(self, model_name: str, hf_model: str, questions: List[str]) -> List[str]:
        """Generate predictions using HuggingFace models."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading {model_name} for inference...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            predictions = []
            
            for question in questions:
                try:
                    # Format prompt based on model
                    if model_name == "llama3":
                        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    elif model_name == "qwen3":
                        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                    elif model_name == "gemma3":
                        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
                    else:  # deepseek
                        prompt = f"User: {question}\n\nAssistant: "
                    
                    # Tokenize
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    # Move to device
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract answer (remove the prompt part)
                    if prompt in response:
                        answer = response.replace(prompt, "").strip()
                    else:
                        answer = response.strip()
                    
                    # Clean up answer
                    if answer.startswith("Assistant:"):
                        answer = answer[10:].strip()
                    
                    predictions.append(answer)
                    
                except Exception as e:
                    logger.warning(f"Error generating prediction for question: {e}")
                    predictions.append("Unable to generate answer")
            
            return predictions
            
        except ImportError:
            logger.error("Required libraries not available. Please install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"HuggingFace prediction error: {e}")
            raise
    
    def _generate_ollama_predictions(self, questions: List[str]) -> List[str]:
        """Generate predictions using Ollama."""
        try:
            import subprocess
            
            predictions = []
            
            for question in questions:
                try:
                    # Create prompt
                    prompt = f"Question: {question}\nAnswer:"
                    
                    # Call Ollama
                    result = subprocess.run([
                        'ollama', 'run', 'gpt-oss:20b', prompt
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        answer = result.stdout.strip()
                        predictions.append(answer)
                    else:
                        logger.warning(f"Ollama error: {result.stderr}")
                        predictions.append("Unable to generate answer")
                        
                except subprocess.TimeoutExpired:
                    logger.warning("Ollama timeout for question")
                    predictions.append("Timeout generating answer")
                except Exception as e:
                    logger.warning(f"Error with Ollama: {e}")
                    predictions.append("Error generating answer")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ollama prediction error: {e}")
            raise
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        results_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = self._make_serializable(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {results_file}")
    
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
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generate a comparison report of all models."""
        report_file = os.path.join(self.output_dir, "comparison_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# LLM Evaluation Comparison Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Extract model names
            models = set()
            stages = set()
            for key in results.keys():
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    models.add(parts[0])
                    stages.add(parts[1])
            
            models = sorted(models)
            stages = sorted(stages)
            
            f.write("## Summary\n\n")
            f.write(f"- **Models Evaluated**: {len(models)}\n")
            f.write(f"- **Evaluation Stages**: {', '.join(stages)}\n")
            f.write(f"- **Total Evaluations**: {len(results)}\n\n")
            
            # Metrics comparison table
            f.write("## Metrics Comparison\n\n")
            
            # Get all metric names
            all_metrics = set()
            for result in results.values():
                if 'metrics' in result:
                    all_metrics.update(result['metrics'].keys())
            
            main_metrics = ['exact_match', 'f1_score', 'bleu_score', 'rouge_l', 'semantic_similarity']
            main_metrics = [m for m in main_metrics if m in all_metrics]
            
            # Create comparison table
            f.write("| Model | Stage | " + " | ".join(m.replace('_', ' ').title() for m in main_metrics) + " |\n")
            f.write("|-------|-------|" + "|".join(["-------"] * len(main_metrics)) + "|\n")
            
            for model in models:
                for stage in stages:
                    key = f"{model}_{stage}"
                    if key in results:
                        metrics = results[key].get('metrics', {})
                        f.write(f"| {model} | {stage} |")
                        for metric in main_metrics:
                            value = metrics.get(metric, 0.0)
                            f.write(f" {value:.4f} |")
                        f.write("\n")
            
            f.write("\n")
            
            # Improvement analysis
            f.write("## Training Improvement Analysis\n\n")
            
            for model in models:
                before_key = f"{model}_before_training"
                after_key = f"{model}_after_training"
                
                if before_key in results and after_key in results:
                    before_metrics = results[before_key].get('metrics', {})
                    after_metrics = results[after_key].get('metrics', {})
                    
                    f.write(f"### {model.title()}\n\n")
                    
                    for metric in main_metrics:
                        before_val = before_metrics.get(metric, 0.0)
                        after_val = after_metrics.get(metric, 0.0)
                        improvement = after_val - before_val
                        improvement_pct = (improvement / max(before_val, 0.001)) * 100
                        
                        f.write(f"- **{metric.replace('_', ' ').title()}**: ")
                        f.write(f"{before_val:.4f} → {after_val:.4f} ")
                        f.write(f"({improvement:+.4f}, {improvement_pct:+.1f}%)\n")
                    
                    f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for key, result in results.items():
                f.write(f"### {key.replace('_', ' ').title()}\n\n")
                f.write(f"- **Model Path**: {result.get('model_path', 'N/A')}\n")
                f.write(f"- **Evaluation Time**: {result.get('evaluation_time', 0):.2f} seconds\n")
                f.write(f"- **Number of Examples**: {result.get('num_examples', 0)}\n\n")
                
                metrics = result.get('metrics', {})
                f.write("**Metrics**:\n\n")
                for metric, value in sorted(metrics.items()):
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}\n")
                    else:
                        f.write(f"- {metric.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n")
        
        logger.info(f"Comparison report saved to: {report_file}")
    
    def get_best_model(self, metric: str = "f1_score", stage: str = "after_training") -> Tuple[str, float]:
        """Get the best performing model for a specific metric."""
        best_model = None
        best_score = -1
        
        for key, result in self.evaluation_results.items():
            if stage in key:
                metrics = result.get('metrics', {})
                score = metrics.get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_model = key.replace(f"_{stage}", "")
        
        return best_model, best_score


def main():
    """Test the evaluator."""
    evaluator = LLMEvaluator()
    
    # Test configuration
    models_config = {
        "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen3": "Qwen/Qwen2.5-7B-Instruct",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
        "gemma3": "google/gemma-2-9b-it",
        "pansophic": "newport-ai/pansophic-1-preview"
    }
    
    eval_data_path = "/home/ubuntu/llm-evaluation/data/processed/eval.jsonl"
    
    if not os.path.exists(eval_data_path):
        logger.error("Evaluation data not found. Please run preprocessing first.")
        return
    
    try:
        # Run evaluation
        results = evaluator.evaluate_all_models(
            models_config, 
            eval_data_path, 
            stages=["before_training"]  # Test with one stage first
        )
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED")
        print("="*50)
        print(f"Evaluated {len(results)} model configurations")
        
        # Show best model
        best_model, best_score = evaluator.get_best_model()
        if best_model:
            print(f"Best performing model: {best_model} (F1: {best_score:.4f})")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

