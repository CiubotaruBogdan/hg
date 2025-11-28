"""
Results Visualizer
Creates charts and visualizations for LLM evaluation results.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    """Visualizer for LLM evaluation results."""
    
    def __init__(self, results_file: str, output_dir: str):
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = self._load_results()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Loaded results for {len(results)} evaluations")
        return results
    
    def create_all_visualizations(self):
        """Create all available visualizations."""
        logger.info("Creating all visualizations...")
        
        try:
            self.plot_metrics_comparison()
            self.plot_training_improvement()
            self.plot_model_ranking()
            self.plot_metrics_heatmap()
            self.plot_performance_radar()
            
            logger.info("All visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def plot_metrics_comparison(self):
        """Create bar chart comparing metrics across models."""
        # Prepare data
        data = []
        for key, result in self.results.items():
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                model_name, stage = parts
                metrics = result.get('metrics', {})
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and metric in ['exact_match', 'f1_score', 'bleu_score', 'rouge_l']:
                        data.append({
                            'Model': model_name,
                            'Stage': stage,
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': value
                        })
        
        if not data:
            logger.warning("No data available for metrics comparison")
            return
        
        df = pd.DataFrame(data)
        
        # Create subplot for each metric
        metrics = df['Metric'].unique()
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            metric_data = df[df['Metric'] == metric]
            
            # Create grouped bar chart
            sns.barplot(data=metric_data, x='Model', y='Value', hue='Stage', ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(title='Stage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Metrics comparison chart saved")
    
    def plot_training_improvement(self):
        """Create chart showing training improvements."""
        # Calculate improvements
        improvements = []
        models = set()
        
        for key in self.results.keys():
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                models.add(parts[0])
        
        for model in models:
            before_key = f"{model}_before_training"
            after_key = f"{model}_after_training"
            
            if before_key in self.results and after_key in self.results:
                before_metrics = self.results[before_key].get('metrics', {})
                after_metrics = self.results[after_key].get('metrics', {})
                
                for metric in ['exact_match', 'f1_score', 'bleu_score', 'rouge_l']:
                    before_val = before_metrics.get(metric, 0.0)
                    after_val = after_metrics.get(metric, 0.0)
                    improvement = after_val - before_val
                    
                    improvements.append({
                        'Model': model,
                        'Metric': metric.replace('_', ' ').title(),
                        'Improvement': improvement,
                        'Improvement_Pct': (improvement / max(before_val, 0.001)) * 100
                    })
        
        if not improvements:
            logger.warning("No improvement data available")
            return
        
        df = pd.DataFrame(improvements)
        
        # Create improvement chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute improvement
        pivot_abs = df.pivot(index='Model', columns='Metric', values='Improvement')
        pivot_abs.plot(kind='bar', ax=ax1)
        ax1.set_title('Absolute Improvement After Training')
        ax1.set_ylabel('Score Improvement')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Percentage improvement
        pivot_pct = df.pivot(index='Model', columns='Metric', values='Improvement_Pct')
        pivot_pct.plot(kind='bar', ax=ax2)
        ax2.set_title('Percentage Improvement After Training')
        ax2.set_ylabel('Improvement (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_improvement.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training improvement chart saved")
    
    def plot_model_ranking(self):
        """Create model ranking visualization."""
        # Calculate average scores for each model
        model_scores = {}
        
        for key, result in self.results.items():
            if 'after_training' in key:  # Focus on post-training performance
                model_name = key.replace('_after_training', '')
                metrics = result.get('metrics', {})
                
                # Calculate average of main metrics
                main_metrics = ['exact_match', 'f1_score', 'bleu_score', 'rouge_l']
                scores = [metrics.get(metric, 0.0) for metric in main_metrics if metric in metrics]
                
                if scores:
                    model_scores[model_name] = {
                        'average_score': np.mean(scores),
                        'individual_scores': {metric: metrics.get(metric, 0.0) for metric in main_metrics}
                    }
        
        if not model_scores:
            logger.warning("No model ranking data available")
            return
        
        # Sort models by average score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['average_score'], reverse=True)
        
        # Create ranking chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average score ranking
        models = [item[0] for item in sorted_models]
        avg_scores = [item[1]['average_score'] for item in sorted_models]
        
        bars = ax1.bar(models, avg_scores, color=sns.color_palette("viridis", len(models)))
        ax1.set_title('Model Ranking by Average Score')
        ax1.set_ylabel('Average Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Individual metrics comparison for top models
        top_models = models[:3]  # Top 3 models
        metrics_data = []
        
        for model in top_models:
            for metric, score in model_scores[model]['individual_scores'].items():
                metrics_data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': score
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            sns.barplot(data=df_metrics, x='Metric', y='Score', hue='Model', ax=ax2)
            ax2.set_title('Top 3 Models - Detailed Metrics')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model ranking chart saved")
    
    def plot_metrics_heatmap(self):
        """Create heatmap of all metrics across models."""
        # Prepare data for heatmap
        heatmap_data = []
        
        for key, result in self.results.items():
            if 'after_training' in key:  # Focus on post-training
                model_name = key.replace('_after_training', '')
                metrics = result.get('metrics', {})
                
                # Select relevant metrics
                relevant_metrics = [
                    'exact_match', 'f1_score', 'bleu_score', 'rouge_l',
                    'semantic_similarity', 'romanian_accuracy'
                ]
                
                row_data = {'Model': model_name}
                for metric in relevant_metrics:
                    if metric in metrics:
                        row_data[metric.replace('_', ' ').title()] = metrics[metric]
                
                if len(row_data) > 1:  # Has at least one metric
                    heatmap_data.append(row_data)
        
        if not heatmap_data:
            logger.warning("No heatmap data available")
            return
        
        df = pd.DataFrame(heatmap_data)
        df = df.set_index('Model')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Model Performance Heatmap (After Training)')
        plt.ylabel('Models')
        plt.xlabel('Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Metrics heatmap saved")
    
    def plot_performance_radar(self):
        """Create radar chart for model performance."""
        try:
            # Prepare data for radar chart
            models_data = {}
            
            for key, result in self.results.items():
                if 'after_training' in key:
                    model_name = key.replace('_after_training', '')
                    metrics = result.get('metrics', {})
                    
                    # Select metrics for radar chart
                    radar_metrics = ['exact_match', 'f1_score', 'bleu_score', 'rouge_l', 'semantic_similarity']
                    scores = [metrics.get(metric, 0.0) for metric in radar_metrics]
                    
                    if any(score > 0 for score in scores):
                        models_data[model_name] = scores
            
            if not models_data:
                logger.warning("No radar chart data available")
                return
            
            # Create radar chart
            metrics_labels = ['Exact Match', 'F1 Score', 'BLEU', 'ROUGE-L', 'Semantic Sim.']
            
            # Number of variables
            N = len(metrics_labels)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Create subplot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Plot each model
            colors = sns.color_palette("husl", len(models_data))
            
            for i, (model, scores) in enumerate(models_data.items()):
                # Add first value at the end to close the circle
                scores += scores[:1]
                
                ax.plot(angles, scores, 'o-', linewidth=2, label=model, color=colors[i])
                ax.fill(angles, scores, alpha=0.25, color=colors[i])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart', size=16, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance radar chart saved")
            
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")
    
    def generate_summary_report(self):
        """Generate a summary report with key insights."""
        report_file = os.path.join(self.output_dir, 'visualization_summary.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# LLM Evaluation Visualization Summary\n\n")
            
            # Count evaluations
            total_evals = len(self.results)
            models = set()
            for key in self.results.keys():
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    models.add(parts[0])
            
            f.write(f"## Overview\n\n")
            f.write(f"- **Total Evaluations**: {total_evals}\n")
            f.write(f"- **Models Evaluated**: {len(models)}\n")
            f.write(f"- **Visualizations Created**: 5\n\n")
            
            f.write("## Generated Visualizations\n\n")
            f.write("1. **Metrics Comparison** (`metrics_comparison.png`)\n")
            f.write("   - Bar charts comparing key metrics across models and stages\n\n")
            
            f.write("2. **Training Improvement** (`training_improvement.png`)\n")
            f.write("   - Shows absolute and percentage improvements after training\n\n")
            
            f.write("3. **Model Ranking** (`model_ranking.png`)\n")
            f.write("   - Ranks models by average performance and shows detailed metrics for top performers\n\n")
            
            f.write("4. **Metrics Heatmap** (`metrics_heatmap.png`)\n")
            f.write("   - Heatmap visualization of all metrics across models\n\n")
            
            f.write("5. **Performance Radar** (`performance_radar.png`)\n")
            f.write("   - Radar chart showing multi-dimensional performance comparison\n\n")
            
            # Find best performing model
            best_model = None
            best_score = -1
            
            for key, result in self.results.items():
                if 'after_training' in key:
                    metrics = result.get('metrics', {})
                    avg_score = np.mean([
                        metrics.get('exact_match', 0),
                        metrics.get('f1_score', 0),
                        metrics.get('bleu_score', 0),
                        metrics.get('rouge_l', 0)
                    ])
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = key.replace('_after_training', '')
            
            if best_model:
                f.write(f"## Key Insights\n\n")
                f.write(f"- **Best Performing Model**: {best_model} (Average Score: {best_score:.4f})\n")
                f.write(f"- **Evaluation Data**: Based on HG 585 Romanian government document\n")
                f.write(f"- **Metrics Used**: Exact Match, F1 Score, BLEU, ROUGE-L, Semantic Similarity\n\n")
        
        logger.info("Visualization summary report saved")


def main():
    """Test the visualizer."""
    results_file = "/home/ubuntu/llm-evaluation/results/evaluation_results.json"
    output_dir = "/home/ubuntu/llm-evaluation/results/visualizations"
    
    if not os.path.exists(results_file):
        logger.error("Results file not found. Please run evaluation first.")
        return
    
    try:
        visualizer = ResultsVisualizer(results_file, output_dir)
        visualizer.create_all_visualizations()
        visualizer.generate_summary_report()
        
        print("\n" + "="*50)
        print("VISUALIZATIONS CREATED")
        print("="*50)
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

