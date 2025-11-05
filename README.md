# LLM Training and Evaluation System

A comprehensive system for training and evaluating Large Language Models (LLMs) on Romanian government documents, specifically designed for the HG 585/2002 document analysis.

## Overview

This system trains and compares 5 popular LLMs on a Romanian legal document to evaluate their question-answering capabilities before and after fine-tuning. The goal is to determine which model performs best for Romanian legal text comprehension.

## Selected Models

The system evaluates and compares **5 state-of-the-art LLMs**:

1. **Llama 3.1** (Meta) - General-purpose flagship model
2. **Qwen 3** (Alibaba) - Multilingual powerhouse  
3. **DeepSeek-V2** (DeepSeek AI) - Specialized reasoning model
4. **Gemma 3** (Google) - Latest open-source model
5. **GPT-OSS 20B** (Ollama) - Open-source GPT model via Ollama

## Features

- **Document Processing**: Extracts and cleans text from Word and PDF documents
- **Data Preparation**: Creates question-answer pairs for training
- **Model Training**: Fine-tunes models using LoRA (Low-Rank Adaptation)
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, F1, exact match
- **Visualization**: Professional charts and reports
- **Romanian Language Support**: Specialized handling for Romanian text and diacritics

## Project Structure

```
llm-evaluation/
├── data/
│   ├── raw/                 # Original documents
│   └── processed/           # Processed training data
├── src/
│   ├── preprocessing/       # Data preprocessing modules
│   ├── training/           # Model training scripts
│   ├── evaluation/         # Evaluation and metrics
│   └── main.py            # Main application interface
├── models/                 # Trained model outputs
├── results/               # Evaluation results and visualizations
└── requirements.txt       # Python dependencies
```

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Place your HG 585 document** as `data/raw/source.pdf` (or `source.docx`)

## Usage

### Interactive Menu

Simply double-click `main.py` or run from your terminal:

```bash
python src/main.py
```

This will launch the interactive menu:

```
======================================================================
           HG 585 LLM EVALUATION SYSTEM
         Interactive Menu Interface
======================================================================

Please select an option:

  1. Preprocess Data Source (HG585.pdf)
  2. Download Models
  3. Check Models Status
  4. Evaluate Initial Models (Before Training)
  5. Train Models
  6. Evaluate Models Post Training

  7. Run Complete Workflow (Steps 1-6)
  8. Create Visualizations
  9. View System Status
  0. Exit

Enter your choice (0-9): 
```

### Step-by-Step Usage

#### 1. Preprocessing
```bash
python src/main.py preprocess
```
This will:
- Parse the document at data/raw/source.pdf (or .docx)
- Clean and normalize text
- Generate question-answer pairs
- Create train/eval datasets

#### 2. Training
```bash
python src/main.py train --models llama3 qwen3 --epochs 3 --batch-size 4
```
This will:
- Load pre-trained models
- Apply LoRA fine-tuning
- Train on the HG 585 dataset
- Save trained models

#### 3. Evaluation
```bash
python src/main.py evaluate --models llama3 qwen3
```
This will:
- Test models before and after training
- Compute comprehensive metrics
- Generate comparison reports
- Create visualizations

### Configuration Options

- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Training batch size (default: 2)
- `--models`: Specific models to process
- `--force`: Force reprocessing of existing data
- `--no-viz`: Skip visualization creation

## Evaluation Metrics

The system computes multiple metrics to assess model performance:

- **Exact Match**: Percentage of exactly matching answers
- **F1 Score**: Token-level overlap between prediction and reference
- **BLEU Score**: Machine translation quality metric
- **ROUGE-L**: Longest common subsequence-based metric
- **Semantic Similarity**: Content-based similarity measure
- **Romanian Accuracy**: Romanian-specific character and diacritic accuracy
- **Question-Answering Accuracy**: Relevance to original questions

## Output Files

### Results Directory
- `evaluation_results.json`: Complete evaluation data
- `comparison_report.md`: Detailed comparison analysis
- `visualizations/`: Charts and graphs
  - `metrics_comparison.png`: Bar charts of all metrics
  - `training_improvement.png`: Before/after training analysis
  - `model_ranking.png`: Performance ranking
  - `metrics_heatmap.png`: Heatmap visualization
  - `performance_radar.png`: Multi-dimensional comparison

### Model Directories
Each trained model saves:
- Model weights and configuration
- Training statistics and logs
- LoRA adapter weights

## Technical Details

### Training Configuration
- **LoRA Parameters**: r=16, alpha=32, dropout=0.1
- **Learning Rate**: 1e-5 to 2e-5 (model-dependent)
- **Sequence Length**: 512 tokens
- **Gradient Accumulation**: 4 steps
- **Evaluation Strategy**: Every 250 steps

### Data Processing
- **Chunk Size**: 512 tokens with 50-token overlap
- **Q&A Generation**: 2 pairs per chunk
- **Train/Eval Split**: 80/20
- **Text Cleaning**: Romanian diacritic normalization

### Hardware Requirements
- **GPU**: Recommended for training (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

## Mock Mode

When GPU or required libraries are unavailable, the system automatically falls back to mock mode, which:
- Simulates training and evaluation
- Generates realistic mock metrics
- Allows testing of the complete pipeline
- Useful for development and demonstration

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Use gradient accumulation
   - Enable LoRA (default)

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Document Not Found**
   - Ensure HG 585 document is in `data/raw/`
   - Check file permissions

4. **Model Download Issues**
   - Ensure internet connection
   - Check Hugging Face Hub access
   - Models will fall back to mock mode if unavailable

### Performance Tips

- Use GPU for training when available
- Start with smaller models for testing
- Reduce sequence length for faster processing
- Use mock mode for development

## Example Workflow

```bash
# 1. Check status
python src/main.py status

# 2. List available models
python src/main.py list

# 3. Run preprocessing (uses data/raw/source.pdf)
python src/main.py preprocess

# 4. Train specific models
python src/main.py train --models llama3 gpt_oss --epochs 2

# 5. Evaluate all models
python src/main.py evaluate

# 6. Create additional visualizations
python src/main.py visualize
```

## Results Interpretation

### Key Metrics to Watch
- **F1 Score**: Overall performance indicator
- **BLEU Score**: Translation/generation quality
- **Romanian Accuracy**: Language-specific performance
- **Training Improvement**: Before vs. after training gains

### Expected Outcomes
- Pansophic should excel at Romanian-specific tasks
- Llama 3.1 should show strong general performance
- All models should improve after training
- Visualizations will highlight strengths/weaknesses

## Contributing

To extend the system:
1. Add new models in `src/training/`
2. Implement new metrics in `src/evaluation/metrics.py`
3. Add visualizations in `src/evaluation/visualizer.py`
4. Update model configurations in `src/main.py`

## License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all dependencies are installed
4. Verify input document format and location

