# HG 585 LLM Evaluation System

A comprehensive system for training and evaluating Large Language Models (LLMs) on Romanian government documents, specifically designed for the HG 585/2002 document analysis.

## Overview

This system trains and compares 5 popular LLMs on the Romanian legal document HG 585/2002 to evaluate their question-answering capabilities before and after fine-tuning. The goal is to determine which model performs best for Romanian legal text comprehension.

## Selected Models

The system evaluates and compares **5 state-of-the-art LLMs**:

1. **Llama 3.1** (Meta) - General-purpose flagship model
2. **Qwen 3** (Alibaba) - Multilingual powerhouse  
3. **DeepSeek-V2** (DeepSeek AI) - Specialized reasoning model
4. **Gemma 3** (Google) - Latest open-source model
5. **GPT-OSS 20B** (Ollama) - Open-source GPT model via Ollama

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your HG 585 document** as `data/raw/source.pdf`

3. **Run the complete pipeline**:
   ```bash
   python src/main.py full
   ```

## Features

- **Document Processing**: Extracts and cleans text from PDF and Word documents
- **Data Preparation**: Creates question-answer pairs for training
- **Model Training**: Fine-tunes models using LoRA (Low-Rank Adaptation)
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, F1, exact match
- **Professional Visualizations**: Charts, heatmaps, and comparison reports
- **Romanian Language Support**: Specialized handling for Romanian text

## Results

The system generates:
- Before/after training comparison metrics
- Model performance rankings
- Professional visualizations
- Detailed analysis reports

## Documentation

See the complete documentation in the repository files:
- `README.md` - Full documentation
- `DELIVERY_NOTES.md` - Quick start guide
- `testing_report.md` - System testing results

## Repository Structure

```
hg/
├── src/                    # Source code
│   ├── preprocessing/      # Document processing
│   ├── training/          # Model training scripts
│   ├── evaluation/        # Evaluation and metrics
│   └── main.py           # Main CLI application
├── data/                  # Data directory
│   ├── raw/              # Input documents
│   └── processed/        # Processed datasets
├── models/               # Trained model outputs
├── results/              # Evaluation results
└── requirements.txt      # Dependencies
```

## License

This project is for research and educational purposes.
