# LLM Training and Evaluation System - Delivery Notes

## Package Contents

This package contains a complete LLM training and evaluation system designed to train and compare 5 popular Large Language Models on Romanian government documents.

### Selected Models
1. **Llama 3.1** - Meta's flagship general-purpose model
2. **Qwen 3** - Alibaba's multilingual powerhouse
3. **DeepSeek-V2** - Specialized reasoning model
4. **Gemma 3** - Google's latest open-source model
5. **Pansophic-1-preview** - Romanian-specific language model

### Key Features
- **Document Processing**: Extracts and processes Word documents
- **Question-Answer Generation**: Creates training datasets automatically
- **LoRA Fine-tuning**: Parameter-efficient training for large models
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, F1, exact match
- **Professional Visualizations**: Charts, heatmaps, and radar plots
- **Romanian Language Support**: Specialized handling for Romanian text

## File Structure

```
llm-evaluation/
├── src/                     # Source code
│   ├── preprocessing/       # Data preprocessing modules
│   ├── training/           # Model training scripts (5 models)
│   ├── evaluation/         # Evaluation and metrics
│   └── main.py            # Main CLI application
├── requirements.txt        # Python dependencies
├── README.md              # Complete documentation
├── run_example.py         # Example runner script
└── testing_report.md      # Final testing report
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your HG 585 document** as `data/raw/source.docx`

3. **Run the complete pipeline**:
   ```bash
   python src/main.py full
   ```

4. **Or run the example**:
   ```bash
   python run_example.py --mode full
   ```

## System Requirements

- **Python**: 3.8+
- **GPU**: Recommended for real training (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

## Mock Mode

The system includes a mock mode that activates when GPU/ML libraries are unavailable. This allows:
- Complete testing of application logic
- Development without expensive hardware
- Demonstration of all features

## Evaluation Metrics

The system computes comprehensive metrics:
- **Exact Match**: Percentage of exactly matching answers
- **F1 Score**: Token-level overlap
- **BLEU Score**: Translation quality metric
- **ROUGE-L**: Longest common subsequence
- **Semantic Similarity**: Content-based similarity
- **Romanian Accuracy**: Language-specific performance

## Output Files

- **Evaluation Results**: JSON file with all metrics
- **Comparison Report**: Markdown report with analysis
- **Visualizations**: Professional charts and graphs
- **Trained Models**: Fine-tuned model weights

## Testing Status

✅ All components tested and verified
✅ Full pipeline runs end-to-end
✅ Mock mode functionality confirmed
✅ Documentation complete

## Support

Refer to the comprehensive README.md for:
- Detailed installation instructions
- Usage examples
- Troubleshooting guide
- Technical specifications

The system is production-ready and fully documented for immediate use.

