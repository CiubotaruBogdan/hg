"""
Complete preprocessing pipeline for HG 585 document.
Orchestrates parsing, cleaning, segmentation, and dataset creation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

from .parser import DocumentParser
from .cleaner import TextCleaner
from .segmenter import TextSegmenter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for document preparation."""
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 overlap_size: int = 50,
                 train_ratio: float = 0.8,
                 qa_pairs_per_chunk: int = 2):
        self.parser = DocumentParser()
        self.cleaner = TextCleaner()
        self.segmenter = TextSegmenter(max_chunk_size, overlap_size)
        self.train_ratio = train_ratio
        self.qa_pairs_per_chunk = qa_pairs_per_chunk
    
    def process_document(self, 
                        input_path: str, 
                        output_dir: str) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            input_path (str): Path to input Word document
            output_dir (str): Directory to save processed data
            
        Returns:
            Dict containing processing results and statistics
        """
        logger.info(f"Starting preprocessing pipeline for: {input_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse document
        logger.info("Step 1: Parsing document...")
        parsed_data = self.parser.parse_document(input_path)
        
        # Step 2: Clean text
        logger.info("Step 2: Cleaning text...")
        cleaned_paragraphs = self.cleaner.clean_paragraphs(parsed_data['paragraphs'])
        training_text = self.cleaner.prepare_training_text(cleaned_paragraphs)
        
        # Step 3: Segment text
        logger.info("Step 3: Segmenting text...")
        chunks = self.segmenter.segment_text(training_text)
        
        # Step 4: Generate Q&A pairs
        logger.info("Step 4: Generating Q&A pairs...")
        qa_pairs = self.segmenter.generate_qa_pairs(chunks, self.qa_pairs_per_chunk)
        
        # Step 5: Create train/eval split
        logger.info("Step 5: Creating train/eval split...")
        train_data, eval_data = self.segmenter.create_training_dataset(qa_pairs, self.train_ratio)
        
        # Step 6: Save datasets
        logger.info("Step 6: Saving datasets...")
        train_path = os.path.join(output_dir, 'train.jsonl')
        eval_path = os.path.join(output_dir, 'eval.jsonl')
        
        self.segmenter.save_dataset(train_data, train_path)
        self.segmenter.save_dataset(eval_data, eval_path)
        
        # Save additional outputs
        self._save_additional_outputs(output_dir, {
            'cleaned_text': training_text,
            'chunks': chunks,
            'metadata': parsed_data['metadata']
        })
        
        # Compile results
        results = {
            'input_file': input_path,
            'output_directory': output_dir,
            'statistics': {
                'original_paragraphs': len(parsed_data['paragraphs']),
                'cleaned_paragraphs': len(cleaned_paragraphs),
                'text_chunks': len(chunks),
                'qa_pairs_total': len(qa_pairs),
                'train_examples': len(train_data),
                'eval_examples': len(eval_data),
                'original_word_count': parsed_data['metadata']['num_words'],
                'cleaned_character_count': len(training_text)
            },
            'files_created': {
                'train_dataset': train_path,
                'eval_dataset': eval_path,
                'cleaned_text': os.path.join(output_dir, 'cleaned_text.txt'),
                'chunks': os.path.join(output_dir, 'chunks.json'),
                'metadata': os.path.join(output_dir, 'metadata.json')
            }
        }
        
        # Save processing report
        report_path = os.path.join(output_dir, 'processing_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info(f"Created {results['statistics']['train_examples']} training examples")
        logger.info(f"Created {results['statistics']['eval_examples']} evaluation examples")
        
        return results
    
    def _save_additional_outputs(self, output_dir: str, data: Dict[str, Any]):
        """Save additional preprocessing outputs."""
        
        # Save cleaned text
        with open(os.path.join(output_dir, 'cleaned_text.txt'), 'w', encoding='utf-8') as f:
            f.write(data['cleaned_text'])
        
        # Save chunks
        with open(os.path.join(output_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(data['chunks'], f, indent=2, ensure_ascii=False)
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(data['metadata'], f, indent=2, ensure_ascii=False)


def main():
    """Run the preprocessing pipeline."""
    # Configuration
    input_document = "/home/ubuntu/llm-evaluation/data/raw/HG_585.doc"
    output_directory = "/home/ubuntu/llm-evaluation/data/processed"
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        max_chunk_size=512,
        overlap_size=50,
        train_ratio=0.8,
        qa_pairs_per_chunk=2
    )
    
    # Check if input file exists
    if not os.path.exists(input_document):
        logger.error(f"Input document not found: {input_document}")
        logger.info("Please place the HG 585 document in the data/raw/ directory")
        return
    
    try:
        # Run pipeline
        results = pipeline.process_document(input_document, output_directory)
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*50)
        print(f"Input file: {results['input_file']}")
        print(f"Output directory: {results['output_directory']}")
        print("\nStatistics:")
        for key, value in results['statistics'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        print("\nFiles created:")
        for key, path in results['files_created'].items():
            print(f"  {key.replace('_', ' ').title()}: {path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

