#!/usr/bin/env python3
"""
Example Runner for LLM Evaluation System
Demonstrates how to use the system with sample data.
"""

import os
import sys
import logging
from docx import Document

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import LLMEvaluationApp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_document():
    """Create a sample document for testing."""
    sample_content = """
ROMANIA
GOVERNMENT OF ROMANIA

GOVERNMENT DECISION no. 585/2002

NATIONAL STANDARDS ON THE PROTECTION OF
CLASSIFIED INFORMATION IN ROMANIA

Article 1
The national standards for the protection of classified information in Romania are approved, as set out in the annex which forms an integral part of this decision.

Article 2
Classified information means any information that requires protection against unauthorized disclosure in the interest of national security, public order, or other legitimate interests.

Article 3
The classification levels for classified information are:
a) SECRET - information whose unauthorized disclosure could cause exceptionally grave damage to national security;
b) CONFIDENTIAL - information whose unauthorized disclosure could cause serious damage to national security;
c) RESTRICTED - information whose unauthorized disclosure could cause damage to national security.

Article 4
Access to classified information shall be granted only to persons who:
a) have a legitimate need to know the information;
b) have been granted appropriate security clearance;
c) have signed a confidentiality agreement.

Article 5
The protection of classified information shall be ensured through:
a) physical security measures;
b) personnel security measures;
c) information security measures;
d) industrial security measures.

Article 6
Classified information shall be marked with appropriate classification markings indicating the level of classification and any special handling requirements.

Article 7
The transmission of classified information shall be conducted through secure channels and in accordance with established procedures.

Article 8
The storage of classified information shall be in approved security containers or facilities that provide adequate protection against unauthorized access.

Article 9
The destruction of classified information shall be conducted in a manner that ensures complete destruction and prevents reconstruction of the information.

Article 10
Violations of these standards may result in administrative, civil, or criminal penalties as provided by law.

Article 11
These standards shall enter into force on the date of publication in the Official Gazette of Romania.
"""
    
    # Create sample document
    sample_file = "data/raw/sample_HG_585.docx"
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    
    # Create a new document
    document = Document()
    document.add_paragraph(sample_content)
    document.save(sample_file)
    
    logger.info(f"Created sample document: {sample_file}")
    return sample_file


def run_example():
    """Run a complete example of the LLM evaluation system."""
    logger.info("Starting LLM Evaluation System Example")
    logger.info("=" * 50)
    
    # Create application
    app = LLMEvaluationApp()
    
    # Create sample document (since we don't have the actual HG 585 doc)
    sample_doc = create_sample_document()
    
    # Configuration for quick testing
    test_config = {
        'epochs': 1,  # Very short training for demo
        'batch_size': 1,
        'max_length': 128,  # Short sequences for speed
        'save_steps': 100,
        'eval_steps': 50,
        'logging_steps': 10
    }
    
    # Select subset of models for demo (to save time)
    demo_models = ['llama3', 'pansophic']  # Test with 2 models
    
    try:
        logger.info("Step 1: Running preprocessing...")
        if not app.run_preprocessing(sample_doc, force=True):
            logger.error("Preprocessing failed!")
            return False
        
        logger.info("Step 2: Running training...")
        if not app.run_training(demo_models, test_config):
            logger.error("Training failed!")
            return False
        
        logger.info("Step 3: Running evaluation...")
        if not app.run_evaluation(demo_models):
            logger.error("Evaluation failed!")
            return False
        
        logger.info("Example completed successfully!")
        logger.info("=" * 50)
        
        # Show final status
        app.get_status()
        
        # Show results location
        print("\nResults can be found in:")
        print(f"- Evaluation results: {app.results_dir}/evaluation_results.json")
        print(f"- Comparison report: {app.results_dir}/comparison_report.md")
        print(f"- Visualizations: {app.results_dir}/visualizations/")
        
        return True
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        return False


def run_quick_test():
    """Run a quick test of individual components."""
    logger.info("Running quick component tests...")
    
    app = LLMEvaluationApp()
    
    # Test 1: Preprocessing only
    logger.info("Test 1: Preprocessing")
    sample_doc = create_sample_document()
    
    if app.run_preprocessing(sample_doc, force=True):
        logger.info("✓ Preprocessing test passed")
    else:
        logger.error("✗ Preprocessing test failed")
        return False
    
    # Test 2: Mock training (single model)
    logger.info("Test 2: Mock training")
    test_config = {'epochs': 1, 'batch_size': 1, 'max_length': 64}
    
    if app.run_training(['llama3'], test_config):
        logger.info("✓ Training test passed")
    else:
        logger.error("✗ Training test failed")
        return False
    
    # Test 3: Evaluation
    logger.info("Test 3: Evaluation")
    
    if app.run_evaluation(['llama3'], ['before_training'], create_visualizations=False):
        logger.info("✓ Evaluation test passed")
    else:
        logger.error("✗ Evaluation test failed")
        return False
    
    logger.info("All component tests passed!")
    return True


def main():
    """Main function with command line options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Evaluation System Example")
    parser.add_argument('--mode', choices=['full', 'quick'], default='quick',
                       help='Run mode: full example or quick test')
    parser.add_argument('--models', nargs='+', 
                       choices=['llama3', 'qwen3', 'deepseek', 'gemma3', 'pansophic'],
                       help='Models to test (for full mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        models = args.models or ['llama3', 'pansophic']
        logger.info(f"Running full example with models: {models}")
        
        # Update demo models if specified
        global demo_models
        demo_models = models
        
        success = run_example()
    else:
        logger.info("Running quick component tests")
        success = run_quick_test()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Check the results in the 'results/' directory")
        print("2. View the comparison report (comparison_report.md)")
        print("3. Examine the visualizations")
        print("4. Try running with your own HG 585 document")
    else:
        print("\n❌ Example failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

