"""
Text Cleaner for LLM Training Data
Cleans and normalizes text extracted from documents.
"""

import re
import string
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaning utilities for preprocessing documents."""
    
    def __init__(self):
        # Common patterns to remove or normalize
        self.patterns_to_remove = [
            r'\f',  # Form feed characters
            r'\r',  # Carriage returns
            r'\t+',  # Multiple tabs
            r' {3,}',  # Multiple spaces (3 or more)
            r'\n{3,}',  # Multiple newlines (3 or more)
            r'Page \d+ of \d+',  # Page numbers
            r'^\d+\s*$',  # Lines with only numbers
            r'^[^\w\s]*$',  # Lines with only punctuation
        ]
        
        # Patterns for legal document artifacts
        self.legal_artifacts = [
            r'MONITORUL OFICIAL.*?\d{4}',
            r'HOTĂRÂRE.*?nr\.\s*\d+',
            r'GOVERNMENT DECISION.*?no\.\s*\d+',
            r'UNOFFICIAL TRANSLATION',
            r'ROMANIA\s*GOVERNMENT OF ROMANIA',
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Start with the original text
        cleaned = text
        
        # Remove legal document artifacts
        for pattern in self.legal_artifacts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove common unwanted patterns
        for pattern in self.patterns_to_remove:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n+', '\n', cleaned)  # Multiple newlines to single
        cleaned = re.sub(r' +', ' ', cleaned)    # Multiple spaces to single
        
        # Remove leading/trailing whitespace from each line
        lines = cleaned.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        cleaned = '\n'.join(lines)
        
        return cleaned.strip()
    
    def clean_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Clean a list of paragraphs.
        
        Args:
            paragraphs (List[str]): List of paragraph texts
            
        Returns:
            List[str]: Cleaned paragraphs
        """
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            cleaned = self.clean_text(paragraph)
            
            # Skip very short paragraphs (likely artifacts)
            if len(cleaned) < 10:
                continue
            
            # Skip paragraphs that are mostly numbers or punctuation
            if self._is_mostly_non_text(cleaned):
                continue
            
            cleaned_paragraphs.append(cleaned)
        
        logger.info(f"Cleaned {len(paragraphs)} paragraphs to {len(cleaned_paragraphs)}")
        return cleaned_paragraphs
    
    def _is_mostly_non_text(self, text: str, threshold: float = 0.7) -> bool:
        """
        Check if text is mostly non-alphabetic characters.
        
        Args:
            text (str): Text to check
            threshold (float): Threshold for non-text ratio
            
        Returns:
            bool: True if text is mostly non-alphabetic
        """
        if not text:
            return True
        
        # Count alphabetic characters
        alpha_count = sum(1 for char in text if char.isalpha())
        total_count = len(text)
        
        alpha_ratio = alpha_count / total_count
        return alpha_ratio < (1 - threshold)
    
    def normalize_romanian_text(self, text: str) -> str:
        """
        Normalize Romanian-specific characters and patterns.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Romanian diacritic normalization
        romanian_chars = {
            'ă': 'ă', 'â': 'â', 'î': 'î', 'ș': 'ș', 'ț': 'ț',
            'Ă': 'Ă', 'Â': 'Â', 'Î': 'Î', 'Ș': 'Ș', 'Ț': 'Ț',
            # Handle common encoding issues
            'ã': 'ă', 'ş': 'ș', 'ţ': 'ț',
            'Ã': 'Ă', 'Ş': 'Ș', 'Ţ': 'Ț'
        }
        
        normalized = text
        for old_char, new_char in romanian_chars.items():
            normalized = normalized.replace(old_char, new_char)
        
        return normalized
    
    def extract_meaningful_sentences(self, text: str, min_length: int = 20) -> List[str]:
        """
        Extract meaningful sentences from text.
        
        Args:
            text (str): Input text
            min_length (int): Minimum sentence length
            
        Returns:
            List[str]: List of meaningful sentences
        """
        # Split into sentences using common punctuation
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip short sentences
            if len(sentence) < min_length:
                continue
            
            # Skip sentences that are mostly uppercase (likely headers)
            if sentence.isupper() and len(sentence) > 50:
                continue
            
            # Skip sentences with too many numbers
            digit_ratio = sum(1 for char in sentence if char.isdigit()) / len(sentence)
            if digit_ratio > 0.3:
                continue
            
            meaningful_sentences.append(sentence)
        
        return meaningful_sentences
    
    def prepare_training_text(self, paragraphs: List[str]) -> str:
        """
        Prepare text for LLM training by cleaning and formatting.
        
        Args:
            paragraphs (List[str]): List of paragraphs
            
        Returns:
            str: Formatted training text
        """
        # Clean paragraphs
        cleaned_paragraphs = self.clean_paragraphs(paragraphs)
        
        # Normalize Romanian text
        normalized_paragraphs = [
            self.normalize_romanian_text(p) for p in cleaned_paragraphs
        ]
        
        # Join with double newlines for clear separation
        training_text = '\n\n'.join(normalized_paragraphs)
        
        logger.info(f"Prepared training text with {len(normalized_paragraphs)} paragraphs")
        logger.info(f"Total characters: {len(training_text)}")
        
        return training_text


def main():
    """Test the text cleaner."""
    cleaner = TextCleaner()
    
    # Test with sample text
    sample_text = """
    UNOFFICIAL TRANSLATION
    
    ROMANIA
    GOVERNMENT OF ROMANIA
    
    GOVERNMENT DECISION no. 585/2002
    
    NATIONAL STANDARDS ON THE PROTECTION OF
    CLASSIFIED INFORMATION IN ROMANIA
    
    
    Article 1
    The national standards for the protection of classified information in Romania are approved.
    
    
    Page 1 of 84
    
    
    Article 2
    These standards establish the requirements for protecting classified information.
    """
    
    print("Original text:")
    print(repr(sample_text))
    print("\nCleaned text:")
    cleaned = cleaner.clean_text(sample_text)
    print(repr(cleaned))
    
    # Test paragraph cleaning
    paragraphs = sample_text.split('\n\n')
    cleaned_paragraphs = cleaner.clean_paragraphs(paragraphs)
    print(f"\nCleaned paragraphs ({len(cleaned_paragraphs)}):")
    for i, p in enumerate(cleaned_paragraphs):
        print(f"{i+1}: {p}")


if __name__ == "__main__":
    main()

