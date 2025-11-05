"""
Text Segmenter for LLM Training
Segments text into appropriate chunks for training and creates Q&A pairs.
"""

import re
import json
import random
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSegmenter:
    """Segments text into training chunks and generates Q&A pairs."""
    
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Question templates for different types of content
        self.question_templates = {
            'definition': [
                "What is {term}?",
                "How is {term} defined?",
                "What does {term} mean?",
                "Explain {term}.",
            ],
            'procedure': [
                "What is the procedure for {action}?",
                "How should {action} be performed?",
                "What are the steps for {action}?",
                "Describe the process of {action}.",
            ],
            'requirement': [
                "What are the requirements for {subject}?",
                "What must be done regarding {subject}?",
                "What are the obligations concerning {subject}?",
                "What conditions apply to {subject}?",
            ],
            'general': [
                "What does this text say about {topic}?",
                "According to the document, what is stated about {topic}?",
                "What information is provided about {topic}?",
                "What can you tell me about {topic}?",
            ]
        }
    
    def segment_text(self, text: str) -> List[str]:
        """
        Segment text into chunks suitable for training.
        
        Args:
            text (str): Input text to segment
            
        Returns:
            List[str]: List of text chunks
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed max size, save current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.overlap_size//10:] if len(current_chunk) > self.overlap_size//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        logger.info(f"Segmented text into {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex to split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def generate_qa_pairs(self, chunks: List[str], num_pairs_per_chunk: int = 2) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from text chunks.
        
        Args:
            chunks (List[str]): Text chunks
            num_pairs_per_chunk (int): Number of Q&A pairs to generate per chunk
            
        Returns:
            List[Dict]: List of Q&A pairs
        """
        qa_pairs = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Extract key terms and concepts from the chunk
            key_terms = self._extract_key_terms(chunk)
            
            # Generate questions for this chunk
            chunk_questions = self._generate_questions_for_chunk(chunk, key_terms, num_pairs_per_chunk)
            
            for question in chunk_questions:
                qa_pair = {
                    'question': question,
                    'answer': chunk,
                    'chunk_id': chunk_idx,
                    'source': 'HG_585'
                }
                qa_pairs.append(qa_pair)
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {len(chunks)} chunks")
        return qa_pairs
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text chunk."""
        # Look for capitalized terms (likely important concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        
        # Look for numbered items or articles
        numbered_items = re.findall(r'Article\s+\d+|Section\s+\d+|Chapter\s+\d+', text, re.IGNORECASE)
        
        # Combine and deduplicate
        all_terms = capitalized_terms + quoted_terms + numbered_items
        unique_terms = list(set(all_terms))
        
        # Filter out very common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'Article', 'Section', 'Chapter'}
        filtered_terms = [term for term in unique_terms if term not in common_words and len(term) > 2]
        
        return filtered_terms[:5]  # Return top 5 terms
    
    def _generate_questions_for_chunk(self, chunk: str, key_terms: List[str], num_questions: int) -> List[str]:
        """Generate questions for a specific chunk."""
        questions = []
        
        # Determine question types based on content
        question_types = self._classify_content_type(chunk)
        
        for i in range(num_questions):
            if key_terms and i < len(key_terms):
                # Use specific terms
                term = key_terms[i]
                question_type = random.choice(question_types)
                template = random.choice(self.question_templates[question_type])
                
                if '{term}' in template:
                    question = template.format(term=term)
                elif '{action}' in template:
                    question = template.format(action=term.lower())
                elif '{subject}' in template:
                    question = template.format(subject=term.lower())
                elif '{topic}' in template:
                    question = template.format(topic=term.lower())
                else:
                    question = template
            else:
                # Use general questions
                template = random.choice(self.question_templates['general'])
                topic = "classified information" if "classified" in chunk.lower() else "this topic"
                question = template.format(topic=topic)
            
            questions.append(question)
        
        return questions
    
    def _classify_content_type(self, text: str) -> List[str]:
        """Classify the type of content to determine appropriate question types."""
        content_types = []
        
        text_lower = text.lower()
        
        # Check for definitions
        if any(word in text_lower for word in ['means', 'defined as', 'refers to', 'is understood as']):
            content_types.append('definition')
        
        # Check for procedures
        if any(word in text_lower for word in ['shall', 'must', 'procedure', 'process', 'steps']):
            content_types.append('procedure')
        
        # Check for requirements
        if any(word in text_lower for word in ['required', 'obligation', 'mandatory', 'compliance']):
            content_types.append('requirement')
        
        # Default to general if no specific type found
        if not content_types:
            content_types.append('general')
        
        return content_types
    
    def create_training_dataset(self, qa_pairs: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """
        Split Q&A pairs into training and evaluation datasets.
        
        Args:
            qa_pairs (List[Dict]): List of Q&A pairs
            train_ratio (float): Ratio of data for training
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Training and evaluation datasets
        """
        # Shuffle the data
        shuffled_pairs = qa_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Split into train and eval
        split_idx = int(len(shuffled_pairs) * train_ratio)
        train_data = shuffled_pairs[:split_idx]
        eval_data = shuffled_pairs[split_idx:]
        
        logger.info(f"Created training dataset with {len(train_data)} examples")
        logger.info(f"Created evaluation dataset with {len(eval_data)} examples")
        
        return train_data, eval_data
    
    def save_dataset(self, data: List[Dict[str, str]], output_path: str):
        """Save dataset to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(data)} examples to {output_path}")


def main():
    """Test the segmenter."""
    segmenter = TextSegmenter(max_chunk_size=200, overlap_size=20)
    
    # Test with sample text
    sample_text = """
    Article 1. The national standards for the protection of classified information in Romania are approved.
    These standards establish the requirements for protecting classified information.
    
    Article 2. Classified information means any information that requires protection against unauthorized disclosure.
    The classification levels are: SECRET, CONFIDENTIAL, and RESTRICTED.
    
    Article 3. Access to classified information shall be granted only to authorized personnel.
    Authorization procedures must be followed according to established protocols.
    """
    
    print("Segmenting text...")
    chunks = segmenter.segment_text(sample_text)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: {chunk}")
    
    print("\nGenerating Q&A pairs...")
    qa_pairs = segmenter.generate_qa_pairs(chunks, num_pairs_per_chunk=2)
    print(f"Generated {len(qa_pairs)} Q&A pairs:")
    for i, pair in enumerate(qa_pairs[:4]):  # Show first 4
        print(f"\nQ&A {i+1}:")
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer'][:100]}...")


if __name__ == "__main__":
    main()

