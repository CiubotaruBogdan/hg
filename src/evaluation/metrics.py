"""
Evaluation Metrics for LLM Performance Assessment
Implements various metrics to evaluate LLM performance on question-answering tasks.
"""

import re
import math
import string
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Class for computing various evaluation metrics."""
    
    def __init__(self):
        self.metrics_computed = {}
    
    def compute_all_metrics(self, 
                          predictions: List[str], 
                          references: List[str],
                          questions: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions (List[str]): Model predictions
            references (List[str]): Ground truth references
            questions (List[str], optional): Original questions
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        metrics = {}
        
        # Basic metrics
        metrics['exact_match'] = self.exact_match_score(predictions, references)
        metrics['f1_score'] = self.f1_score(predictions, references)
        metrics['bleu_score'] = self.bleu_score(predictions, references)
        metrics['rouge_l'] = self.rouge_l_score(predictions, references)
        
        # Semantic similarity metrics
        metrics['semantic_similarity'] = self.semantic_similarity(predictions, references)
        metrics['answer_relevance'] = self.answer_relevance(predictions, references)
        
        # Length-based metrics
        metrics['avg_prediction_length'] = self.average_length(predictions)
        metrics['avg_reference_length'] = self.average_length(references)
        metrics['length_ratio'] = metrics['avg_prediction_length'] / max(metrics['avg_reference_length'], 1)
        
        # Romanian-specific metrics
        metrics['romanian_accuracy'] = self.romanian_specific_accuracy(predictions, references)
        
        # Question-specific metrics (if questions provided)
        if questions:
            metrics['question_answering_accuracy'] = self.question_answering_accuracy(
                predictions, references, questions
            )
        
        self.metrics_computed = metrics
        logger.info(f"Computed {len(metrics)} evaluation metrics")
        
        return metrics
    
    def exact_match_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match score."""
        if not predictions or not references:
            return 0.0
        
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if self._normalize_text(pred) == self._normalize_text(ref):
                exact_matches += 1
        
        return exact_matches / len(predictions)
    
    def f1_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute F1 score based on token overlap."""
        if not predictions or not references:
            return 0.0
        
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue
            elif not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)
    
    def bleu_score(self, predictions: List[str], references: List[str], n: int = 4) -> float:
        """Compute BLEU score."""
        if not predictions or not references:
            return 0.0
        
        try:
            # Try to use nltk if available
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(predictions, references):
                pred_tokens = self._tokenize(pred)
                ref_tokens = [self._tokenize(ref)]  # BLEU expects list of reference lists
                
                if not pred_tokens or not ref_tokens[0]:
                    bleu_scores.append(0.0)
                    continue
                
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            
            return sum(bleu_scores) / len(bleu_scores)
            
        except ImportError:
            # Fallback to simple n-gram overlap
            return self._simple_bleu(predictions, references, n)
    
    def _simple_bleu(self, predictions: List[str], references: List[str], n: int = 4) -> float:
        """Simple BLEU implementation without nltk."""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if not pred_tokens or not ref_tokens:
                bleu_scores.append(0.0)
                continue
            
            # Compute n-gram precisions
            precisions = []
            for i in range(1, min(n + 1, len(pred_tokens) + 1)):
                pred_ngrams = self._get_ngrams(pred_tokens, i)
                ref_ngrams = self._get_ngrams(ref_tokens, i)
                
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / sum(pred_ngrams.values())
                precisions.append(precision)
            
            if not precisions or all(p == 0 for p in precisions):
                bleu_scores.append(0.0)
                continue
            
            # Geometric mean of precisions
            log_sum = sum(math.log(p) for p in precisions if p > 0)
            geo_mean = math.exp(log_sum / len(precisions))
            
            # Brevity penalty
            bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
            
            bleu = bp * geo_mean
            bleu_scores.append(bleu)
        
        return sum(bleu_scores) / len(bleu_scores)
    
    def rouge_l_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute ROUGE-L score."""
        if not predictions or not references:
            return 0.0
        
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if not pred_tokens or not ref_tokens:
                rouge_scores.append(0.0)
                continue
            
            # Longest Common Subsequence
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            if lcs_length == 0:
                rouge_scores.append(0.0)
                continue
            
            precision = lcs_length / len(pred_tokens)
            recall = lcs_length / len(ref_tokens)
            
            if precision + recall == 0:
                rouge_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                rouge_scores.append(f1)
        
        return sum(rouge_scores) / len(rouge_scores)
    
    def semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Compute semantic similarity (simplified version)."""
        if not predictions or not references:
            return 0.0
        
        # Simple semantic similarity based on word overlap and length
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_words = set(self._tokenize(pred.lower()))
            ref_words = set(self._tokenize(ref.lower()))
            
            if not pred_words and not ref_words:
                similarities.append(1.0)
                continue
            elif not pred_words or not ref_words:
                similarities.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(pred_words.intersection(ref_words))
            union = len(pred_words.union(ref_words))
            
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def answer_relevance(self, predictions: List[str], references: List[str]) -> float:
        """Compute answer relevance score."""
        # For now, use semantic similarity as a proxy for relevance
        return self.semantic_similarity(predictions, references)
    
    def romanian_specific_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute Romanian-specific accuracy metrics."""
        if not predictions or not references:
            return 0.0
        
        # Check for Romanian-specific patterns and diacritics
        romanian_chars = set('ăâîșțĂÂÎȘȚ')
        
        accuracies = []
        for pred, ref in zip(predictions, references):
            # Check if both contain Romanian characters
            pred_has_romanian = any(char in romanian_chars for char in pred)
            ref_has_romanian = any(char in romanian_chars for char in ref)
            
            if ref_has_romanian:
                # If reference has Romanian chars, check if prediction preserves them
                if pred_has_romanian:
                    # Both have Romanian chars, compute character-level accuracy
                    accuracy = self._character_accuracy(pred, ref, romanian_chars)
                else:
                    # Reference has Romanian chars but prediction doesn't
                    accuracy = 0.5  # Partial credit
            else:
                # Reference doesn't have Romanian chars
                if not pred_has_romanian:
                    accuracy = 1.0  # Both don't have Romanian chars
                else:
                    accuracy = 0.8  # Prediction has extra Romanian chars
            
            accuracies.append(accuracy)
        
        return sum(accuracies) / len(accuracies)
    
    def question_answering_accuracy(self, 
                                  predictions: List[str], 
                                  references: List[str],
                                  questions: List[str]) -> float:
        """Compute question-answering specific accuracy."""
        if not predictions or not references or not questions:
            return 0.0
        
        accuracies = []
        for pred, ref, question in zip(predictions, references, questions):
            # Extract key terms from question
            question_terms = set(self._extract_key_terms(question.lower()))
            
            # Check if answer contains relevant terms
            pred_terms = set(self._tokenize(pred.lower()))
            ref_terms = set(self._tokenize(ref.lower()))
            
            # Compute relevance to question
            pred_relevance = len(question_terms.intersection(pred_terms)) / max(len(question_terms), 1)
            ref_relevance = len(question_terms.intersection(ref_terms)) / max(len(question_terms), 1)
            
            # Combine with semantic similarity
            semantic_sim = self.semantic_similarity([pred], [ref])
            
            # Weighted combination
            accuracy = 0.6 * semantic_sim + 0.4 * min(pred_relevance, ref_relevance)
            accuracies.append(accuracy)
        
        return sum(accuracies) / len(accuracies)
    
    def average_length(self, texts: List[str]) -> float:
        """Compute average text length."""
        if not texts:
            return 0.0
        
        total_length = sum(len(text.split()) for text in texts)
        return total_length / len(texts)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _character_accuracy(self, pred: str, ref: str, special_chars: set) -> float:
        """Compute character-level accuracy for special characters."""
        pred_special = [char for char in pred if char in special_chars]
        ref_special = [char for char in ref if char in special_chars]
        
        if not ref_special:
            return 1.0 if not pred_special else 0.8
        
        matches = 0
        for char in ref_special:
            if char in pred_special:
                matches += 1
        
        return matches / len(ref_special)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple key term extraction
        words = self._tokenize(text)
        # Filter out common stop words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'the', 'a', 'an'}
        key_terms = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        return key_terms
    
    def get_metrics_summary(self) -> str:
        """Get a formatted summary of computed metrics."""
        if not self.metrics_computed:
            return "No metrics computed yet."
        
        summary = "Evaluation Metrics Summary:\n"
        summary += "=" * 30 + "\n"
        
        for metric, value in self.metrics_computed.items():
            summary += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        return summary


def main():
    """Test the evaluation metrics."""
    metrics = EvaluationMetrics()
    
    # Test data
    predictions = [
        "The national standards establish requirements for protecting classified information.",
        "Article 1 approves the standards for information protection.",
        "Classification levels include SECRET, CONFIDENTIAL, and RESTRICTED."
    ]
    
    references = [
        "The national standards for the protection of classified information establish requirements.",
        "Article 1 approves the national standards for protecting classified information.",
        "The classification levels are SECRET, CONFIDENTIAL, and RESTRICTED."
    ]
    
    questions = [
        "What do the national standards establish?",
        "What does Article 1 approve?",
        "What are the classification levels?"
    ]
    
    # Compute metrics
    results = metrics.compute_all_metrics(predictions, references, questions)
    
    print("Test Results:")
    print(metrics.get_metrics_summary())


if __name__ == "__main__":
    main()

