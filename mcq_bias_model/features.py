"""
Feature Extraction Module for MCQ Bias Prediction
AI/ML Engineer Implementation - Focus on Quiz-Writing Biases

This module extracts bias-based features from multiple-choice questions
to predict correct answers without domain knowledge.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import string


class MCQBiasFeatureExtractor:
    """
    Advanced feature extractor for detecting quiz-writing biases in MCQs.
    
    Features focus on:
    1. Option length patterns (longest/shortest bias)
    2. Keyword detection (common answer patterns)
    3. Numeric option biases (middle value, mathematical patterns)
    4. Option overlap analysis (similarity patterns)
    5. Contextual features (position, frequency patterns)
    6. Rule-based scoring (composite bias scores)
    """
    
    def __init__(self):
        self.bias_keywords = {
            'all_above': ['all of the above', 'all above', 'all of these', 'all the above'],
            'none_above': ['none of the above', 'none above', 'none of these', 'none mentioned'],
            'both_options': ['both a and b', 'both b and c', 'both a and c', 'both options'],
            'qualifier_words': ['most', 'least', 'best', 'worst', 'always', 'never', 'only', 'except'],
            'uncertainty': ['might', 'could', 'possibly', 'probably', 'likely', 'may'],
            'absolute': ['definitely', 'certainly', 'absolutely', 'completely', 'entirely']
        }
        
        # Track patterns across questions for contextual features
        self.question_history = []
        self.answer_frequency = defaultdict(int)
        
    def extract_features(self, question_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract comprehensive bias features from a single MCQ.
        
        Args:
            question_data: Dict with keys: question_text, options, correct_answer, 
                         question_number, exam_id
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic data extraction
        question_text = str(question_data.get('question_text', ''))
        options = question_data.get('options', [])
        correct_answer = str(question_data.get('correct_answer', ''))
        question_number = question_data.get('question_number', 0)
        
        # Ensure we have valid options
        if not options or len(options) < 2:
            return self._get_default_features()
        
        # Extract different feature categories
        features.update(self._extract_length_features(options))
        features.update(self._extract_keyword_features(options))
        features.update(self._extract_numeric_features(options))
        features.update(self._extract_overlap_features(options))
        features.update(self._extract_context_features(question_number, options))
        features.update(self._extract_rule_based_scores(options, question_text))
        
        # Update tracking for contextual features
        self._update_question_history(question_data)
        
        return features
    
    def extract_features_batch(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for a batch of questions efficiently.
        
        Args:
            questions_df: DataFrame with unified schema
            
        Returns:
            DataFrame with original data + feature columns
        """
        print("🤖 AI/ML Engineer: Extracting bias features from MCQ dataset...")
        
        # Reset tracking for batch processing
        self.question_history = []
        self.answer_frequency = defaultdict(int)
        
        feature_rows = []
        
        for idx, row in questions_df.iterrows():
            question_data = {
                'question_text': row['question_text'],
                'options': row['options'],
                'correct_answer': row['correct_answer'],
                'question_number': row['question_number'],
                'exam_id': row['exam_id']
            }
            
            features = self.extract_features(question_data)
            
            # Add original data
            features.update({
                'question_id': row['question_id'],
                'exam_id': row['exam_id'],
                'question_number': row['question_number'],
                'question_text': row['question_text'],
                'options': row['options'],
                'correct_answer': row['correct_answer']
            })
            
            feature_rows.append(features)
            
            # Progress tracking
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,} questions...")
        
        result_df = pd.DataFrame(feature_rows)
        print(f"✅ Feature extraction complete: {len(result_df)} questions, {len([c for c in result_df.columns if c.startswith('feat_')])} features")
        
        return result_df
    
    def _extract_length_features(self, options: List[str]) -> Dict[str, float]:
        """Extract features based on option length patterns."""
        if not options:
            return {}
        
        # Calculate lengths
        char_lengths = [len(str(opt)) for opt in options]
        word_lengths = [len(str(opt).split()) for opt in options]
        
        features = {
            # Character length features
            'feat_char_len_mean': np.mean(char_lengths),
            'feat_char_len_std': np.std(char_lengths),
            'feat_char_len_max': max(char_lengths),
            'feat_char_len_min': min(char_lengths),
            'feat_char_len_range': max(char_lengths) - min(char_lengths),
            
            # Word length features  
            'feat_word_len_mean': np.mean(word_lengths),
            'feat_word_len_std': np.std(word_lengths),
            'feat_word_len_max': max(word_lengths),
            'feat_word_len_min': min(word_lengths),
            
            # Length bias indicators
            'feat_longest_option_bias': 1.0 if max(char_lengths) == char_lengths[0] else 0.0,
            'feat_shortest_option_bias': 1.0 if min(char_lengths) == char_lengths[0] else 0.0,
        }
        
        # Per-option length features (for 4 options)
        for i in range(min(4, len(options))):
            features[f'feat_opt_{i}_char_len'] = char_lengths[i] if i < len(char_lengths) else 0
            features[f'feat_opt_{i}_word_len'] = word_lengths[i] if i < len(word_lengths) else 0
            
        return features
    
    def _extract_keyword_features(self, options: List[str]) -> Dict[str, float]:
        """Extract features based on bias-indicating keywords."""
        features = {}
        
        for i, option in enumerate(options):
            option_lower = str(option).lower()
            
            # Check for each keyword category
            for category, keywords in self.bias_keywords.items():
                has_keyword = any(keyword in option_lower for keyword in keywords)
                features[f'feat_opt_{i}_{category}'] = 1.0 if has_keyword else 0.0
            
            # Additional keyword features
            features[f'feat_opt_{i}_has_numbers'] = 1.0 if re.search(r'\d', option_lower) else 0.0
            features[f'feat_opt_{i}_has_punctuation'] = 1.0 if any(p in option_lower for p in '.,;:!?') else 0.0
            features[f'feat_opt_{i}_is_single_word'] = 1.0 if len(str(option).split()) == 1 else 0.0
        
        # Global keyword presence
        all_options_text = ' '.join([str(opt).lower() for opt in options])
        for category, keywords in self.bias_keywords.items():
            has_any = any(keyword in all_options_text for keyword in keywords)
            features[f'feat_has_{category}'] = 1.0 if has_any else 0.0
            
        return features
    
    def _extract_numeric_features(self, options: List[str]) -> Dict[str, float]:
        """Extract features for numeric options (common in MCQs)."""
        features = {
            'feat_all_numeric': 0.0,
            'feat_numeric_count': 0.0,
            'feat_numeric_middle_bias': 0.0,
            'feat_numeric_extreme_bias': 0.0,
            'feat_numeric_ascending': 0.0,
            'feat_numeric_descending': 0.0
        }
        
        # Extract numbers from options
        numeric_values = []
        numeric_indices = []
        
        for i, option in enumerate(options):
            # Try to extract number from option text
            numbers = re.findall(r'-?\d+\.?\d*', str(option))
            if numbers:
                try:
                    value = float(numbers[0])  # Take first number found
                    numeric_values.append(value)
                    numeric_indices.append(i)
                except ValueError:
                    continue
        
        if len(numeric_values) >= 3:  # Need at least 3 numeric options for meaningful analysis
            features['feat_all_numeric'] = 1.0 if len(numeric_values) == len(options) else 0.0
            features['feat_numeric_count'] = len(numeric_values) / len(options)
            
            # Sort analysis
            sorted_values = sorted(numeric_values)
            is_ascending = numeric_values == sorted_values
            is_descending = numeric_values == sorted_values[::-1]
            
            features['feat_numeric_ascending'] = 1.0 if is_ascending else 0.0
            features['feat_numeric_descending'] = 1.0 if is_descending else 0.0
            
            # Middle value bias (common bias: correct answer is middle value)
            if len(sorted_values) >= 3:
                middle_values = sorted_values[1:-1]  # Exclude min and max
                first_option_value = numeric_values[0] if numeric_values else None
                
                if first_option_value in middle_values:
                    features['feat_numeric_middle_bias'] = 1.0
                
                # Extreme value bias
                if first_option_value in [min(sorted_values), max(sorted_values)]:
                    features['feat_numeric_extreme_bias'] = 1.0
        
        return features
    
    def _extract_overlap_features(self, options: List[str]) -> Dict[str, float]:
        """Extract features based on text overlap between options."""
        features = {}
        
        if len(options) < 2:
            return features
        
        # Calculate pairwise token overlaps
        overlaps = []
        
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                opt1_tokens = set(str(options[i]).lower().split())
                opt2_tokens = set(str(options[j]).lower().split())
                
                if len(opt1_tokens) > 0 and len(opt2_tokens) > 0:
                    intersection = len(opt1_tokens & opt2_tokens)
                    union = len(opt1_tokens | opt2_tokens)
                    
                    jaccard = intersection / union if union > 0 else 0
                    overlaps.append(jaccard)
        
        if overlaps:
            features['feat_overlap_mean'] = np.mean(overlaps)
            features['feat_overlap_max'] = max(overlaps)
            features['feat_overlap_min'] = min(overlaps)
            features['feat_overlap_std'] = np.std(overlaps)
        
        # Common prefix/suffix analysis
        common_prefixes = self._find_common_prefixes(options)
        common_suffixes = self._find_common_suffixes(options)
        
        features['feat_has_common_prefix'] = 1.0 if len(common_prefixes) > 0 else 0.0
        features['feat_has_common_suffix'] = 1.0 if len(common_suffixes) > 0 else 0.0
        features['feat_common_prefix_length'] = max([len(p) for p in common_prefixes]) if common_prefixes else 0
        
        return features
    
    def _extract_context_features(self, question_number: int, options: List[str]) -> Dict[str, float]:
        """Extract contextual features based on question position and patterns."""
        features = {
            'feat_question_number': question_number,
            'feat_question_position_norm': question_number / 100.0,  # Normalize to 0-1 range
        }
        
        # Answer pattern analysis (based on history)
        if len(self.question_history) >= 5:
            recent_answers = [q.get('correct_answer', '') for q in self.question_history[-5:]]
            recent_positions = []
            
            for answer in recent_answers:
                for i, opt in enumerate(options):
                    if str(opt) == str(answer):
                        recent_positions.append(i)
                        break
            
            if recent_positions:
                features['feat_recent_answer_position_mean'] = np.mean(recent_positions)
                features['feat_answer_position_std'] = np.std(recent_positions) if len(recent_positions) > 1 else 0
            else:
                # Default values when no valid positions found
                features['feat_recent_answer_position_mean'] = 1.5  # Middle position
                features['feat_answer_position_std'] = 1.0
        else:
            # Default values when insufficient history
            features['feat_recent_answer_position_mean'] = 1.5  # Middle position (0-3 range, so 1.5 is center)
            features['feat_answer_position_std'] = 1.0  # Moderate variance
        
        # Option frequency patterns (A, B, C, D bias)
        for i in range(min(4, len(options))):
            option_letter = chr(ord('A') + i)
            frequency = self.answer_frequency.get(option_letter, 0)
            total_questions = len(self.question_history)
            
            if total_questions > 0:
                features[f'feat_option_{option_letter.lower()}_frequency'] = frequency / total_questions
            else:
                features[f'feat_option_{option_letter.lower()}_frequency'] = 0.25  # Default uniform
        
        return features
    
    def _extract_rule_based_scores(self, options: List[str], question_text: str) -> Dict[str, float]:
        """Create composite rule-based bias scores."""
        features = {}
        
        # Longest option score (bias toward longest answer)
        if options:
            lengths = [len(str(opt)) for opt in options]
            longest_idx = lengths.index(max(lengths))
            features['feat_longest_option_score'] = 1.0 if longest_idx == 0 else 0.5
        
        # Keyword density score
        question_lower = str(question_text).lower()
        keyword_scores = []
        
        for option in options:
            option_lower = str(option).lower()
            score = 0
            
            # Bonus for "all of the above" type answers
            if any(keyword in option_lower for keyword in self.bias_keywords['all_above']):
                score += 0.8
            
            # Penalty for absolute words (often wrong in good MCQs)
            if any(keyword in option_lower for keyword in self.bias_keywords['absolute']):
                score -= 0.3
            
            # Bonus for qualifier words (often correct)
            if any(keyword in option_lower for keyword in self.bias_keywords['qualifier_words']):
                score += 0.2
            
            keyword_scores.append(score)
        
        # Normalize scores
        if keyword_scores:
            max_score = max(keyword_scores)
            features['feat_first_option_keyword_score'] = keyword_scores[0] / max(max_score, 1.0)
            features['feat_max_keyword_score'] = max_score
        
        # Question complexity score (longer questions often have more specific answers)
        question_words = len(str(question_text).split())
        features['feat_question_complexity'] = min(question_words / 20.0, 1.0)  # Normalize
        
        return features
    
    def _find_common_prefixes(self, options: List[str]) -> List[str]:
        """Find common prefixes among options."""
        if len(options) < 2:
            return []
        
        prefixes = []
        for length in range(1, min(10, min(len(str(opt)) for opt in options)) + 1):
            prefix = str(options[0])[:length].lower()
            if all(str(opt)[:length].lower() == prefix for opt in options):
                prefixes.append(prefix)
        
        return prefixes
    
    def _find_common_suffixes(self, options: List[str]) -> List[str]:
        """Find common suffixes among options."""
        if len(options) < 2:
            return []
        
        suffixes = []
        for length in range(1, min(10, min(len(str(opt)) for opt in options)) + 1):
            suffix = str(options[0])[-length:].lower()
            if all(str(opt)[-length:].lower() == suffix for opt in options):
                suffixes.append(suffix)
        
        return suffixes
    
    def _update_question_history(self, question_data: Dict[str, Any]):
        """Update tracking data for contextual features."""
        self.question_history.append(question_data)
        
        # Update answer frequency (A, B, C, D pattern)
        correct_answer = str(question_data.get('correct_answer', ''))
        options = question_data.get('options', [])
        
        for i, option in enumerate(options):
            if str(option) == correct_answer:
                option_letter = chr(ord('A') + i)
                self.answer_frequency[option_letter] += 1
                break
        
        # Keep history manageable
        if len(self.question_history) > 1000:
            self.question_history = self.question_history[-500:]
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values for invalid inputs."""
        default_features = {}
        
        # Length features
        for feature in ['feat_char_len_mean', 'feat_char_len_std', 'feat_char_len_max', 
                       'feat_char_len_min', 'feat_char_len_range', 'feat_word_len_mean', 
                       'feat_word_len_std', 'feat_word_len_max', 'feat_word_len_min']:
            default_features[feature] = 0.0
        
        # Option-specific features
        for i in range(4):
            for suffix in ['char_len', 'word_len', 'all_above', 'none_above', 'both_options', 
                          'qualifier_words', 'uncertainty', 'absolute', 'has_numbers', 
                          'has_punctuation', 'is_single_word']:
                default_features[f'feat_opt_{i}_{suffix}'] = 0.0
        
        return default_features
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions and potential importance."""
        feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
        
        analysis = {
            'total_features': len(feature_cols),
            'feature_statistics': {},
            'bias_indicators': {},
            'recommendations': []
        }
        
        for col in feature_cols:
            stats = {
                'mean': features_df[col].mean(),
                'std': features_df[col].std(),
                'min': features_df[col].min(),
                'max': features_df[col].max(),
                'non_zero_percentage': (features_df[col] != 0).mean() * 100
            }
            analysis['feature_statistics'][col] = stats
        
        # Identify potential bias indicators
        high_variance_features = [col for col in feature_cols 
                                if features_df[col].std() > 0.1]
        analysis['bias_indicators']['high_variance_features'] = high_variance_features
        
        # Generate recommendations
        if len(high_variance_features) > 10:
            analysis['recommendations'].append("Good feature diversity - multiple bias patterns detected")
        else:
            analysis['recommendations'].append("Consider adding more diverse bias features")
        
        return analysis


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("🤖 AI/ML Engineer: Demonstrating MCQ Bias Feature Extraction")
    print("=" * 60)
    
    # Import data loader
    from data_loader import DataLoader
    
    # Load a sample dataset
    loader = DataLoader()
    try:
        # Use a smaller dataset for demo
        sample_data = loader.load_dataset('../data/mental_health/mhqa.csv', 'mental_health')
        
        # Take first 100 questions for demo
        sample_data = sample_data.head(100)
        
        print(f"📊 Loaded {len(sample_data)} sample questions for feature extraction demo")
        
        # Initialize feature extractor
        extractor = MCQBiasFeatureExtractor()
        
        # Extract features
        features_df = extractor.extract_features_batch(sample_data)
        
        # Analyze features
        analysis = extractor.get_feature_importance_analysis(features_df)
        
        print(f"\n✨ Feature Extraction Results:")
        print(f"   Total Features: {analysis['total_features']}")
        print(f"   High Variance Features: {len(analysis['bias_indicators']['high_variance_features'])}")
        
        # Show sample features
        feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
        sample_features = features_df[feature_cols].head(3)
        
        print(f"\n📋 Sample Features (first 3 questions, first 10 features):")
        print(sample_features.iloc[:, :10].to_string())
        
        return features_df, analysis
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None, None


if __name__ == "__main__":
    demo_feature_extraction()
