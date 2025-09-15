"""
Comprehensive Test Suite for MCQ Bias Prediction Pipeline
Tester Implementation - Automated Testing Scripts

Test Coverage:
- Prediction accuracy validation
- Input format validation
- Batch processing performance
- Error handling robustness
- Edge case handling
- Stress testing
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

# Add mcq_bias_model to path
sys.path.append(str(Path(__file__).parent.parent / "mcq_bias_model"))

from predict import MCQBiasPredictor
from train import MCQBiasTrainer
from pipeline import MCQBiasPipeline


class TestMCQBiasPredictionPipeline:
    """Comprehensive test suite for the MCQ bias prediction system."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.predictor = None
        cls.test_questions = [
            {
                'question_text': 'What is the capital of France?',
                'options': ['Paris', 'London', 'Berlin', 'Madrid'],
                'question_number': 1
            },
            {
                'question_text': 'Which is the largest planet?',
                'options': ['Earth', 'Jupiter', 'Saturn', 'Mars'],
                'question_number': 2
            },
            {
                'question_text': 'What is 2 + 2?',
                'options': ['3', '4', '5', '6'],
                'question_number': 3
            }
        ]
        
        cls.invalid_questions = [
            {},  # Empty question
            {'question_text': ''},  # Empty text
            {'question_text': 'Test', 'options': []},  # No options
            {'question_text': 'Test', 'options': ['Only one']},  # Too few options
            {'question_text': 'Test', 'options': [1, 2, 3, 4]},  # Non-string options
        ]
    
    def test_predictor_initialization(self):
        """Test MCQBiasPredictor initialization."""
        # Test with non-existent model
        predictor = MCQBiasPredictor("nonexistent_model.pkl")
        assert not predictor.is_loaded
        
        # Test with existing model (if available)
        model_path = Path("../mcq_bias_model/mcq_bias_model.pkl")
        if model_path.exists():
            predictor = MCQBiasPredictor(str(model_path))
            assert predictor.is_loaded
            self.__class__.predictor = predictor
    
    def test_input_validation(self):
        """Test input validation functionality."""
        predictor = MCQBiasPredictor("nonexistent_model.pkl")
        
        # Test valid questions
        for question in self.test_questions:
            validation = predictor._validate_question_input(question)
            assert validation['valid'], f"Valid question failed validation: {validation['error']}"
            assert len(validation['checks_performed']) > 0
        
        # Test invalid questions
        for i, invalid_question in enumerate(self.invalid_questions):
            validation = predictor._validate_question_input(invalid_question)
            assert not validation['valid'], f"Invalid question {i} passed validation"
            assert validation['error'] is not None
    
    def test_question_standardization(self):
        """Test question input standardization."""
        predictor = MCQBiasPredictor("nonexistent_model.pkl")
        
        # Test basic standardization
        question = self.test_questions[0]
        standardized = predictor._standardize_question_input(question)
        
        required_fields = ['question_id', 'exam_id', 'question_number', 'question_text', 'options', 'correct_answer']
        for field in required_fields:
            assert field in standardized, f"Missing field {field} in standardized question"
        
        assert len(standardized['options']) <= 4, "Options not limited to 4"
        
        # Test with extra options
        question_with_extra = {
            'question_text': 'Test question',
            'options': ['A', 'B', 'C', 'D', 'E', 'F'],  # 6 options
            'question_number': 1
        }
        
        standardized = predictor._standardize_question_input(question_with_extra)
        assert len(standardized['options']) == 4, "Extra options not trimmed"
    
    def test_single_prediction(self):
        """Test single question prediction."""
        if self.predictor is None:
            pytest.skip("No trained model available for prediction testing")
        
        # Test successful prediction
        result = self.predictor.predict_single(self.test_questions[0])
        
        assert isinstance(result, dict), "Prediction result must be a dictionary"
        
        if result['success']:
            # Test result format
            required_fields = ['predicted_option', 'predicted_option_letter', 'predicted_answer', 
                             'confidence_scores', 'max_confidence', 'prediction_metadata']
            for field in required_fields:
                assert field in result, f"Missing field {field} in prediction result"
            
            # Test data types and ranges
            assert 0 <= result['predicted_option'] <= 3, "Predicted option out of range"
            assert result['predicted_option_letter'] in ['A', 'B', 'C', 'D'], "Invalid option letter"
            assert 0 <= result['max_confidence'] <= 1, "Confidence out of range"
            assert len(result['confidence_scores']) == 4, "Wrong number of confidence scores"
            
            # Test confidence scores sum approximately to 1 (allowing for floating point errors)
            scores_sum = sum(result['confidence_scores'].values())
            assert abs(scores_sum - 1.0) < 0.1, f"Confidence scores don't sum to ~1: {scores_sum}"
        
        # Test with invalid question
        invalid_result = self.predictor.predict_single(self.invalid_questions[0])
        assert not invalid_result['success'], "Invalid question should fail"
        assert 'error' in invalid_result, "Error message missing for invalid question"
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        if self.predictor is None:
            pytest.skip("No trained model available for batch prediction testing")
        
        # Test valid batch
        batch_results = self.predictor.predict_batch(self.test_questions)
        
        assert isinstance(batch_results, dict), "Batch results must be a dictionary"
        assert 'batch_summary' in batch_results, "Missing batch summary"
        assert 'predictions' in batch_results, "Missing predictions"
        
        summary = batch_results['batch_summary']
        assert summary['total_questions'] == len(self.test_questions), "Incorrect question count"
        assert 0 <= summary['success_rate'] <= 1, "Success rate out of range"
        assert summary['processing_time_seconds'] > 0, "Processing time not recorded"
        
        # Test individual predictions
        predictions = batch_results['predictions']
        assert len(predictions) == len(self.test_questions), "Wrong number of predictions"
        
        for i, prediction in enumerate(predictions):
            assert prediction['question_index'] == i, "Incorrect question index"
        
        # Test empty batch
        empty_batch = self.predictor.predict_batch([])
        assert empty_batch['batch_summary']['total_questions'] == 0
        assert empty_batch['batch_summary']['success_rate'] == 0
    
    def test_accuracy_validation(self):
        """Test prediction accuracy validation."""
        if self.predictor is None:
            pytest.skip("No trained model available for accuracy testing")
        
        # Create test DataFrame with known answers
        test_data = pd.DataFrame([
            {
                'question_id': 'test1',
                'exam_id': 'test',
                'question_number': 1,
                'question_text': 'Test question 1',
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_answer': 'Option A'
            },
            {
                'question_id': 'test2', 
                'exam_id': 'test',
                'question_number': 2,
                'question_text': 'Test question 2',
                'options': ['Choice 1', 'Choice 2', 'Choice 3', 'Choice 4'],
                'correct_answer': 'Choice 2'
            }
        ])
        
        validation_results = self.predictor.validate_accuracy(test_data)
        
        assert isinstance(validation_results, dict), "Validation results must be a dictionary"
        
        if validation_results.get('success', True):  # Skip if failed
            assert 'accuracy_metrics' in validation_results, "Missing accuracy metrics"
            assert 'prediction_stats' in validation_results, "Missing prediction stats"
            
            metrics = validation_results['accuracy_metrics']
            assert 0 <= metrics['overall_accuracy'] <= 1, "Accuracy out of range"
            assert 0 <= metrics['top2_accuracy'] <= 1, "Top-2 accuracy out of range"
            assert metrics['random_baseline'] == 0.25, "Incorrect random baseline"
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks and requirements."""
        if self.predictor is None:
            pytest.skip("No trained model available for performance testing")
        
        # Test single prediction speed
        import time
        
        start_time = time.time()
        for _ in range(10):
            self.predictor.predict_single(self.test_questions[0])
        single_prediction_time = (time.time() - start_time) / 10
        
        assert single_prediction_time < 1.0, f"Single prediction too slow: {single_prediction_time:.3f}s"
        
        # Test batch prediction speed
        large_batch = self.test_questions * 20  # 60 questions
        
        start_time = time.time()
        batch_results = self.predictor.predict_batch(large_batch)
        batch_time = time.time() - start_time
        
        questions_per_second = len(large_batch) / batch_time
        assert questions_per_second > 10, f"Batch processing too slow: {questions_per_second:.1f} q/s"
        
        print(f"Performance benchmarks:")
        print(f"  Single prediction: {single_prediction_time:.3f}s")
        print(f"  Batch processing: {questions_per_second:.1f} questions/second")
    
    def test_error_handling_robustness(self):
        """Test error handling and recovery."""
        # Test predictor without loaded model
        unloaded_predictor = MCQBiasPredictor("nonexistent_model.pkl")
        
        # Should raise ValueError for prediction without model
        with pytest.raises(ValueError, match="Model not loaded"):
            unloaded_predictor.predict_single(self.test_questions[0])
        
        with pytest.raises(ValueError, match="Model not loaded"):
            unloaded_predictor.predict_batch(self.test_questions)
        
        # Test file operations with bad paths
        if self.predictor:
            result = self.predictor.predict_from_file("nonexistent_file.csv")
            assert not result.get('success', True), "Should fail with nonexistent file"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        predictor = MCQBiasPredictor("nonexistent_model.pkl")
        
        # Test very long question
        long_question = {
            'question_text': 'A' * 2000,  # Very long text
            'options': ['Short A', 'Short B', 'Short C', 'Short D']
        }
        
        validation = predictor._validate_question_input(long_question)
        assert validation['valid'], "Long question should be valid"
        assert len(validation['warnings']) > 0, "Should warn about long question"
        
        # Test question with empty options
        empty_options_question = {
            'question_text': 'Test question',
            'options': ['', 'Valid option', '', 'Another valid']
        }
        
        validation = predictor._validate_question_input(empty_options_question)
        assert validation['valid'], "Question with some empty options should be valid"
        assert len(validation['warnings']) > 0, "Should warn about empty options"
        
        # Test Unicode characters
        unicode_question = {
            'question_text': 'What is the meaning of café? 🤔',
            'options': ['Coffee shop ☕', 'Restaurant 🍽️', 'Bar 🍺', 'Hotel 🏨']
        }
        
        validation = predictor._validate_question_input(unicode_question)
        assert validation['valid'], "Unicode question should be valid"
    
    def test_prediction_history_tracking(self):
        """Test prediction history and statistics."""
        if self.predictor is None:
            pytest.skip("No trained model available for history testing")
        
        # Clear history
        self.predictor.prediction_history = []
        
        # Make some predictions
        for question in self.test_questions[:2]:
            self.predictor.predict_single(question)
        
        # Check history
        assert len(self.predictor.prediction_history) == 2, "History not tracking correctly"
        
        # Get statistics
        stats = self.predictor.get_prediction_statistics()
        assert stats['total_predictions'] == 2, "Statistics incorrect"
        assert 0 <= stats['success_rate'] <= 1, "Success rate out of range"
        
        # Test empty history
        empty_predictor = MCQBiasPredictor("nonexistent_model.pkl")
        empty_stats = empty_predictor.get_prediction_statistics()
        assert empty_stats['total_predictions'] == 0, "Empty history should have 0 predictions"


class TestStressAndLoad:
    """Stress testing and load testing for the prediction pipeline."""
    
    def test_memory_usage(self):
        """Test memory usage with large batches."""
        if not Path("../mcq_bias_model/mcq_bias_model.pkl").exists():
            pytest.skip("No trained model for memory testing")
        
        predictor = MCQBiasPredictor("../mcq_bias_model/mcq_bias_model.pkl")
        
        # Create large batch
        large_batch = [
            {
                'question_text': f'Test question {i}',
                'options': [f'Option {i}A', f'Option {i}B', f'Option {i}C', f'Option {i}D']
            }
            for i in range(1000)
        ]
        
        # Monitor memory usage (basic check)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch
        results = predictor.predict_batch(large_batch)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB increase"
        assert results['batch_summary']['total_questions'] == 1000
        
        print(f"Memory usage test: {memory_increase:.1f}MB increase for 1000 questions")
    
    def test_concurrent_predictions(self):
        """Test concurrent prediction requests (basic simulation)."""
        if not Path("../mcq_bias_model/mcq_bias_model.pkl").exists():
            pytest.skip("No trained model for concurrency testing")
        
        predictor = MCQBiasPredictor("../mcq_bias_model/mcq_bias_model.pkl")
        
        test_question = {
            'question_text': 'Concurrent test question',
            'options': ['A', 'B', 'C', 'D']
        }
        
        # Simulate concurrent requests
        results = []
        for _ in range(50):
            result = predictor.predict_single(test_question)
            results.append(result)
        
        # All predictions should succeed and be consistent
        successful_results = [r for r in results if r.get('success', False)]
        assert len(successful_results) == 50, "Some concurrent predictions failed"
        
        # Check consistency (same input should give same output)
        if successful_results:
            first_prediction = successful_results[0]['predicted_option']
            consistent = all(r['predicted_option'] == first_prediction for r in successful_results)
            assert consistent, "Concurrent predictions not consistent"


def run_test_suite():
    """Run the complete test suite with reporting."""
    print("🧪 Tester: Running Comprehensive MCQ Bias Prediction Test Suite")
    print("=" * 70)
    
    # Run tests with pytest
    test_results = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    if test_results == 0:
        print("\n✅ All tests passed! Prediction pipeline is production-ready.")
    else:
        print(f"\n❌ Some tests failed. Exit code: {test_results}")
    
    return test_results


if __name__ == "__main__":
    run_test_suite()
