"""
MCQ Bias Prediction Interface - Tester Implementation
Production-ready inference pipeline with comprehensive validation and testing.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from pipeline import MCQBiasPipeline
from train import MCQBiasTrainer


class MCQBiasPredictor:
    """
    Tester implementation for MCQ bias prediction inference.
    
    Provides production-ready prediction interface with:
    - Batch prediction capabilities
    - Input validation and error handling
    - Performance monitoring
    - Comprehensive logging
    - Format validation
    """
    
    def __init__(self, model_path: str = "mcq_bias_model.pkl"):
        """
        Initialize the MCQ bias predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.trainer = None
        self.pipeline = MCQBiasPipeline(cache_features=True)
        self.is_loaded = False
        self.prediction_history = []
        
        print("🧪 Tester: Initializing MCQ Bias Predictor")
        
        # Load model if it exists
        if Path(model_path).exists():
            self.load_model()
        else:
            print(f"⚠️ Model file {model_path} not found. Train a model first.")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model with validation.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            True if loading successful, False otherwise
        """
        path = model_path or self.model_path
        
        try:
            print(f"🧪 Tester: Loading model from {path}...")
            
            self.trainer = MCQBiasTrainer()
            self.trainer.load_model(path)
            self.is_loaded = True
            
            # Validate model integrity
            if not self.trainer.is_trained:
                raise ValueError("Loaded model is not properly trained")
            
            if not hasattr(self.trainer, 'feature_names') or not self.trainer.feature_names:
                raise ValueError("Model missing feature names")
            
            print(f"✅ Model loaded successfully")
            print(f"   Model type: {self.trainer.model_type}")
            print(f"   Features: {len(self.trainer.feature_names)}")
            print(f"   Trained: {self.trainer.training_history.get('trained_at', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def predict_single(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict correct option for a single MCQ with comprehensive validation.
        
        Args:
            question_data: Dict with keys: question_text, options, question_number (optional)
            
        Returns:
            Prediction results with validation metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input format
        validation_result = self._validate_question_input(question_data)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'validation_details': validation_result
            }
        
        try:
            # Prepare question for pipeline
            standardized_question = self._standardize_question_input(question_data)
            
            # Create temporary DataFrame for pipeline
            df = pd.DataFrame([standardized_question])
            
            # Extract features
            features_df = self.pipeline.feature_extractor.extract_features_batch(df)
            
            # Get feature matrix
            feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
            X = features_df[feature_cols].values
            
            # Make prediction
            predicted_options, prediction_probs = self.trainer.predict(X)
            
            # Format results
            prediction_result = {
                'success': True,
                'predicted_option': int(predicted_options[0]),
                'predicted_option_letter': chr(ord('A') + predicted_options[0]),
                'predicted_answer': standardized_question['options'][predicted_options[0]],
                'confidence_scores': {
                    'A': float(prediction_probs[0][0]),
                    'B': float(prediction_probs[0][1]),
                    'C': float(prediction_probs[0][2]),
                    'D': float(prediction_probs[0][3])
                },
                'max_confidence': float(np.max(prediction_probs[0])),
                'confidence_margin': float(np.sort(prediction_probs[0])[-1] - np.sort(prediction_probs[0])[-2]),
                'prediction_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': self.trainer.model_type,
                    'features_extracted': len(feature_cols),
                    'input_validation': validation_result
                }
            }
            
            # Store prediction in history
            self.prediction_history.append({
                'input': question_data,
                'output': prediction_result,
                'timestamp': datetime.now().isoformat()
            })
            
            return prediction_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f"Prediction failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
            
            self.prediction_history.append({
                'input': question_data,
                'output': error_result,
                'timestamp': datetime.now().isoformat()
            })
            
            return error_result
    
    def predict_batch(self, questions: List[Dict[str, Any]], 
                     validate_each: bool = True) -> Dict[str, Any]:
        """
        Predict correct options for multiple MCQs with batch processing.
        
        Args:
            questions: List of question dictionaries
            validate_each: Whether to validate each question individually
            
        Returns:
            Batch prediction results with statistics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"🧪 Tester: Running batch prediction on {len(questions)} questions...")
        
        start_time = datetime.now()
        results = []
        successful_predictions = 0
        failed_predictions = 0
        validation_failures = 0
        
        for i, question in enumerate(questions):
            try:
                # Individual validation if requested
                if validate_each:
                    validation_result = self._validate_question_input(question)
                    if not validation_result['valid']:
                        results.append({
                            'question_index': i,
                            'success': False,
                            'error': 'Validation failed',
                            'validation_details': validation_result
                        })
                        validation_failures += 1
                        continue
                
                # Make prediction
                prediction = self.predict_single(question)
                prediction['question_index'] = i
                results.append(prediction)
                
                if prediction['success']:
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                
                # Progress reporting for large batches
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(questions)} questions...")
                    
            except Exception as e:
                results.append({
                    'question_index': i,
                    'success': False,
                    'error': f"Batch prediction error: {str(e)}"
                })
                failed_predictions += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate batch statistics
        batch_results = {
            'batch_summary': {
                'total_questions': len(questions),
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'validation_failures': validation_failures,
                'success_rate': successful_predictions / len(questions) if questions else 0,
                'processing_time_seconds': processing_time,
                'questions_per_second': len(questions) / processing_time if processing_time > 0 else 0
            },
            'predictions': results,
            'batch_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.trainer.model_type,
                'validation_enabled': validate_each
            }
        }
        
        print(f"✅ Batch prediction completed:")
        print(f"   Success rate: {batch_results['batch_summary']['success_rate']:.1%}")
        print(f"   Processing speed: {batch_results['batch_summary']['questions_per_second']:.1f} questions/second")
        
        return batch_results
    
    def predict_from_file(self, file_path: str, dataset_type: str = None,
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict correct options for questions from a CSV file.
        
        Args:
            file_path: Path to CSV file with questions
            dataset_type: Type of dataset ('indiabix', 'medqs', etc.)
            output_path: Optional path to save predictions
            
        Returns:
            File prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"🧪 Tester: Predicting from file {file_path}...")
        
        try:
            # Load data using pipeline
            data = self.pipeline.data_loader.load_dataset(file_path, dataset_type)
            
            # Convert to question format
            questions = []
            for _, row in data.iterrows():
                questions.append({
                    'question_text': row['question_text'],
                    'options': row['options'],
                    'question_number': row['question_number'],
                    'question_id': row['question_id'],
                    'exam_id': row['exam_id'],
                    'correct_answer': row.get('correct_answer')  # May not be available for prediction
                })
            
            # Run batch prediction
            batch_results = self.predict_batch(questions, validate_each=False)
            
            # Add file-specific metadata
            file_results = {
                'file_info': {
                    'file_path': file_path,
                    'dataset_type': dataset_type or 'auto-detected',
                    'questions_loaded': len(questions)
                },
                **batch_results
            }
            
            # Save results if output path provided
            if output_path:
                self._save_predictions_to_file(file_results, output_path)
            
            return file_results
            
        except Exception as e:
            print(f"❌ Error predicting from file: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def validate_accuracy(self, test_data: Union[str, pd.DataFrame],
                         dataset_type: str = None) -> Dict[str, Any]:
        """
        Validate prediction accuracy against known correct answers.
        
        Args:
            test_data: Path to test file or DataFrame with correct answers
            dataset_type: Type of dataset if test_data is a file path
            
        Returns:
            Accuracy validation results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("🧪 Tester: Validating prediction accuracy...")
        
        try:
            # Load test data
            if isinstance(test_data, str):
                data = self.pipeline.data_loader.load_dataset(test_data, dataset_type)
            else:
                data = test_data
            
            # Prepare questions and true answers
            questions = []
            true_answers = []
            
            for _, row in data.iterrows():
                questions.append({
                    'question_text': row['question_text'],
                    'options': row['options'],
                    'question_number': row['question_number']
                })
                
                # Find true answer index
                correct_answer = str(row['correct_answer'])
                true_answer_idx = 0
                for i, option in enumerate(row['options']):
                    if str(option) == correct_answer:
                        true_answer_idx = i
                        break
                true_answers.append(true_answer_idx)
            
            # Get predictions
            batch_results = self.predict_batch(questions, validate_each=False)
            
            # Calculate accuracy metrics
            predicted_answers = []
            successful_predictions = 0
            
            for result in batch_results['predictions']:
                if result['success']:
                    predicted_answers.append(result['predicted_option'])
                    successful_predictions += 1
                else:
                    predicted_answers.append(-1)  # Invalid prediction
            
            # Accuracy calculation (only for successful predictions)
            valid_indices = [i for i, pred in enumerate(predicted_answers) if pred != -1]
            
            if valid_indices:
                valid_true = [true_answers[i] for i in valid_indices]
                valid_pred = [predicted_answers[i] for i in valid_indices]
                
                accuracy = sum(t == p for t, p in zip(valid_true, valid_pred)) / len(valid_true)
                
                # Top-2 accuracy
                top2_correct = 0
                for result_idx, result in enumerate(batch_results['predictions']):
                    if result['success'] and result_idx in valid_indices:
                        scores = list(result['confidence_scores'].values())
                        top2_options = sorted(range(4), key=lambda x: scores[x], reverse=True)[:2]
                        if true_answers[result_idx] in top2_options:
                            top2_correct += 1
                
                top2_accuracy = top2_correct / len(valid_true)
            else:
                accuracy = 0.0
                top2_accuracy = 0.0
            
            validation_results = {
                'accuracy_metrics': {
                    'overall_accuracy': accuracy,
                    'top2_accuracy': top2_accuracy,
                    'random_baseline': 0.25,
                    'improvement_over_random': (accuracy - 0.25) / 0.25 * 100 if accuracy > 0.25 else 0
                },
                'prediction_stats': {
                    'total_questions': len(questions),
                    'successful_predictions': successful_predictions,
                    'prediction_success_rate': successful_predictions / len(questions),
                    'valid_for_accuracy': len(valid_indices)
                },
                'detailed_results': batch_results,
                'validation_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'test_data_source': str(test_data) if isinstance(test_data, str) else 'DataFrame'
                }
            }
            
            print(f"✅ Accuracy validation completed:")
            print(f"   Overall accuracy: {accuracy:.1%}")
            print(f"   Top-2 accuracy: {top2_accuracy:.1%}")
            print(f"   Improvement over random: {validation_results['accuracy_metrics']['improvement_over_random']:.1f}%")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Accuracy validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_question_input(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input question format and content."""
        validation_result = {
            'valid': True,
            'error': None,
            'warnings': [],
            'checks_performed': []
        }
        
        # Check required fields
        required_fields = ['question_text', 'options']
        for field in required_fields:
            validation_result['checks_performed'].append(f'check_{field}_present')
            if field not in question_data:
                validation_result['valid'] = False
                validation_result['error'] = f"Missing required field: {field}"
                return validation_result
        
        # Validate question text
        validation_result['checks_performed'].append('check_question_text_validity')
        if not isinstance(question_data['question_text'], str) or not question_data['question_text'].strip():
            validation_result['valid'] = False
            validation_result['error'] = "Question text must be a non-empty string"
            return validation_result
        
        # Validate options
        validation_result['checks_performed'].append('check_options_format')
        options = question_data['options']
        if not isinstance(options, list):
            validation_result['valid'] = False
            validation_result['error'] = "Options must be a list"
            return validation_result
        
        validation_result['checks_performed'].append('check_options_count')
        if len(options) < 2:
            validation_result['valid'] = False
            validation_result['error'] = "Must have at least 2 options"
            return validation_result
        
        if len(options) != 4:
            validation_result['warnings'].append(f"Expected 4 options, got {len(options)}")
        
        # Validate option content
        validation_result['checks_performed'].append('check_options_content')
        for i, option in enumerate(options):
            if not isinstance(option, str) or not option.strip():
                validation_result['warnings'].append(f"Option {i} is empty or not a string")
        
        # Check for question length
        validation_result['checks_performed'].append('check_question_length')
        if len(question_data['question_text']) > 1000:
            validation_result['warnings'].append("Question text is very long (>1000 characters)")
        
        return validation_result
    
    def _standardize_question_input(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize question input to pipeline format."""
        return {
            'question_id': question_data.get('question_id', 'pred_question'),
            'exam_id': question_data.get('exam_id', 'prediction'),
            'question_number': question_data.get('question_number', 1),
            'question_text': question_data['question_text'],
            'options': question_data['options'][:4],  # Limit to 4 options
            'correct_answer': question_data.get('correct_answer', question_data['options'][0])
        }
    
    def _save_predictions_to_file(self, results: Dict[str, Any], output_path: str):
        """Save prediction results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Predictions saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to save predictions: {e}")
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction history."""
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        successful = sum(1 for pred in self.prediction_history if pred['output'].get('success', False))
        
        return {
            'total_predictions': len(self.prediction_history),
            'successful_predictions': successful,
            'success_rate': successful / len(self.prediction_history),
            'average_confidence': np.mean([
                pred['output'].get('max_confidence', 0)
                for pred in self.prediction_history
                if pred['output'].get('success', False)
            ]) if successful > 0 else 0,
            'first_prediction': self.prediction_history[0]['timestamp'],
            'last_prediction': self.prediction_history[-1]['timestamp']
        }


def demo_prediction():
    """Demonstrate the prediction pipeline capabilities."""
    print("🧪 Tester: Demonstrating MCQ Bias Prediction Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MCQBiasPredictor()
    
    if not predictor.is_loaded:
        print("❌ No trained model found. Please run train.py first.")
        return None
    
    # Test single prediction
    print("\n🔮 Testing Single Prediction...")
    test_question = {
        'question_text': 'What is the capital of France?',
        'options': ['Paris', 'London', 'Berlin', 'Madrid'],
        'question_number': 1
    }
    
    result = predictor.predict_single(test_question)
    
    if result['success']:
        print(f"   ✅ Predicted answer: {result['predicted_option_letter']} - {result['predicted_answer']}")
        print(f"   🎯 Confidence: {result['max_confidence']:.3f}")
        print(f"   📊 All scores: {result['confidence_scores']}")
    else:
        print(f"   ❌ Prediction failed: {result['error']}")
    
    # Test batch prediction
    print(f"\n📚 Testing Batch Prediction...")
    test_questions = [
        {
            'question_text': 'Which is the largest planet?',
            'options': ['Earth', 'Jupiter', 'Saturn', 'Mars']
        },
        {
            'question_text': 'What is 2 + 2?',
            'options': ['3', '4', '5', '6']
        },
        {
            'question_text': 'Who wrote Romeo and Juliet?',
            'options': ['Shakespeare', 'Dickens', 'Austen', 'Twain']
        }
    ]
    
    batch_results = predictor.predict_batch(test_questions)
    
    print(f"   ✅ Batch completed: {batch_results['batch_summary']['success_rate']:.1%} success rate")
    print(f"   ⚡ Speed: {batch_results['batch_summary']['questions_per_second']:.1f} questions/second")
    
    # Show individual results
    for i, pred in enumerate(batch_results['predictions']):
        if pred['success']:
            print(f"   Question {i+1}: {pred['predicted_option_letter']} (confidence: {pred['max_confidence']:.3f})")
    
    # Test file prediction
    print(f"\n📁 Testing File Prediction...")
    try:
        file_results = predictor.predict_from_file(
            '../data/mental_health/mhqa.csv',
            dataset_type='mental_health'
        )
        
        if 'batch_summary' in file_results:
            print(f"   ✅ File processed: {file_results['batch_summary']['successful_predictions']} predictions")
            print(f"   📊 Success rate: {file_results['batch_summary']['success_rate']:.1%}")
        
    except Exception as e:
        print(f"   ⚠️ File prediction test skipped: {e}")
    
    # Get prediction statistics
    stats = predictor.get_prediction_statistics()
    print(f"\n📈 Prediction Statistics:")
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    
    return predictor


if __name__ == "__main__":
    demo_prediction()
