"""
MCQ Bias Model Training Pipeline - AI/ML Engineer Implementation
Trains per-option classifiers using LightGBM/XGBoost to predict correct answers based on bias features.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from random_baseline import RandomBaselineValidator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

from pipeline import MCQBiasPipeline


class MCQBiasTrainer:
    """
    AI/ML Engineer implementation for training MCQ bias prediction models.
    
    Uses per-option binary classification approach:
    1. Transform each question into 4 option examples (A, B, C, D)
    2. Train binary classifier to predict if each option is correct
    3. At inference, use argmax across 4 options to select answer
    """
    
    def __init__(self, model_type: str = 'auto', random_state: int = 42):
        """
        Initialize the MCQ bias trainer.
        
        Args:
            model_type: 'lightgbm', 'xgboost', 'random_forest', 'logistic', or 'auto'
            random_state: Random seed for reproducibility
        """
        self.model_type = self._select_model_type(model_type)
        self.random_state = random_state
        
        # Model and training state
        self.model = None
        self.feature_names = None
        self.training_history = {}
        self.is_trained = False
        
        print(f"🤖 AI/ML Engineer: Initializing MCQ Bias Trainer with {self.model_type}")
    
    def _select_model_type(self, requested_type: str) -> str:
        """Select the best available model type."""
        if requested_type == 'auto':
            if LIGHTGBM_AVAILABLE:
                return 'lightgbm'
            elif XGBOOST_AVAILABLE:
                return 'xgboost'
            else:
                return 'random_forest'
        
        # Validate requested type is available
        if requested_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            print("⚠️ LightGBM not available, falling back to XGBoost")
            return 'xgboost' if XGBOOST_AVAILABLE else 'random_forest'
        
        if requested_type == 'xgboost' and not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost not available, falling back to Random Forest")
            return 'random_forest'
        
        return requested_type
    
    def prepare_multiclass_dataset(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for proper multi-class classification.
        
        CRITICAL FIX: Instead of creating binary per-option classification,
        we use the original feature matrix directly with 4-class targets (0, 1, 2, 3).
        
        Args:
            X: Feature matrix (n_questions, n_features)
            y: Target vector (n_questions,) with values 0-3 (correct option indices)
            feature_names: List of feature names
            
        Returns:
            (X, y, feature_names) - Ready for multi-class classification
        """
        print("🤖 AI/ML Engineer: Preparing data for proper multi-class classification...")
        
        # Validate target values
        unique_targets = np.unique(y)
        if not all(target in [0, 1, 2, 3] for target in unique_targets):
            raise ValueError(f"Invalid target values: {unique_targets}. Must be in [0, 1, 2, 3]")
        
        n_questions, n_features = X.shape
        
        print(f"   ✅ Multi-class dataset ready: {n_questions:,} questions, {n_features} features")
        print(f"   📊 Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, feature_names
    
    def _create_model(self) -> Any:
        """Create the specified model with optimized hyperparameters."""
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='multiclass',  # CRITICAL FIX: Use multiclass instead of binary
                num_class=4,            # CRITICAL FIX: Specify 4 classes (A, B, C, D)
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                n_estimators=100,  # Fast training for 2-hour constraint
                metric='multi_logloss'  # CRITICAL FIX: Use multi-class metric
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                objective='multi:softprob',  # CRITICAL FIX: Use multi-class with softmax
                num_class=4,                 # CRITICAL FIX: Specify 4 classes
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='mlogloss',      # CRITICAL FIX: Use multi-class metric
                verbosity=0
            )
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs',
                multi_class='auto'  # CRITICAL FIX: Enable multi-class
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
              validation_split: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the multi-class MCQ bias classifier.
        
        CRITICAL FIX: Now uses proper 4-class classification instead of binary per-option.
        
        Args:
            X: Feature matrix (n_questions, n_features)
            y: Target vector (n_questions,) with values 0-3 (correct option indices)
            feature_names: List of feature names
            validation_split: Proportion for validation set
            verbose: Whether to print training progress
            
        Returns:
            Training results dictionary
        """
        print(f"🤖 AI/ML Engineer: Training {self.model_type} with proper multi-class classification...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Prepare multi-class dataset (no more binary per-option nonsense!)
        X_clean, y_clean, clean_feature_names = self.prepare_multiclass_dataset(
            X, y, feature_names
        )
        
        # Create validation split at question level
        n_questions = len(X_clean)
        n_val = int(n_questions * validation_split)
        
        # Split at question level to prevent leakage
        val_indices = np.random.RandomState(self.random_state).choice(
            n_questions, size=n_val, replace=False
        )
        train_indices = np.setdiff1d(np.arange(n_questions), val_indices)
        
        X_train, X_val = X_clean[train_indices], X_clean[val_indices]
        y_train, y_val = y_clean[train_indices], y_clean[val_indices]
        
        if verbose:
            print(f"   📊 Training set: {len(X_train):,} questions")
            print(f"   📊 Validation set: {len(X_val):,} questions")
            print(f"   🎯 Target classes: {sorted(np.unique(y_train))}")
        
        # Train model
        start_time = datetime.now()
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on validation set with proper multi-class predictions
        val_pred_proba = self.model.predict_proba(X_val)  # Shape: (n_questions, 4)
        val_predictions = np.argmax(val_pred_proba, axis=1)  # Argmax on proper probabilities
        
        # Validate probability distributions sum to 1.0
        prob_sums = np.sum(val_pred_proba, axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-6):
            print(f"⚠️ Warning: Probability sums not equal to 1.0: {prob_sums[:5]}")
        else:
            print(f"   ✅ Probability distributions validated: sum = 1.0 ± {np.std(prob_sums):.6f}")
        
        # Calculate metrics
        question_level_results = self._evaluate_multiclass_performance(
            y_val, val_predictions, val_pred_proba
        )
        
        # Store training results
        self.training_history = {
            'model_type': self.model_type,
            'training_time_seconds': training_time,
            'n_train_questions': len(X_train),
            'n_val_questions': len(X_val),
            'n_features': X_clean.shape[1],
            'feature_names': clean_feature_names,
            'validation_results': question_level_results
        }
        
        self.is_trained = True
        
        if verbose:
            accuracy = question_level_results['accuracy']
            top2_acc = question_level_results['top2_accuracy']
            print(f"   ✅ Training completed in {training_time:.2f}s")
            print(f"   🎯 Validation accuracy: {accuracy:.1%}")
            print(f"   🎯 Top-2 accuracy: {top2_acc:.1%}")
            print(f"   🎯 Improvement over random: {((accuracy - 0.25) / 0.25 * 100):.1f}%")
        
        return self.training_history
    
    def _evaluate_multiclass_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate multi-class performance with proper random baseline comparison.
        
        CRITICAL FIX: Now includes proper random baseline validation and statistical testing.
        
        Args:
            y_true: True labels (0, 1, 2, 3)
            y_pred: Predicted labels (0, 1, 2, 3)
            y_pred_proba: Prediction probabilities (n_questions, 4)
            
        Returns:
            Performance metrics dictionary with random baseline comparison
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Top-2 accuracy
        top2_indices = np.argsort(y_pred_proba, axis=1)[:, -2:]  # Top 2 indices
        top2_accuracy = np.mean([true_label in top2 for true_label, top2 in zip(y_true, top2_indices)])
        
        # Class-wise metrics
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except:
            class_report = {}
        
        # CRITICAL FIX: Proper random baseline comparison with statistical testing
        validator = RandomBaselineValidator(n_classes=4)
        
        # Get actual class distribution for realistic random baseline
        class_counts = np.bincount(y_true, minlength=4)
        class_weights = class_counts / len(y_true) if len(y_true) > 0 else [0.25, 0.25, 0.25, 0.25]
        
        # Theoretical random baseline (uniform)
        theoretical_random = 0.25
        
        # Realistic random baseline (considering class imbalance)
        realistic_random = np.sum(class_weights ** 2)  # Expected accuracy with class imbalance
        
        # Statistical comparison to random
        comparison_results = validator.compare_to_random(y_pred, y_true, "Model")
        
        # Calculate improvements
        improvement_over_theoretical = ((accuracy - theoretical_random) / theoretical_random) * 100
        improvement_over_realistic = ((accuracy - realistic_random) / realistic_random) * 100
        
        return {
            'accuracy': accuracy,
            'top2_accuracy': top2_accuracy,
            'theoretical_random_baseline': theoretical_random,
            'realistic_random_baseline': realistic_random,
            'class_distribution': class_counts.tolist(),
            'class_weights': class_weights.tolist(),
            'improvement_over_theoretical': improvement_over_theoretical,
            'improvement_over_realistic': improvement_over_realistic,
            'statistical_comparison': {
                'p_value': comparison_results['p_value'],
                'is_significant': comparison_results['is_significant'],
                'mcnemar_statistic': comparison_results['mcnemar_statistic'],
                'confidence_interval_95': comparison_results['model_ci_95']
            },
            'n_questions': len(y_true),
            'predictions': y_pred.tolist(),
            'true_answers': y_true.tolist(),
            'class_report': class_report,
            'mean_confidence': np.mean(np.max(y_pred_proba, axis=1)),
            'prediction_entropy': np.mean(-np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10), axis=1))
        }
    
    def _evaluate_question_level(self, X_val_options: np.ndarray, y_val_binary: np.ndarray,
                                val_question_indices: np.ndarray, X_original: np.ndarray, 
                                y_original: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance at question level using argmax strategy."""
        
        # Get predictions for validation options
        val_pred_proba = self.model.predict_proba(X_val_options)[:, 1]
        
        # Reshape predictions back to question format (n_questions, 4)
        n_val_questions = len(val_question_indices)
        val_pred_reshaped = val_pred_proba.reshape(n_val_questions, 4)
        
        # Use argmax to select predicted answer for each question
        predicted_answers = np.argmax(val_pred_reshaped, axis=1)
        true_answers = y_original[val_question_indices]
        
        # Calculate metrics
        accuracy = accuracy_score(true_answers, predicted_answers)
        
        # Top-2 accuracy
        top2_indices = np.argsort(val_pred_reshaped, axis=1)[:, -2:]
        top2_accuracy = np.mean([true_ans in top2 for true_ans, top2 in zip(true_answers, top2_indices)])
        
        return {
            'accuracy': accuracy,
            'top2_accuracy': top2_accuracy,
            'n_questions': n_val_questions,
            'predictions': predicted_answers.tolist(),
            'true_answers': true_answers.tolist()
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict correct options for questions using trained multi-class model.
        
        CRITICAL FIX: Now uses proper multi-class prediction instead of binary per-option.
        
        Args:
            X: Feature matrix (n_questions, n_features)
            
        Returns:
            (predicted_options, prediction_probabilities)
            predicted_options: Array of shape (n_questions,) with values 0-3
            prediction_probabilities: Array of shape (n_questions, 4) with probabilities that sum to 1.0
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Direct multi-class prediction (no more per-option nonsense!)
        pred_proba = self.model.predict_proba(X)  # Shape: (n_questions, 4)
        predicted_options = np.argmax(pred_proba, axis=1)  # Argmax on proper probabilities
        
        # Validate probability distributions
        prob_sums = np.sum(pred_proba, axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-6):
            print(f"⚠️ Warning: Prediction probabilities don't sum to 1.0: {prob_sums[:3]}")
        
        return predicted_options, pred_proba
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                      groups: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform grouped cross-validation to prevent exam-level data leakage.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            groups: Group identifiers (exam_ids) for GroupKFold
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        print(f"🤖 AI/ML Engineer: Running {cv_folds}-fold grouped cross-validation...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Prepare multi-class dataset
        X_clean, y_clean, clean_feature_names = self.prepare_multiclass_dataset(
            X, y, feature_names
        )
        
        # Use groups as-is for question-level cross-validation
        # Group K-Fold to prevent exam leakage
        gkf = GroupKFold(n_splits=cv_folds)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_clean, y_clean, groups)):
            print(f"   🔄 Training fold {fold + 1}/{cv_folds}...")
            
            # Split data at question level
            X_train_fold, X_val_fold = X_clean[train_idx], X_clean[val_idx]
            y_train_fold, y_val_fold = y_clean[train_idx], y_clean[val_idx]
            
            # Train model for this fold
            model_fold = self._create_model()
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Evaluate with multi-class predictions
            val_pred_proba = model_fold.predict_proba(X_val_fold)
            val_predictions = np.argmax(val_pred_proba, axis=1)
            
            question_results = self._evaluate_multiclass_performance(
                y_val_fold, val_predictions, val_pred_proba
            )
            
            cv_scores.append(question_results['accuracy'])
            fold_results.append(question_results)
            
            print(f"      ✅ Fold {fold + 1} accuracy: {question_results['accuracy']:.3f}")
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'fold_results': fold_results,
            'cv_folds': cv_folds,
            'model_type': self.model_type
        }
        
        print(f"   🎯 Cross-validation completed:")
        print(f"      Mean accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        print(f"      Best fold: {max(cv_scores):.3f}")
        print(f"      Worst fold: {min(cv_scores):.3f}")
        
        return cv_results
    
    def _evaluate_question_level_cv(self, model, X_val_options: np.ndarray, y_val_binary: np.ndarray,
                                   val_question_indices: np.ndarray, X_original: np.ndarray, 
                                   y_original: np.ndarray) -> Dict[str, float]:
        """Helper for question-level evaluation during cross-validation."""
        
        val_pred_proba = model.predict_proba(X_val_options)[:, 1]
        n_val_questions = len(val_question_indices)
        val_pred_reshaped = val_pred_proba.reshape(n_val_questions, 4)
        
        predicted_answers = np.argmax(val_pred_reshaped, axis=1)
        true_answers = y_original[val_question_indices]
        
        accuracy = accuracy_score(true_answers, predicted_answers)
        
        return {'accuracy': accuracy, 'n_questions': n_val_questions}
    
    def save_model(self, filepath: str):
        """Save trained model and metadata."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_type': self.model_type,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"🤖 AI/ML Engineer: Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and metadata."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.model_type = model_data['model_type']
        self.random_state = model_data.get('random_state', 42)
        self.is_trained = True
        
        print(f"🤖 AI/ML Engineer: Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_[0])
        else:
            print("⚠️ Model doesn't support feature importance")
            return {}
        
        # Use stored feature names from multi-class training
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])


def demo_training():
    """Demonstrate the complete training pipeline."""
    print("🤖 AI/ML Engineer: Demonstrating MCQ Bias Model Training")
    print("=" * 60)
    
    # Initialize pipeline and trainer
    pipeline = MCQBiasPipeline(cache_features=True)
    trainer = MCQBiasTrainer(model_type='auto')
    
    # Dataset configuration
    dataset_configs = [
        {'path': '../data/indiabix_ca/bix_ca.csv', 'type': 'indiabix'},
        {'path': '../data/medqs/train.csv', 'type': 'medqs'},
        {'path': '../data/mental_health/mhqa.csv', 'type': 'mental_health'},
        {'path': '../data/qna/Train.csv', 'type': 'qna'}
    ]
    
    try:
        # Create ML dataset
        print("\n📊 Creating ML dataset...")
        ml_dataset = pipeline.create_ml_dataset(
            dataset_configs, 
            sample_size=2000,  # Reasonable size for demo
            test_size=0.2,
            validation_size=0.1
        )
        
        X_train = ml_dataset['X_train']
        y_train = ml_dataset['y_train']
        feature_names = ml_dataset['feature_names']
        
        print(f"   ✅ Training data ready: {X_train.shape[0]} questions, {X_train.shape[1]} features")
        
        # Train model
        print(f"\n🚀 Training {trainer.model_type} model...")
        training_results = trainer.train(X_train, y_train, feature_names, verbose=True)
        
        # Test prediction
        print(f"\n🔮 Testing prediction on validation set...")
        X_test = ml_dataset['X_test']
        y_test = ml_dataset['y_test']
        
        predicted_options, prediction_probs = trainer.predict(X_test)
        test_accuracy = accuracy_score(y_test, predicted_options)
        
        print(f"   🎯 Test accuracy: {test_accuracy:.3f}")
        print(f"   📊 Random baseline: 0.250")
        print(f"   📈 Improvement: {(test_accuracy - 0.25) / 0.25 * 100:.1f}% over random")
        
        # Show feature importance
        print(f"\n🔍 Top 10 Most Important Features:")
        importance = trainer.get_feature_importance(top_n=10)
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"   {i:2d}. {feature}: {score:.4f}")
        
        # Save model
        model_path = "mcq_bias_model.pkl"
        trainer.save_model(model_path)
        
        return trainer, ml_dataset, training_results
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        return None, None, None


if __name__ == "__main__":
    demo_training()
