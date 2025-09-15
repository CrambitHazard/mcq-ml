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
    
    def create_per_option_dataset(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform question-level data into per-option binary classification format.
        
        Each question with 4 options becomes 4 training examples:
        - Option A features + is_correct=1 if y==0 else 0
        - Option B features + is_correct=1 if y==1 else 0
        - Option C features + is_correct=1 if y==2 else 0
        - Option D features + is_correct=1 if y==3 else 0
        
        Args:
            X: Feature matrix (n_questions, n_features)
            y: Target vector (n_questions,) with values 0-3
            feature_names: List of feature names
            
        Returns:
            (X_options, y_binary) where X_options has shape (n_questions*4, n_features)
        """
        print("🤖 AI/ML Engineer: Creating per-option binary classification dataset...")
        
        n_questions, n_features = X.shape
        n_options = 4  # A, B, C, D
        
        # Create expanded feature matrix
        X_options = np.repeat(X, n_options, axis=0)  # (n_questions*4, n_features)
        
        # Create binary target vector
        y_binary = np.zeros(n_questions * n_options)
        
        for i in range(n_questions):
            correct_option = y[i]
            # Set the correct option to 1, others remain 0
            option_start_idx = i * n_options
            if 0 <= correct_option < n_options:
                y_binary[option_start_idx + correct_option] = 1
        
        # Add option index as a feature (helps model distinguish option positions)
        option_indices = np.tile(np.arange(n_options), n_questions).reshape(-1, 1)
        X_options = np.column_stack([X_options, option_indices])
        
        # Update feature names
        extended_feature_names = feature_names + ['option_index']
        
        print(f"   ✅ Created per-option dataset: {X_options.shape[0]:,} examples, {X_options.shape[1]} features")
        print(f"   📊 Binary target distribution: {np.bincount(y_binary.astype(int))}")
        
        return X_options, y_binary, extended_feature_names
    
    def _create_model(self) -> Any:
        """Create the specified model with optimized hyperparameters."""
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                n_estimators=100  # Fast training for 2-hour constraint
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
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
                solver='lbfgs'
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
              validation_split: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the per-option bias classifier.
        
        Args:
            X: Feature matrix (n_questions, n_features)
            y: Target vector (n_questions,) with values 0-3
            feature_names: List of feature names
            validation_split: Proportion for validation set
            verbose: Whether to print training progress
            
        Returns:
            Training results dictionary
        """
        print(f"🤖 AI/ML Engineer: Training {self.model_type} on MCQ bias features...")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Create per-option dataset
        X_options, y_binary, extended_feature_names = self.create_per_option_dataset(
            X, y, feature_names
        )
        
        # Create validation split
        n_questions = len(X)
        n_val = int(n_questions * validation_split)
        
        # Split at question level to prevent leakage
        val_question_indices = np.random.RandomState(self.random_state).choice(
            n_questions, size=n_val, replace=False
        )
        train_question_indices = np.setdiff1d(np.arange(n_questions), val_question_indices)
        
        # Convert to option-level indices
        train_option_indices = np.concatenate([
            np.arange(i*4, (i+1)*4) for i in train_question_indices
        ])
        val_option_indices = np.concatenate([
            np.arange(i*4, (i+1)*4) for i in val_question_indices
        ])
        
        X_train, X_val = X_options[train_option_indices], X_options[val_option_indices]
        y_train, y_val = y_binary[train_option_indices], y_binary[val_option_indices]
        
        if verbose:
            print(f"   📊 Training set: {len(X_train):,} option examples ({len(train_question_indices)} questions)")
            print(f"   📊 Validation set: {len(X_val):,} option examples ({len(val_question_indices)} questions)")
        
        # Train model
        start_time = datetime.now()
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on validation set
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]  # Probability of being correct
        val_pred_binary = (val_pred_proba > 0.5).astype(int)
        
        # Convert back to question-level predictions for evaluation
        question_level_results = self._evaluate_question_level(
            X_val, y_val, val_question_indices, X, y
        )
        
        # Store training results
        self.training_history = {
            'model_type': self.model_type,
            'training_time_seconds': training_time,
            'n_training_questions': len(train_question_indices),
            'n_validation_questions': len(val_question_indices),
            'n_features': X_train.shape[1],
            'feature_names': extended_feature_names,
            'validation_results': question_level_results,
            'trained_at': datetime.now().isoformat()
        }
        
        self.is_trained = True
        
        if verbose:
            print(f"   ✅ Training completed in {training_time:.2f}s")
            print(f"   🎯 Validation accuracy: {question_level_results['accuracy']:.3f}")
            print(f"   📊 Random baseline: 0.250 (25%)")
            print(f"   📈 Improvement: {(question_level_results['accuracy'] - 0.25) / 0.25 * 100:.1f}% over random")
        
        return self.training_history
    
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
        Predict correct options for questions using trained model.
        
        Args:
            X: Feature matrix (n_questions, n_features)
            
        Returns:
            (predicted_options, prediction_probabilities)
            predicted_options: Array of shape (n_questions,) with values 0-3
            prediction_probabilities: Array of shape (n_questions, 4) with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create per-option dataset (without using stored feature_names that might have different length)
        n_questions, n_features = X.shape
        n_options = 4
        
        # Create expanded feature matrix
        X_options = np.repeat(X, n_options, axis=0)  # (n_questions*4, n_features)
        
        # Add option index as a feature
        option_indices = np.tile(np.arange(n_options), n_questions).reshape(-1, 1)
        X_options = np.column_stack([X_options, option_indices])
        
        # Get prediction probabilities
        pred_proba = self.model.predict_proba(X_options)[:, 1]
        
        # Reshape to question format
        pred_proba_reshaped = pred_proba.reshape(n_questions, 4)
        
        # Use argmax to select predicted answers
        predicted_options = np.argmax(pred_proba_reshaped, axis=1)
        
        return predicted_options, pred_proba_reshaped
    
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
        
        # Create per-option dataset
        X_options, y_binary, extended_feature_names = self.create_per_option_dataset(
            X, y, feature_names
        )
        
        # Expand groups to match option-level data
        groups_expanded = np.repeat(groups, 4)
        
        # Group K-Fold to prevent exam leakage
        gkf = GroupKFold(n_splits=cv_folds)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_options, y_binary, groups_expanded)):
            print(f"   🔄 Training fold {fold + 1}/{cv_folds}...")
            
            # Split data
            X_train_fold, X_val_fold = X_options[train_idx], X_options[val_idx]
            y_train_fold, y_val_fold = y_binary[train_idx], y_binary[val_idx]
            
            # Train model for this fold
            model_fold = self._create_model()
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Evaluate at question level
            val_question_indices = np.unique(val_idx // 4)  # Convert back to question indices
            question_results = self._evaluate_question_level_cv(
                model_fold, X_val_fold, y_val_fold, val_question_indices, X, y
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
        
        # Get extended feature names (including option_index)
        extended_names = self.feature_names + ['option_index']
        
        # Create importance dictionary
        importance_dict = dict(zip(extended_names, importance_scores))
        
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
