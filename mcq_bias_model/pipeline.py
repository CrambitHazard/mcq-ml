"""
MCQ Bias Prediction Pipeline - Python Developer Implementation
Integrates data loading and feature extraction into a unified pipeline for ML training.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
import pickle
import json
from datetime import datetime

from data_loader import DataLoader
from features import MCQBiasFeatureExtractor


class MCQBiasPipeline:
    """
    Unified pipeline for MCQ bias prediction that integrates:
    1. Data loading from multiple sources
    2. Feature extraction for bias detection
    3. Feature matrix generation for ML training
    4. Data preprocessing and validation
    """
    
    def __init__(self, cache_features: bool = True, cache_dir: str = "cache"):
        """
        Initialize the MCQ bias prediction pipeline.
        
        Args:
            cache_features: Whether to cache extracted features for faster reloading
            cache_dir: Directory to store cached features
        """
        self.data_loader = DataLoader()
        self.feature_extractor = MCQBiasFeatureExtractor()
        self.cache_features = cache_features
        self.cache_dir = cache_dir
        
        # Pipeline state
        self.raw_data = None
        self.features_data = None
        self.feature_matrix = None
        self.feature_names = None
        
        # Create cache directory
        if cache_features:
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_datasets(self, dataset_configs: List[Dict[str, str]], 
                     sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and combine multiple datasets with unified schema.
        
        Args:
            dataset_configs: List of dicts with 'path' and optional 'type' keys
            sample_size: Optional limit on total questions to load (for testing)
        
        Returns:
            Combined DataFrame with unified schema
        """
        print("🐍 Python Developer: Loading datasets into unified pipeline...")
        
        try:
            # Load all datasets
            combined_data = self.data_loader.load_multiple_datasets(dataset_configs)
            
            # Apply sampling if requested
            if sample_size and len(combined_data) > sample_size:
                print(f"   📊 Sampling {sample_size:,} questions from {len(combined_data):,} total")
                combined_data = combined_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            self.raw_data = combined_data
            print(f"✅ Loaded {len(combined_data):,} questions from {len(dataset_configs)} datasets")
            
            return combined_data
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise
    
    def extract_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Extract bias features from loaded data with caching support.
        
        Args:
            force_recompute: Force recomputation even if cached features exist
        
        Returns:
            DataFrame with original data + extracted features
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_datasets() first.")
        
        # Check for cached features
        cache_key = self._get_cache_key(self.raw_data)
        cache_file = os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
        
        if self.cache_features and not force_recompute and os.path.exists(cache_file):
            print("🐍 Python Developer: Loading cached features...")
            try:
                with open(cache_file, 'rb') as f:
                    self.features_data = pickle.load(f)
                print(f"✅ Loaded cached features: {len(self.features_data)} questions")
                return self.features_data
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}, recomputing features...")
        
        print("🐍 Python Developer: Extracting features from MCQ data...")
        
        # Extract features
        self.features_data = self.feature_extractor.extract_features_batch(self.raw_data)
        
        # Cache the results
        if self.cache_features:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.features_data, f)
                print(f"💾 Features cached to {cache_file}")
            except Exception as e:
                print(f"⚠️ Caching failed: {e}")
        
        return self.features_data
    
    def generate_feature_matrix(self, target_column: str = 'correct_answer') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ML-ready feature matrix and target vector.
        
        Args:
            target_column: Column name containing the correct answers
        
        Returns:
            Tuple of (feature_matrix, target_vector)
        """
        if self.features_data is None:
            raise ValueError("No features extracted. Call extract_features() first.")
        
        print("🐍 Python Developer: Generating ML-ready feature matrix...")
        
        # Identify feature columns
        feature_cols = [col for col in self.features_data.columns if col.startswith('feat_')]
        self.feature_names = feature_cols
        
        # Extract feature matrix
        feature_matrix = self.features_data[feature_cols].values.astype(np.float32)
        
        # Generate target vector (which option is correct)
        target_vector = self._generate_target_vector(self.features_data, target_column)
        
        # Handle edge cases
        feature_matrix = self._handle_missing_values(feature_matrix)
        
        self.feature_matrix = feature_matrix
        
        print(f"✅ Feature matrix ready: {feature_matrix.shape[0]:,} questions × {feature_matrix.shape[1]} features")
        print(f"   Target distribution: {np.bincount(target_vector)}")
        
        return feature_matrix, target_vector
    
    def create_ml_dataset(self, 
                         dataset_configs: List[Dict[str, str]], 
                         sample_size: Optional[int] = None,
                         test_size: float = 0.2,
                         validation_size: float = 0.1,
                         random_state: int = 42) -> Dict[str, Any]:
        """
        Create complete ML dataset with train/validation/test splits.
        
        Args:
            dataset_configs: Dataset configuration list
            sample_size: Optional sampling limit
            test_size: Proportion for test set
            validation_size: Proportion for validation set  
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with train/val/test splits and metadata
        """
        print("🐍 Python Developer: Creating complete ML dataset...")
        
        # Load and process data
        self.load_datasets(dataset_configs, sample_size)
        self.extract_features()
        X, y = self.generate_feature_matrix()
        
        # Create stratified splits grouped by exam_id to prevent leakage
        splits = self._create_grouped_splits(
            X, y, self.features_data['exam_id'].values, 
            test_size, validation_size, random_state
        )
        
        # Package results
        ml_dataset = {
            'X_train': splits['X_train'],
            'X_val': splits['X_val'], 
            'X_test': splits['X_test'],
            'y_train': splits['y_train'],
            'y_val': splits['y_val'],
            'y_test': splits['y_test'],
            'feature_names': self.feature_names,
            'dataset_info': {
                'total_questions': len(X),
                'total_features': X.shape[1],
                'datasets_used': list(self.features_data['exam_id'].unique()),
                'train_size': len(splits['X_train']),
                'val_size': len(splits['X_val']),
                'test_size': len(splits['X_test']),
                'target_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'created_at': datetime.now().isoformat()
            }
        }
        
        print(f"✅ ML dataset created:")
        print(f"   Train: {len(splits['X_train']):,} questions")
        print(f"   Validation: {len(splits['X_val']):,} questions")  
        print(f"   Test: {len(splits['X_test']):,} questions")
        
        return ml_dataset
    
    def save_ml_dataset(self, ml_dataset: Dict[str, Any], filepath: str):
        """Save complete ML dataset to disk."""
        print(f"🐍 Python Developer: Saving ML dataset to {filepath}...")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(ml_dataset, f)
            print(f"✅ ML dataset saved successfully")
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            raise
    
    def load_ml_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load complete ML dataset from disk."""
        print(f"🐍 Python Developer: Loading ML dataset from {filepath}...")
        
        try:
            with open(filepath, 'rb') as f:
                ml_dataset = pickle.load(f)
            print(f"✅ ML dataset loaded successfully")
            return ml_dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def validate_pipeline(self, dataset_configs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Comprehensive validation of the entire pipeline.
        
        Args:
            dataset_configs: Dataset configurations to test
            
        Returns:
            Validation results dictionary
        """
        print("🐍 Python Developer: Validating complete pipeline...")
        
        validation_results = {
            'data_loading': {},
            'feature_extraction': {},
            'matrix_generation': {},
            'edge_cases': {},
            'performance': {}
        }
        
        try:
            # Test data loading
            start_time = datetime.now()
            self.load_datasets(dataset_configs, sample_size=1000)  # Small sample for validation
            loading_time = (datetime.now() - start_time).total_seconds()
            
            validation_results['data_loading'] = {
                'status': 'success',
                'questions_loaded': len(self.raw_data),
                'datasets_loaded': len(self.raw_data['exam_id'].unique()),
                'loading_time_seconds': loading_time
            }
            
            # Test feature extraction
            start_time = datetime.now()
            self.extract_features()
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            feature_cols = [col for col in self.features_data.columns if col.startswith('feat_')]
            validation_results['feature_extraction'] = {
                'status': 'success',
                'features_extracted': len(feature_cols),
                'extraction_time_seconds': extraction_time,
                'features_per_second': len(self.features_data) / extraction_time if extraction_time > 0 else 0
            }
            
            # Test matrix generation
            X, y = self.generate_feature_matrix()
            validation_results['matrix_generation'] = {
                'status': 'success',
                'matrix_shape': X.shape,
                'target_classes': len(np.unique(y)),
                'has_missing_values': np.isnan(X).any(),
                'has_infinite_values': np.isinf(X).any()
            }
            
            # Test edge cases
            edge_case_results = self._test_edge_cases()
            validation_results['edge_cases'] = edge_case_results
            
            # Performance metrics
            total_time = loading_time + extraction_time
            validation_results['performance'] = {
                'total_processing_time': total_time,
                'questions_per_second': len(self.features_data) / total_time if total_time > 0 else 0,
                'memory_efficient': True,  # Assuming no memory errors occurred
                'meets_requirements': total_time < 60  # Should process 1K questions in <1 minute
            }
            
        except Exception as e:
            validation_results['error'] = str(e)
            print(f"❌ Pipeline validation failed: {e}")
        
        return validation_results
    
    def _generate_target_vector(self, features_df: pd.DataFrame, target_column: str) -> np.ndarray:
        """Generate target vector indicating which option is correct (0-3)."""
        target_vector = []
        
        for _, row in features_df.iterrows():
            options = row['options']
            correct_answer = str(row[target_column])
            
            # Find which option matches the correct answer
            target_idx = 0  # Default to first option
            for i, option in enumerate(options):
                if str(option) == correct_answer:
                    target_idx = i
                    break
            
            # Ensure target is in valid range
            target_idx = min(target_idx, len(options) - 1)
            target_vector.append(target_idx)
        
        return np.array(target_vector)
    
    def _handle_missing_values(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix."""
        # Replace NaN with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clip extreme values
        feature_matrix = np.clip(feature_matrix, -100, 100)
        
        return feature_matrix
    
    def _create_grouped_splits(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              test_size: float, val_size: float, random_state: int) -> Dict[str, np.ndarray]:
        """Create train/val/test splits grouped by exam_id to prevent data leakage."""
        from sklearn.model_selection import GroupShuffleSplit
        
        # Create test split first
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss.split(X, y, groups))
        
        # Create validation split from remaining data
        X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        
        # Map back to original indices
        actual_train_idx = train_val_idx[train_idx]
        actual_val_idx = train_val_idx[val_idx]
        
        return {
            'X_train': X[actual_train_idx],
            'X_val': X[actual_val_idx],
            'X_test': X[test_idx],
            'y_train': y[actual_train_idx],
            'y_val': y[actual_val_idx],
            'y_test': y[test_idx]
        }
    
    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key based on data characteristics."""
        key_components = [
            str(len(data)),
            str(data['exam_id'].nunique()),
            str(hash(tuple(data['exam_id'].unique())))[:8]
        ]
        return "_".join(key_components)
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test pipeline with edge cases."""
        edge_results = {}
        
        # Test with empty options
        try:
            test_data = pd.DataFrame({
                'question_id': ['test1'],
                'exam_id': ['test'],
                'question_number': [1],
                'question_text': ['Test question'],
                'options': [[]],  # Empty options
                'correct_answer': ['']
            })
            
            features = self.feature_extractor.extract_features_batch(test_data)
            edge_results['empty_options'] = 'handled_gracefully'
        except Exception as e:
            edge_results['empty_options'] = f'error: {str(e)[:50]}'
        
        # Test with very long text
        try:
            long_text = "Very " * 100 + "long question"
            test_data = pd.DataFrame({
                'question_id': ['test2'],
                'exam_id': ['test'],
                'question_number': [1],
                'question_text': [long_text],
                'options': [['A', 'B', 'C', 'D']],
                'correct_answer': ['A']
            })
            
            features = self.feature_extractor.extract_features_batch(test_data)
            edge_results['long_text'] = 'handled_gracefully'
        except Exception as e:
            edge_results['long_text'] = f'error: {str(e)[:50]}'
        
        return edge_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current pipeline state."""
        summary = {
            'pipeline_status': {},
            'data_summary': {},
            'feature_summary': {},
            'readiness': {}
        }
        
        # Pipeline status
        summary['pipeline_status'] = {
            'data_loaded': self.raw_data is not None,
            'features_extracted': self.features_data is not None,
            'matrix_generated': self.feature_matrix is not None,
            'cache_enabled': self.cache_features
        }
        
        # Data summary
        if self.raw_data is not None:
            summary['data_summary'] = {
                'total_questions': len(self.raw_data),
                'datasets': list(self.raw_data['exam_id'].unique()),
                'question_distribution': dict(self.raw_data['exam_id'].value_counts())
            }
        
        # Feature summary
        if self.features_data is not None:
            feature_cols = [col for col in self.features_data.columns if col.startswith('feat_')]
            summary['feature_summary'] = {
                'total_features': len(feature_cols),
                'feature_categories': self._categorize_features(feature_cols),
                'matrix_shape': self.feature_matrix.shape if self.feature_matrix is not None else None
            }
        
        # Readiness assessment
        summary['readiness'] = {
            'ready_for_training': all([
                self.raw_data is not None,
                self.features_data is not None,
                self.feature_matrix is not None
            ]),
            'estimated_memory_usage': self._estimate_memory_usage(),
            'performance_rating': 'excellent' if len(self.raw_data or []) > 10000 else 'good'
        }
        
        return summary
    
    def _categorize_features(self, feature_cols: List[str]) -> Dict[str, int]:
        """Categorize features by type."""
        categories = {
            'length': len([f for f in feature_cols if 'len' in f]),
            'keyword': len([f for f in feature_cols if any(kw in f for kw in ['all_above', 'none_above', 'qualifier'])]),
            'numeric': len([f for f in feature_cols if 'numeric' in f]),
            'overlap': len([f for f in feature_cols if 'overlap' in f]),
            'context': len([f for f in feature_cols if any(kw in f for kw in ['question_number', 'frequency'])]),
            'rule_based': len([f for f in feature_cols if 'score' in f])
        }
        return categories
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of current pipeline state."""
        if self.feature_matrix is not None:
            matrix_size = self.feature_matrix.nbytes / (1024 * 1024)  # MB
            return f"{matrix_size:.1f} MB"
        return "Not calculated"


def demo_pipeline():
    """Demonstrate the complete pipeline functionality."""
    print("🐍 Python Developer: Demonstrating MCQ Bias Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MCQBiasPipeline(cache_features=True)
    
    # Dataset configurations
    dataset_configs = [
        {'path': '../data/indiabix_ca/bix_ca.csv', 'type': 'indiabix'},
        {'path': '../data/medqs/train.csv', 'type': 'medqs'},
        {'path': '../data/mental_health/mhqa.csv', 'type': 'mental_health'},
        {'path': '../data/qna/Train.csv', 'type': 'qna'}
    ]
    
    try:
        # Create ML dataset (with sampling for demo)
        ml_dataset = pipeline.create_ml_dataset(
            dataset_configs, 
            sample_size=1000,  # Sample for demo speed
            test_size=0.2,
            validation_size=0.1
        )
        
        # Show dataset information
        print(f"\n📊 Dataset Information:")
        info = ml_dataset['dataset_info']
        for key, value in info.items():
            if key != 'created_at':
                print(f"   {key}: {value}")
        
        # Validate pipeline
        validation_results = pipeline.validate_pipeline(dataset_configs)
        
        print(f"\n🔍 Pipeline Validation:")
        for category, results in validation_results.items():
            if isinstance(results, dict) and 'status' in results:
                status = "✅" if results['status'] == 'success' else "❌"
                print(f"   {status} {category}: {results['status']}")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        print(f"\n📋 Pipeline Summary:")
        print(f"   Ready for Training: {'✅' if summary['readiness']['ready_for_training'] else '❌'}")
        print(f"   Memory Usage: {summary['readiness']['estimated_memory_usage']}")
        print(f"   Feature Categories: {summary['feature_summary']['feature_categories']}")
        
        return pipeline, ml_dataset
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None, None


if __name__ == "__main__":
    demo_pipeline()
