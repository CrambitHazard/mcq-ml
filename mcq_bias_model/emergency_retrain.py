"""
Emergency Model Retraining - Task 1.4
Implements the final emergency retraining with optimized multi-class approach

Author: AI/ML Engineer
Priority: CRITICAL
"""

from pipeline import MCQBiasPipeline
from train import MCQBiasTrainer
from random_baseline import RandomBaselineValidator
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb

def emergency_retrain():
    print('🚨 EMERGENCY MODEL RETRAINING - TASK 1.4')
    print('=' * 50)
    print('🤖 AI/ML Engineer: Implementing final emergency fixes')
    
    # Load optimized feature set
    print('\n📊 Loading optimized feature configuration...')
    try:
        with open('feature_analysis_results.json', 'r') as f:
            feature_analysis = json.load(f)
        selected_features = feature_analysis['top_20']
        print(f'   ✅ Loaded {len(selected_features)} optimized features')
    except:
        print('   ❌ Could not load feature analysis - using all features')
        selected_features = None
    
    # Load comprehensive training data from multiple sources
    print('\n📊 Loading comprehensive training data...')
    pipeline = MCQBiasPipeline()
    
    training_datasets = [
        ('../data/mental_health/mhqa.csv', 'mental_health'),
        ('../data/medqs/train.csv', 'medqs'),
        ('../data/indiabix_ca/bix_ca.csv', 'indiabix'),
    ]
    
    all_features = []
    all_targets = []
    all_feature_names = None
    dataset_sources = []
    
    for dataset_path, dataset_type in training_datasets:
        try:
            print(f'   Loading {dataset_type}...')
            dataset_configs = [{'path': dataset_path, 'type': dataset_type}]
            pipeline.load_datasets(dataset_configs, sample_size=2000)  # Larger sample for final training
            
            # Extract features
            pipeline.extract_features()
            X, y = pipeline.generate_feature_matrix()
            feature_names = pipeline.feature_names
            
            all_features.append(X)
            all_targets.append(y)
            all_feature_names = feature_names
            dataset_sources.extend([dataset_type] * len(X))
            
            print(f'      ✅ {dataset_type}: {X.shape[0]} questions, {X.shape[1]} features')
            
        except Exception as e:
            print(f'      ❌ Failed {dataset_type}: {e}')
    
    if not all_features:
        print('❌ No training data loaded - cannot proceed')
        return None
    
    # Combine all training data
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_targets)
    dataset_sources = np.array(dataset_sources)
    
    print(f'\n🎯 COMPREHENSIVE TRAINING DATASET:')
    print(f'   Total questions: {len(X_combined):,}')
    print(f'   Total features: {X_combined.shape[1]}')
    print(f'   Class distribution: {np.bincount(y_combined)}')
    print(f'   Dataset sources: {dict(zip(*np.unique(dataset_sources, return_counts=True)))}')
    
    # Apply feature selection if available
    if selected_features and all_feature_names:
        print(f'\n🔍 Applying optimized feature selection...')
        feature_indices = []
        for feat in selected_features:
            if feat in all_feature_names:
                feature_indices.append(all_feature_names.index(feat))
        
        if len(feature_indices) > 0:
            X_selected = X_combined[:, feature_indices]
            selected_feature_names = [all_feature_names[i] for i in feature_indices]
            
            print(f'   ✅ Selected {len(feature_indices)} features from {X_combined.shape[1]} total')
            
            # Use selected features
            X_train = X_selected
            feature_names_final = selected_feature_names
        else:
            print('   ⚠️ No feature indices found - using all features')
            X_train = X_combined
            feature_names_final = all_feature_names
    else:
        print(f'\n📊 Using all available features...')
        X_train = X_combined
        feature_names_final = all_feature_names
    
    print(f'   Final training matrix: {X_train.shape}')
    
    # Hyperparameter optimization for multi-class objective
    print(f'\n🔧 Optimizing hyperparameters for multi-class classification...')
    
    # Define hyperparameter grid for LightGBM multi-class
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 30, 50]
    }
    
    # Create base LightGBM model with multi-class objective
    base_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        metric='multi_logloss',
        verbose=-1,
        random_state=42
    )
    
    # Perform grid search with cross-validation
    print(f'   🔍 Running hyperparameter optimization...')
    from sklearn.model_selection import StratifiedKFold
    
    # Use smaller param grid for faster optimization
    quick_param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [5, 7],
        'learning_rate': [0.1, 0.15],
        'num_leaves': [50, 100]
    }
    
    cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, 
        quick_param_grid, 
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_combined)
    
    print(f'   ✅ Best hyperparameters found:')
    for param, value in grid_search.best_params_.items():
        print(f'      {param}: {value}')
    print(f'   📊 Best CV accuracy: {grid_search.best_score_:.3f}')
    
    # Train final optimized model
    print(f'\n🚀 Training final optimized model...')
    
    # Create trainer with optimized hyperparameters
    trainer = MCQBiasTrainer(model_type='lightgbm')
    
    # Override model creation to use optimized hyperparameters
    optimized_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        metric='multi_logloss',
        verbose=-1,
        random_state=42,
        **grid_search.best_params_
    )
    
    # Train with custom model
    trainer.model = optimized_model
    trainer.model_type = 'lightgbm'
    
    # Perform training
    results = trainer.train(X_train, y_combined, feature_names_final, verbose=True)
    
    # Comprehensive validation against random baseline
    print(f'\n🧪 Comprehensive Random Baseline Validation...')
    validator = RandomBaselineValidator()
    
    # Get final model predictions
    val_results = results['validation_results']
    final_accuracy = val_results['accuracy']
    theoretical_baseline = val_results['theoretical_random_baseline']
    realistic_baseline = val_results['realistic_random_baseline']
    improvement_theoretical = val_results['improvement_over_theoretical']
    improvement_realistic = val_results['improvement_over_realistic']
    is_significant = val_results['statistical_comparison']['is_significant']
    p_value = val_results['statistical_comparison']['p_value']
    
    print(f'   📊 Final Model Performance:')
    print(f'      Accuracy: {final_accuracy:.1%}')
    print(f'      Theoretical Random Baseline: {theoretical_baseline:.1%}')
    print(f'      Realistic Random Baseline: {realistic_baseline:.1%}')
    print(f'      Improvement vs Theoretical: {improvement_theoretical:+.1f}%')
    print(f'      Improvement vs Realistic: {improvement_realistic:+.1f}%')
    print(f'      Statistical Significance: p = {p_value:.4f}')
    print(f'      Significant (p < 0.05): {"✅" if is_significant else "❌"} {is_significant}')
    
    # Validate against all test datasets
    print(f'\n🔬 Cross-Dataset Validation...')
    
    test_datasets = [
        ('../data/medqs/validation.csv', 'medqs', 'MedQS Validation'),
        ('../data/qna/Test.csv', 'qna', 'QNA Test'),
        ('../data/indiabix_ca/bix_ca.csv', 'indiabix', 'IndiaBix CA'),
    ]
    
    cross_dataset_results = {}
    total_questions = 0
    weighted_accuracy = 0
    
    for dataset_path, dataset_type, dataset_name in test_datasets:
        try:
            print(f'\n   🎯 Testing on {dataset_name}...')
            
            # Load test dataset
            test_pipeline = MCQBiasPipeline()
            test_configs = [{'path': dataset_path, 'type': dataset_type}]
            test_pipeline.load_datasets(test_configs)
            
            # Extract features
            test_pipeline.extract_features()
            X_test, y_test = test_pipeline.generate_feature_matrix()
            
            # Apply same feature selection
            if selected_features and len(feature_indices) > 0:
                X_test = X_test[:, feature_indices]
            
            # Sample for manageable testing
            n_test = min(1000, len(X_test))
            test_indices = np.random.choice(len(X_test), n_test, replace=False)
            X_test_sample = X_test[test_indices]
            y_test_sample = y_test[test_indices]
            
            # Make predictions
            predictions, probabilities = trainer.predict(X_test_sample)
            test_accuracy = np.mean(predictions == y_test_sample)
            
            # Statistical validation
            comparison = validator.compare_to_random(predictions, y_test_sample, dataset_name)
            
            cross_dataset_results[dataset_name] = {
                'accuracy': test_accuracy,
                'n_questions': n_test,
                'improvement_percent': comparison['improvement_relative_percent'],
                'p_value': comparison['p_value'],
                'is_significant': comparison['is_significant']
            }
            
            total_questions += n_test
            weighted_accuracy += test_accuracy * n_test
            
            print(f'      📊 Accuracy: {test_accuracy:.1%} ({n_test} questions)')
            print(f'      📊 Improvement: {comparison["improvement_relative_percent"]:+.1f}%')
            print(f'      📊 Significant: {"✅" if comparison["is_significant"] else "❌"} (p = {comparison["p_value"]:.4f})')
            
        except Exception as e:
            print(f'      ❌ Failed: {e}')
            cross_dataset_results[dataset_name] = {'error': str(e)}
    
    # Calculate overall performance
    if total_questions > 0:
        overall_accuracy = weighted_accuracy / total_questions
        successful_tests = sum(1 for r in cross_dataset_results.values() 
                             if 'error' not in r and r['accuracy'] > 0.25)
        total_tests = len([r for r in cross_dataset_results.values() if 'error' not in r])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f'\n📊 OVERALL CROSS-DATASET PERFORMANCE:')
        print(f'   Weighted Average Accuracy: {overall_accuracy:.1%}')
        print(f'   Success Rate (>25%): {success_rate:.1%}')
        print(f'   Total Questions Tested: {total_questions:,}')
    
    # Save optimized model
    print(f'\n💾 Saving optimized model...')
    try:
        trainer.save_model('mcq_bias_model_optimized.pkl')
        print(f'   ✅ Model saved to mcq_bias_model_optimized.pkl')
    except Exception as e:
        print(f'   ⚠️ Model save failed: {e}')
    
    # Document comprehensive performance
    print(f'\n📝 Documenting comprehensive performance...')
    
    performance_doc = {
        'emergency_retraining_results': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_data': {
                'total_questions': len(X_combined),
                'total_features': X_combined.shape[1],
                'selected_features': len(feature_names_final),
                'class_distribution': np.bincount(y_combined).tolist(),
                'dataset_sources': dict(zip(*np.unique(dataset_sources, return_counts=True)))
            },
            'hyperparameter_optimization': {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'optimization_method': 'GridSearchCV with 3-fold StratifiedKFold'
            },
            'final_model_performance': {
                'accuracy': final_accuracy,
                'theoretical_random_baseline': theoretical_baseline,
                'realistic_random_baseline': realistic_baseline,
                'improvement_over_theoretical': improvement_theoretical,
                'improvement_over_realistic': improvement_realistic,
                'statistical_significance': {
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'confidence_interval': val_results['statistical_comparison']['confidence_interval_95']
                }
            },
            'cross_dataset_validation': cross_dataset_results,
            'overall_performance': {
                'weighted_accuracy': overall_accuracy if total_questions > 0 else None,
                'success_rate': success_rate if total_questions > 0 else None,
                'total_questions_tested': total_questions
            } if total_questions > 0 else None
        }
    }
    
    with open('emergency_retraining_report.json', 'w') as f:
        json.dump(performance_doc, f, indent=2, default=str)
    
    print(f'   ✅ Performance documentation saved to emergency_retraining_report.json')
    
    # Final verdict
    print(f'\n🎯 EMERGENCY RETRAINING VERDICT:')
    
    if final_accuracy > 0.35:
        print(f'   🎉 EXCELLENT: {final_accuracy:.1%} exceeds 35% target!')
        verdict = 'EXCELLENT'
    elif final_accuracy > 0.30:
        print(f'   ✅ SUCCESS: {final_accuracy:.1%} meets 30% minimum target!')
        verdict = 'SUCCESS'
    elif final_accuracy > 0.25:
        print(f'   📈 PARTIAL: {final_accuracy:.1%} above random but below target')
        verdict = 'PARTIAL'
    else:
        print(f'   ❌ FAILURE: {final_accuracy:.1%} still sub-random')
        verdict = 'FAILURE'
    
    print(f'   Statistical Significance: {"✅ Achieved" if is_significant else "❌ Not Achieved"}')
    print(f'   Cross-Dataset Consistency: {"✅ Good" if total_questions > 0 and success_rate > 0.66 else "⚠️ Mixed"}')
    
    return {
        'verdict': verdict,
        'final_accuracy': final_accuracy,
        'is_significant': is_significant,
        'cross_dataset_results': cross_dataset_results,
        'performance_doc': performance_doc
    }

if __name__ == "__main__":
    print('🚨 Starting Emergency Model Retraining - Task 1.4')
    results = emergency_retrain()
    
    if results:
        print(f'\n🔥 EMERGENCY RETRAINING COMPLETED')
        print(f'   Final Verdict: {results["verdict"]}')
        print(f'   Accuracy: {results["final_accuracy"]:.1%}')
        print(f'   Statistically Significant: {"✅" if results["is_significant"] else "❌"}')
    else:
        print(f'\n❌ EMERGENCY RETRAINING FAILED')
