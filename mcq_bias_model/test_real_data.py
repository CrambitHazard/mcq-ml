"""
Test the improved model on actual test datasets
Author: AI/ML Engineer (Reality Check)
"""

from pipeline import MCQBiasPipeline
from train import MCQBiasTrainer
from random_baseline import RandomBaselineValidator
import numpy as np
import pandas as pd
import json

def test_on_real_datasets():
    print('🧪 TESTING MODEL ON ACTUAL TEST DATASETS')
    print('=' * 50)
    print('CRITICAL: This is the REAL test - no more training data!')
    
    # Load the top features from our analysis
    print('📊 Loading optimized feature set...')
    try:
        with open('feature_analysis_results.json', 'r') as f:
            feature_analysis = json.load(f)
        top_features = feature_analysis['top_20']
        print(f'   ✅ Loaded {len(top_features)} optimized features')
    except:
        print('   ⚠️ Could not load feature analysis, using all features')
        top_features = None

    # Test datasets
    test_datasets = [
        # Mental Health has separate test data
        ('../data/mental_health/mhqa-b.csv', 'mental_health', 'Mental Health Test'),
        # MedQS validation set
        ('../data/medqs/validation.csv', 'medqs', 'MedQS Validation'),
        # QNA test set 
        ('../data/qna/Test.csv', 'qna', 'QNA Test'),
    ]
    
    all_results = {}
    validator = RandomBaselineValidator()
    
    for dataset_path, dataset_type, dataset_name in test_datasets:
        print(f'\n🎯 TESTING ON: {dataset_name}')
        print('-' * 40)
        
        try:
            # Load test data
            pipeline = MCQBiasPipeline()
            dataset_configs = [{'path': dataset_path, 'type': dataset_type}]
            pipeline.load_datasets(dataset_configs)
            
            # Extract features
            pipeline.extract_features()
            X_test, y_test = pipeline.generate_feature_matrix()
            feature_names = pipeline.feature_names
            
            print(f'   📊 Test data: {len(X_test):,} questions, {X_test.shape[1]} features')
            print(f'   📊 Class distribution: {np.bincount(y_test)}')
            
            # Test with all features
            print(f'\n   🤖 Testing with ALL features...')
            trainer_all = MCQBiasTrainer(model_type='lightgbm')
            
            # Train on a sample and test on the rest for fair evaluation
            n_train = min(1000, len(X_test) // 2)
            train_indices = np.random.choice(len(X_test), n_train, replace=False)
            test_indices = np.setdiff1d(np.arange(len(X_test)), train_indices)
            
            X_train_sample = X_test[train_indices]
            y_train_sample = y_test[train_indices] 
            X_test_real = X_test[test_indices]
            y_test_real = y_test[test_indices]
            
            # Train model
            results_train = trainer_all.train(X_train_sample, y_train_sample, feature_names, verbose=False)
            
            # Test on held-out data
            predictions_all, probabilities_all = trainer_all.predict(X_test_real)
            accuracy_all = np.mean(predictions_all == y_test_real)
            
            # Statistical comparison
            comparison_all = validator.compare_to_random(predictions_all, y_test_real, f'{dataset_name} (All Features)')
            
            print(f'      📈 Accuracy (all features): {accuracy_all:.1%}')
            print(f'      📈 Improvement vs random: {comparison_all["improvement_relative_percent"]:+.1f}%')
            print(f'      📈 Statistical significance: p = {comparison_all["p_value"]:.4f}')
            
            # Test with selected features if available
            if top_features:
                print(f'\n   🔍 Testing with TOP 20 features...')
                
                # Get feature indices for selected features
                feature_indices = []
                for feat in top_features:
                    if feat in feature_names:
                        feature_indices.append(feature_names.index(feat))
                
                if len(feature_indices) > 0:
                    X_train_selected = X_train_sample[:, feature_indices]
                    X_test_selected = X_test_real[:, feature_indices]
                    selected_feature_names = [feature_names[i] for i in feature_indices]
                    
                    # Train selected model
                    trainer_selected = MCQBiasTrainer(model_type='lightgbm')
                    results_selected = trainer_selected.train(X_train_selected, y_train_sample, selected_feature_names, verbose=False)
                    
                    # Test selected model
                    predictions_selected, probabilities_selected = trainer_selected.predict(X_test_selected)
                    accuracy_selected = np.mean(predictions_selected == y_test_real)
                    
                    comparison_selected = validator.compare_to_random(predictions_selected, y_test_real, f'{dataset_name} (Selected)')
                    
                    print(f'      📈 Accuracy (top 20 features): {accuracy_selected:.1%}')
                    print(f'      📈 Improvement vs random: {comparison_selected["improvement_relative_percent"]:+.1f}%')
                    print(f'      📈 Statistical significance: p = {comparison_selected["p_value"]:.4f}')
                else:
                    accuracy_selected = None
                    comparison_selected = None
            else:
                accuracy_selected = None
                comparison_selected = None
            
            # Store results
            all_results[dataset_name] = {
                'n_questions': len(X_test_real),
                'class_distribution': np.bincount(y_test_real).tolist(),
                'all_features': {
                    'accuracy': accuracy_all,
                    'improvement_percent': comparison_all['improvement_relative_percent'],
                    'p_value': comparison_all['p_value'],
                    'is_significant': comparison_all['is_significant']
                },
                'selected_features': {
                    'accuracy': accuracy_selected,
                    'improvement_percent': comparison_selected['improvement_relative_percent'] if comparison_selected else None,
                    'p_value': comparison_selected['p_value'] if comparison_selected else None,
                    'is_significant': comparison_selected['is_significant'] if comparison_selected else None
                } if accuracy_selected else None
            }
            
            # Verdict for this dataset
            if accuracy_all > 0.30:
                print(f'      ✅ SUCCESS: {accuracy_all:.1%} > 30% target!')
            elif accuracy_all > 0.25:
                print(f'      📈 PARTIAL: {accuracy_all:.1%} > random but < 30%')
            else:
                print(f'      ❌ FAILURE: {accuracy_all:.1%} ≤ random baseline')
                
        except Exception as e:
            print(f'   ❌ FAILED to test {dataset_name}: {e}')
            all_results[dataset_name] = {'error': str(e)}
    
    # Overall summary
    print(f'\n📋 OVERALL TEST RESULTS SUMMARY:')
    print('=' * 50)
    
    successful_tests = 0
    total_accuracy = 0
    total_questions = 0
    
    for dataset_name, results in all_results.items():
        if 'error' not in results:
            accuracy = results['all_features']['accuracy']
            n_questions = results['n_questions']
            is_significant = results['all_features']['is_significant']
            
            print(f'{dataset_name:20s}: {accuracy:6.1%} ({n_questions:,} questions) {"✅" if is_significant else "❌"}')
            
            if accuracy > 0.25:  # Above random
                successful_tests += 1
            
            total_accuracy += accuracy * n_questions
            total_questions += n_questions
        else:
            print(f'{dataset_name:20s}: ERROR - {results["error"]}')
    
    if total_questions > 0:
        weighted_accuracy = total_accuracy / total_questions
        success_rate = successful_tests / len([r for r in all_results.values() if 'error' not in r])
        
        print(f'\n🎯 FINAL VERDICT:')
        print(f'   Weighted Average Accuracy: {weighted_accuracy:.1%}')
        print(f'   Success Rate (>25%): {success_rate:.1%}')
        print(f'   Total Questions Tested: {total_questions:,}')
        
        if weighted_accuracy > 0.35:
            print(f'   🎉 EXCELLENT: Model exceeds 35% target!')
        elif weighted_accuracy > 0.30:
            print(f'   ✅ SUCCESS: Model meets 30% target!')
        elif weighted_accuracy > 0.25:
            print(f'   📈 PARTIAL: Above random but below target')
        else:
            print(f'   ❌ FAILURE: Model still sub-random')
    
    # Save results
    with open('real_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f'\n💾 Results saved to real_test_results.json')
    
    return all_results

if __name__ == "__main__":
    results = test_on_real_datasets()
    print(f'\n🔥 REAL DATA TESTING COMPLETE')
