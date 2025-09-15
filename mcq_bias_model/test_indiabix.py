"""
Test the model on IndiaBix dataset
Author: AI/ML Engineer (Reality Check)
"""

from pipeline import MCQBiasPipeline
from train import MCQBiasTrainer
from random_baseline import RandomBaselineValidator
import numpy as np
import json

def test_indiabix():
    print('🧪 TESTING ON INDIABIX DATASET')
    print('=' * 40)

    # Load the top features
    try:
        with open('feature_analysis_results.json', 'r') as f:
            feature_analysis = json.load(f)
        top_features = feature_analysis['top_20']
        print(f'✅ Loaded {len(top_features)} optimized features')
    except:
        print('⚠️ Could not load feature analysis, using all features')
        top_features = None

    # Test IndiaBix dataset
    try:
        print(f'\n🎯 TESTING INDIABIX CA DATASET')
        print('-' * 30)
        
        # Load test data
        pipeline = MCQBiasPipeline()
        dataset_configs = [{'path': '../data/indiabix_ca/bix_ca.csv', 'type': 'indiabix'}]
        pipeline.load_datasets(dataset_configs)
        
        # Extract features
        pipeline.extract_features()
        X_test, y_test = pipeline.generate_feature_matrix()
        feature_names = pipeline.feature_names
        
        print(f'📊 Test data: {len(X_test):,} questions, {X_test.shape[1]} features')
        print(f'📊 Class distribution: {np.bincount(y_test)}')
        
        # Split for training/testing
        n_total = len(X_test)
        n_train = min(1000, n_total // 2)
        train_indices = np.random.choice(n_total, n_train, replace=False)
        test_indices = np.setdiff1d(np.arange(n_total), train_indices)
        
        X_train_sample = X_test[train_indices]
        y_train_sample = y_test[train_indices] 
        X_test_real = X_test[test_indices]
        y_test_real = y_test[test_indices]
        
        print(f'📊 Train/Test split: {len(X_train_sample)} / {len(X_test_real)}')
        
        # Test with ALL features
        print(f'\n🤖 Testing with ALL features...')
        trainer_all = MCQBiasTrainer(model_type='lightgbm')
        results_train = trainer_all.train(X_train_sample, y_train_sample, feature_names, verbose=False)
        
        # Test on held-out data
        predictions_all, probabilities_all = trainer_all.predict(X_test_real)
        accuracy_all = np.mean(predictions_all == y_test_real)
        
        # Statistical comparison
        validator = RandomBaselineValidator()
        comparison_all = validator.compare_to_random(predictions_all, y_test_real, 'IndiaBix (All Features)')
        
        improvement_all = comparison_all['improvement_relative_percent']
        p_value_all = comparison_all['p_value']
        is_significant_all = comparison_all['is_significant']
        
        print(f'   📈 Accuracy (all features): {accuracy_all:.1%}')
        print(f'   📈 Improvement vs random: {improvement_all:+.1f}%')
        print(f'   📈 Statistical significance: p = {p_value_all:.4f}')
        print(f'   📈 Significant (p < 0.05): {"✅" if is_significant_all else "❌"} {is_significant_all}')
        
        # Test with selected features if available
        accuracy_selected = None
        comparison_selected = None
        
        if top_features:
            print(f'\n🔍 Testing with TOP 20 features...')
            
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
                
                comparison_selected = validator.compare_to_random(predictions_selected, y_test_real, 'IndiaBix (Selected)')
                
                improvement_selected = comparison_selected['improvement_relative_percent']
                p_value_selected = comparison_selected['p_value']
                is_significant_selected = comparison_selected['is_significant']
                
                print(f'   📈 Accuracy (top 20 features): {accuracy_selected:.1%}')
                print(f'   📈 Improvement vs random: {improvement_selected:+.1f}%')
                print(f'   📈 Statistical significance: p = {p_value_selected:.4f}')
                print(f'   📈 Significant (p < 0.05): {"✅" if is_significant_selected else "❌"} {is_significant_selected}')
        
        # Verdict for IndiaBix
        print(f'\n🎯 INDIABIX VERDICT:')
        if accuracy_all > 0.30:
            print(f'   ✅ SUCCESS: {accuracy_all:.1%} > 30% target!')
        elif accuracy_all > 0.25:
            print(f'   📈 PARTIAL: {accuracy_all:.1%} > random but < 30%')
        else:
            print(f'   ❌ FAILURE: {accuracy_all:.1%} ≤ random baseline')
        
        # Save IndiaBix results
        indiabix_results = {
            'dataset': 'IndiaBix CA',
            'n_questions': len(X_test_real),
            'class_distribution': np.bincount(y_test_real).tolist(),
            'all_features': {
                'accuracy': accuracy_all,
                'improvement_percent': improvement_all,
                'p_value': p_value_all,
                'is_significant': is_significant_all
            },
            'selected_features': {
                'accuracy': accuracy_selected,
                'improvement_percent': comparison_selected['improvement_relative_percent'] if comparison_selected else None,
                'p_value': comparison_selected['p_value'] if comparison_selected else None,
                'is_significant': comparison_selected['is_significant'] if comparison_selected else None
            } if accuracy_selected else None
        }
        
        with open('indiabix_test_results.json', 'w') as f:
            json.dump(indiabix_results, f, indent=2, default=str)
        
        print(f'\n💾 IndiaBix results saved to indiabix_test_results.json')
        
        return indiabix_results
        
    except Exception as e:
        print(f'❌ FAILED to test IndiaBix: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_indiabix()
    print(f'\n🔥 INDIABIX TESTING COMPLETE')
    
    if results:
        accuracy = results['all_features']['accuracy']
        significant = results['all_features']['is_significant']
        
        print(f'\n📊 FINAL INDIABIX SUMMARY:')
        print(f'   Accuracy: {accuracy:.1%}')
        print(f'   Statistically Significant: {"✅" if significant else "❌"}')
        
        if accuracy > 0.30:
            print(f'   🎉 EXCELLENT PERFORMANCE!')
        elif accuracy > 0.25:
            print(f'   📈 MODERATE SUCCESS - ABOVE RANDOM')
        else:
            print(f'   ❌ BELOW RANDOM - NEEDS WORK')
