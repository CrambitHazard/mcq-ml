"""
Emergency Feature Analysis - Fix the actual model performance
Author: AI/ML Engineer (Emergency Response)
"""

from pipeline import MCQBiasPipeline
from train import MCQBiasTrainer
import numpy as np
import pandas as pd
import json

def main():
    print('🔥 EMERGENCY FEATURE ANALYSIS - FIXING THE ACTUAL MODEL')
    print('=' * 60)

    # Load REAL data and get feature importance
    pipeline = MCQBiasPipeline()

    print('📊 Loading real datasets...')
    datasets = [
        ('../data/mental_health/mhqa.csv', 'mental_health'),
        ('../data/medqs/train.csv', 'medqs'),
    ]

    all_features = []
    all_targets = []
    all_feature_names = None

    for dataset_path, dataset_type in datasets:
        try:
            print(f'   Loading {dataset_type}...')
            # Use correct pipeline method
            dataset_configs = [{'path': dataset_path, 'type': dataset_type}]
            pipeline.load_datasets(dataset_configs, sample_size=1000)  # Sample for speed
            
            # Extract features and generate matrix
            pipeline.extract_features()
            X, y = pipeline.generate_feature_matrix()
            feature_names = pipeline.feature_names
            
            # Take sample to speed up analysis
            n_sample = min(1000, len(X))
            sample_indices = np.random.choice(len(X), n_sample, replace=False)
            
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            all_features.append(X_sample)
            all_targets.append(y_sample)
            all_feature_names = feature_names
            
            print(f'      ✅ {dataset_type}: {X_sample.shape[0]} questions, {X_sample.shape[1]} features')
            
        except Exception as e:
            print(f'      ❌ Failed {dataset_type}: {e}')

    if not all_features:
        print('❌ NO DATA LOADED - CANNOT ANALYZE FEATURES')
        return

    # Combine datasets
    X_combined = np.vstack(all_features)
    y_combined = np.hstack(all_targets)

    print('')
    print('🎯 COMBINED DATASET FOR FEATURE ANALYSIS:')
    print(f'   Total questions: {len(X_combined):,}')
    print(f'   Total features: {X_combined.shape[1]}')
    print(f'   Class distribution: {np.bincount(y_combined)}')

    # Train model to get feature importance
    print('')
    print('🔬 TRAINING MODEL FOR FEATURE IMPORTANCE...')
    trainer = MCQBiasTrainer(model_type='lightgbm')
    results = trainer.train(X_combined, y_combined, all_feature_names, verbose=False)

    # Get feature importance
    print('')
    print('📊 ANALYZING FEATURE IMPORTANCE...')
    feature_importance = trainer.get_feature_importance()

    if feature_importance:
        # feature_importance is a dict of {feature_name: importance_score}
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance} 
            for feature, importance in feature_importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print('')
        print('🏆 TOP 20 MOST IMPORTANT FEATURES:')
        print('   Rank | Feature Name                           | Importance')
        print('   -----|----------------------------------------|----------')
        
        for i, (idx, row) in enumerate(importance_df.head(20).iterrows()):
            feature_name = row['feature']
            importance = row['importance']
            print(f'   {i+1:2d}   | {feature_name:38s} | {importance:8.3f}')
        
        # Calculate cumulative importance
        importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        
        # Find number of features for 80% and 90% of importance
        features_80 = len(importance_df[importance_df['cumulative'] <= 0.8])
        features_90 = len(importance_df[importance_df['cumulative'] <= 0.9])
        
        total_importance = importance_df['importance'].sum()
        top10_importance = importance_df.head(10)['importance'].sum()
        top20_importance = importance_df.head(20)['importance'].sum()
        
        print('')
        print('📈 FEATURE SELECTION RECOMMENDATIONS:')
        print(f'   Top 10 features: {top10_importance / total_importance:.1%} of total importance')
        print(f'   Top 20 features: {top20_importance / total_importance:.1%} of total importance')
        print(f'   Features for 80% importance: {features_80}')
        print(f'   Features for 90% importance: {features_90}')
        
        # Save top features for retraining
        top_15_features = importance_df.head(15)['feature'].tolist()
        top_20_features = importance_df.head(20)['feature'].tolist()
        top_30_features = importance_df.head(30)['feature'].tolist()
        
        feature_sets = {
            'top_15': top_15_features,
            'top_20': top_20_features,
            'top_30': top_30_features,
            'full_ranking': importance_df[['feature', 'importance']].to_dict('records')
        }
        
        print('')
        print('💾 SAVING FEATURE SETS FOR EMERGENCY RETRAINING...')
        
        with open('feature_analysis_results.json', 'w') as f:
            json.dump(feature_sets, f, indent=2)
        
        print('   ✅ Saved to feature_analysis_results.json')
        
        # Quick validation - retrain with top 20 features
        print('')
        print('🚀 QUICK VALIDATION - RETRAINING WITH TOP 20 FEATURES...')
        
        # Get feature indices
        feature_indices = [all_feature_names.index(feat) for feat in top_20_features if feat in all_feature_names]
        
        if len(feature_indices) > 0:
            X_selected = X_combined[:, feature_indices]
            selected_feature_names = [all_feature_names[i] for i in feature_indices]
            
            print(f'   Selected {len(feature_indices)} features for validation')
            
            # Train with selected features
            trainer_selected = MCQBiasTrainer(model_type='lightgbm')
            results_selected = trainer_selected.train(X_selected, y_combined, selected_feature_names, verbose=False)
            
            val_results = results_selected['validation_results']
            accuracy_selected = val_results['accuracy']
            improvement_theoretical = val_results['improvement_over_theoretical']
            improvement_realistic = val_results['improvement_over_realistic']
            
            print('')
            print('🎯 FEATURE SELECTION VALIDATION RESULTS:')
            print(f'   Original model (all features): {results["validation_results"]["accuracy"]:.1%}')
            print(f'   Selected model ({len(feature_indices)} features): {accuracy_selected:.1%}')
            print(f'   Improvement vs theoretical random: {improvement_theoretical:+.1f}%')
            print(f'   Improvement vs realistic random: {improvement_realistic:+.1f}%')
            
            if accuracy_selected > 0.28:  # Above realistic random baseline
                print('   ✅ SUCCESS: Feature selection improves performance!')
            elif accuracy_selected > results["validation_results"]["accuracy"]:
                print('   📈 IMPROVEMENT: Better than original but still sub-random')
            else:
                print('   ❌ NO IMPROVEMENT: Feature selection did not help')
        
        return feature_sets
    else:
        print('❌ FAILED TO GET FEATURE IMPORTANCE')
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print('')
        print('🔥 FEATURE ANALYSIS COMPLETE - READY FOR EMERGENCY RETRAINING')
    else:
        print('')
        print('❌ FEATURE ANALYSIS FAILED')
