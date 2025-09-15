"""
Random Baseline Validation Module
Implements proper random baseline validation for MCQ prediction.

Author: Tester + AI/ML Engineer
Purpose: Ensure all model evaluations include proper random baseline comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RandomBaselineValidator:
    """
    Comprehensive random baseline validation for MCQ prediction systems.
    
    Provides:
    - True random prediction baseline
    - Statistical significance testing
    - Cross-dataset validation
    - Confidence interval estimation
    """
    
    def __init__(self, n_classes: int = 4, random_state: int = 42):
        """
        Initialize random baseline validator.
        
        Args:
            n_classes: Number of answer options (default: 4 for A,B,C,D)
            random_state: Random seed for reproducibility
        """
        self.n_classes = n_classes
        self.random_state = random_state
        self.expected_accuracy = 1.0 / n_classes
        
        print(f"🧪 Tester: Initializing Random Baseline Validator")
        print(f"   Expected random accuracy: {self.expected_accuracy:.1%}")
    
    def generate_random_predictions(self, n_questions: int, 
                                  class_weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate truly random predictions for MCQ questions.
        
        Args:
            n_questions: Number of questions to predict
            class_weights: Optional weights for each class (default: uniform)
            
        Returns:
            Array of random predictions (0, 1, 2, 3)
        """
        np.random.seed(self.random_state)
        
        if class_weights is None:
            # Uniform random distribution
            predictions = np.random.choice(self.n_classes, size=n_questions)
        else:
            # Weighted random distribution
            if len(class_weights) != self.n_classes:
                raise ValueError(f"class_weights must have {self.n_classes} elements")
            if not np.isclose(sum(class_weights), 1.0):
                raise ValueError("class_weights must sum to 1.0")
            
            predictions = np.random.choice(self.n_classes, size=n_questions, p=class_weights)
        
        return predictions
    
    def validate_random_baseline(self, n_questions: int, n_trials: int = 1000) -> Dict[str, Any]:
        """
        Validate that random predictions achieve expected baseline accuracy.
        
        Args:
            n_questions: Number of questions in test
            n_trials: Number of random trials to run
            
        Returns:
            Validation results with statistics
        """
        print(f"🧪 Tester: Validating random baseline with {n_trials} trials...")
        
        accuracies = []
        
        for trial in range(n_trials):
            # Generate random true answers and random predictions
            true_answers = np.random.choice(self.n_classes, size=n_questions)
            random_predictions = self.generate_random_predictions(n_questions)
            
            # Calculate accuracy
            accuracy = np.mean(true_answers == random_predictions)
            accuracies.append(accuracy)
        
        accuracies = np.array(accuracies)
        
        # Statistical analysis
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        confidence_interval = stats.t.interval(
            0.95, len(accuracies)-1, 
            loc=mean_accuracy, 
            scale=stats.sem(accuracies)
        )
        
        # Check if results match theory
        theoretical_accuracy = self.expected_accuracy
        is_valid = abs(mean_accuracy - theoretical_accuracy) < 0.01  # Within 1%
        
        results = {
            'n_questions': n_questions,
            'n_trials': n_trials,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'theoretical_accuracy': theoretical_accuracy,
            'confidence_interval_95': confidence_interval,
            'is_valid_baseline': is_valid,
            'all_accuracies': accuracies.tolist()
        }
        
        print(f"   ✅ Random baseline validation completed")
        print(f"   📊 Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"   📊 Theoretical: {theoretical_accuracy:.3f}")
        print(f"   📊 95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        print(f"   {'✅' if is_valid else '❌'} Baseline valid: {is_valid}")
        
        return results
    
    def compare_to_random(self, model_predictions: np.ndarray, 
                         true_answers: np.ndarray,
                         model_name: str = "Model") -> Dict[str, Any]:
        """
        Compare model performance to random baseline with statistical testing.
        
        Args:
            model_predictions: Model predictions (0, 1, 2, 3)
            true_answers: True answers (0, 1, 2, 3)
            model_name: Name of model for reporting
            
        Returns:
            Comparison results with statistical significance
        """
        print(f"🧪 Tester: Comparing {model_name} to random baseline...")
        
        n_questions = len(true_answers)
        
        # Calculate model accuracy
        model_accuracy = np.mean(model_predictions == true_answers)
        
        # Generate random baseline predictions
        random_predictions = self.generate_random_predictions(n_questions)
        random_accuracy = np.mean(random_predictions == true_answers)
        
        # Statistical significance testing
        # Use McNemar's test for paired binary classification results
        model_correct = (model_predictions == true_answers)
        random_correct = (random_predictions == true_answers)
        
        # Create contingency table for McNemar's test
        both_correct = np.sum(model_correct & random_correct)
        model_only = np.sum(model_correct & ~random_correct)
        random_only = np.sum(~model_correct & random_correct)
        both_wrong = np.sum(~model_correct & ~random_correct)
        
        # McNemar's test (if sufficient discordant pairs)
        if model_only + random_only > 0:
            mcnemar_stat = (abs(model_only - random_only) - 1)**2 / (model_only + random_only)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        # Effect size (improvement over random)
        improvement_absolute = model_accuracy - random_accuracy
        improvement_relative = ((model_accuracy - self.expected_accuracy) / 
                               self.expected_accuracy) * 100
        
        # Confidence interval for model accuracy
        model_ci_raw = stats.binom.interval(0.95, n_questions, model_accuracy)
        model_ci = (model_ci_raw[0] / n_questions, model_ci_raw[1] / n_questions)
        
        results = {
            'model_name': model_name,
            'n_questions': n_questions,
            'model_accuracy': model_accuracy,
            'random_accuracy': random_accuracy,
            'theoretical_random': self.expected_accuracy,
            'improvement_absolute': improvement_absolute,
            'improvement_relative_percent': improvement_relative,
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'model_ci_95': model_ci,
            'contingency_table': {
                'both_correct': both_correct,
                'model_only_correct': model_only,
                'random_only_correct': random_only,
                'both_wrong': both_wrong
            }
        }
        
        print(f"   📊 {model_name} accuracy: {model_accuracy:.3f}")
        print(f"   📊 Random accuracy: {random_accuracy:.3f}")
        print(f"   📊 Theoretical random: {self.expected_accuracy:.3f}")
        print(f"   📈 Improvement: {improvement_absolute:+.3f} ({improvement_relative:+.1f}%)")
        print(f"   🧮 Statistical significance: p = {p_value:.4f}")
        print(f"   {'✅' if results['is_significant'] else '❌'} Significant: {results['is_significant']}")
        
        return results
    
    def validate_across_datasets(self, datasets_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate random baseline across multiple datasets.
        
        Args:
            datasets_info: List of dicts with 'name', 'n_questions', 'class_distribution'
            
        Returns:
            Cross-dataset validation results
        """
        print(f"🧪 Tester: Validating random baseline across {len(datasets_info)} datasets...")
        
        dataset_results = {}
        overall_stats = {
            'accuracies': [],
            'n_questions': [],
            'dataset_names': []
        }
        
        for dataset_info in datasets_info:
            name = dataset_info['name']
            n_questions = dataset_info['n_questions']
            class_dist = dataset_info.get('class_distribution', None)
            
            print(f"   🔍 Testing dataset: {name} ({n_questions} questions)")
            
            # Generate random baseline for this dataset
            if class_dist:
                # Use actual class distribution as weights
                total = sum(class_dist)
                weights = [count/total for count in class_dist]
            else:
                weights = None
            
            # Run multiple trials
            accuracies = []
            for _ in range(100):  # 100 trials per dataset
                true_answers = np.random.choice(self.n_classes, size=n_questions)
                predictions = self.generate_random_predictions(n_questions, weights)
                accuracy = np.mean(true_answers == predictions)
                accuracies.append(accuracy)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            dataset_results[name] = {
                'n_questions': n_questions,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'class_distribution': class_dist,
                'weights_used': weights
            }
            
            overall_stats['accuracies'].append(mean_acc)
            overall_stats['n_questions'].append(n_questions)
            overall_stats['dataset_names'].append(name)
            
            print(f"      📊 Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        
        # Overall statistics
        overall_mean = np.mean(overall_stats['accuracies'])
        overall_std = np.std(overall_stats['accuracies'])
        
        results = {
            'dataset_results': dataset_results,
            'overall_statistics': {
                'mean_accuracy': overall_mean,
                'std_accuracy': overall_std,
                'n_datasets': len(datasets_info),
                'total_questions': sum(overall_stats['n_questions'])
            },
            'validation_passed': abs(overall_mean - self.expected_accuracy) < 0.02
        }
        
        print(f"   ✅ Cross-dataset validation completed")
        print(f"   📊 Overall mean: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"   {'✅' if results['validation_passed'] else '❌'} Validation passed: {results['validation_passed']}")
        
        return results
    
    def create_random_baseline_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive random baseline validation report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Complete validation report
        """
        print(f"🧪 Tester: Creating comprehensive random baseline report...")
        
        report = {
            'metadata': {
                'validator_version': '1.0.0',
                'n_classes': self.n_classes,
                'expected_accuracy': self.expected_accuracy,
                'random_state': self.random_state,
                'created_at': pd.Timestamp.now().isoformat()
            },
            'theoretical_analysis': self._analyze_theoretical_baseline(),
            'empirical_validation': {},
            'recommendations': self._generate_baseline_recommendations()
        }
        
        # Run empirical validations
        print("   🔬 Running empirical validations...")
        
        # Small dataset validation
        small_validation = self.validate_random_baseline(100, 1000)
        report['empirical_validation']['small_dataset'] = small_validation
        
        # Medium dataset validation  
        medium_validation = self.validate_random_baseline(1000, 500)
        report['empirical_validation']['medium_dataset'] = medium_validation
        
        # Large dataset validation
        large_validation = self.validate_random_baseline(5000, 100)
        report['empirical_validation']['large_dataset'] = large_validation
        
        # Save report if requested
        if output_path:
            import json
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"   📁 Report saved to: {output_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to save report: {e}")
        
        return report
    
    def _analyze_theoretical_baseline(self) -> Dict[str, Any]:
        """Analyze theoretical properties of random baseline."""
        return {
            'expected_accuracy': self.expected_accuracy,
            'variance_per_question': self.expected_accuracy * (1 - self.expected_accuracy),
            'standard_error_formula': 'sqrt(p*(1-p)/n)',
            'confidence_interval_95_formula': 'p ± 1.96*SE',
            'minimum_sample_size_for_significance': int(1.96**2 / (0.05**2 * self.expected_accuracy)),
            'notes': [
                f"For {self.n_classes} classes, random accuracy should be {self.expected_accuracy:.1%}",
                "Larger sample sizes have smaller confidence intervals",
                "Statistical significance requires sufficient sample size and effect size"
            ]
        }
    
    def _generate_baseline_recommendations(self) -> List[str]:
        """Generate recommendations for baseline validation."""
        return [
            f"Always compare model accuracy to {self.expected_accuracy:.1%} random baseline",
            "Use statistical significance testing (p < 0.05) to validate improvements",
            "Report confidence intervals for model accuracy estimates",
            "Validate random baseline empirically before model evaluation",
            "Consider effect size (improvement %) not just statistical significance",
            "Use sufficient sample sizes for reliable statistical testing",
            f"Any model with accuracy < {self.expected_accuracy:.1%} is worse than random"
        ]


def demo_random_baseline_validation():
    """Demonstrate the random baseline validation system."""
    print("🧪 Tester: Demonstrating Random Baseline Validation")
    print("=" * 60)
    
    validator = RandomBaselineValidator()
    
    # 1. Basic baseline validation
    print("\n1️⃣ Basic Random Baseline Validation")
    basic_validation = validator.validate_random_baseline(1000, 500)
    
    # 2. Model comparison example
    print("\n2️⃣ Model vs Random Comparison")
    n_test = 500
    true_answers = np.random.choice(4, size=n_test)
    
    # Simulate a model with 30% accuracy
    model_predictions = np.random.choice(4, size=n_test)
    # Bias some predictions to be correct (simulate 30% accuracy)
    correct_indices = np.random.choice(n_test, size=int(0.30 * n_test), replace=False)
    model_predictions[correct_indices] = true_answers[correct_indices]
    
    comparison = validator.compare_to_random(model_predictions, true_answers, "Sample Model")
    
    # 3. Cross-dataset validation
    print("\n3️⃣ Cross-Dataset Validation")
    datasets_info = [
        {'name': 'Mental Health', 'n_questions': 1000, 'class_distribution': [200, 250, 300, 250]},
        {'name': 'Medical', 'n_questions': 800, 'class_distribution': [200, 200, 200, 200]},
        {'name': 'General', 'n_questions': 1200, 'class_distribution': [300, 300, 300, 300]}
    ]
    
    cross_validation = validator.validate_across_datasets(datasets_info)
    
    # 4. Generate comprehensive report
    print("\n4️⃣ Comprehensive Report Generation")
    report = validator.create_random_baseline_report("random_baseline_report.json")
    
    print(f"\n✅ Random baseline validation completed successfully!")
    
    return validator


if __name__ == "__main__":
    demo_random_baseline_validation()
