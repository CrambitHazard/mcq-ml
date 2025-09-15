"""
Performance Benchmarking and Stress Testing Suite
Tester Implementation - Performance Validation

Tests:
- Speed benchmarks
- Memory usage validation
- Throughput testing
- Scalability analysis
- Resource utilization
"""

import time
import psutil
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add mcq_bias_model to path
sys.path.append(str(Path(__file__).parent.parent / "mcq_bias_model"))

from predict import MCQBiasPredictor


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for MCQ prediction system."""
    
    def __init__(self, model_path: str = None):
        """Initialize performance benchmark suite."""
        self.model_path = model_path or "../mcq_bias_model/mcq_bias_model.pkl"
        self.predictor = None
        self.benchmark_results = {}
        
        # Performance requirements (from 2-hour constraint)
        self.requirements = {
            'single_prediction_max_time': 0.1,  # 100ms per prediction
            'batch_throughput_min': 50,  # 50 questions per second
            'memory_limit_mb': 1000,  # 1GB memory limit
            'accuracy_threshold': 0.30  # 30% accuracy minimum
        }
    
    def setup(self) -> bool:
        """Setup benchmark environment."""
        print("🧪 Tester: Setting up performance benchmark environment...")
        
        if not Path(self.model_path).exists():
            print(f"❌ Model file not found: {self.model_path}")
            return False
        
        try:
            self.predictor = MCQBiasPredictor(self.model_path)
            if not self.predictor.is_loaded:
                print("❌ Failed to load model")
                return False
            
            print("✅ Benchmark environment ready")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def benchmark_single_prediction_speed(self, iterations: int = 100) -> dict:
        """Benchmark single prediction speed."""
        print(f"\n⚡ Benchmarking single prediction speed ({iterations} iterations)...")
        
        test_question = {
            'question_text': 'Performance test question with moderate length text to simulate realistic usage',
            'options': ['Option A with some text', 'Option B with some text', 'Option C with some text', 'Option D with some text'],
            'question_number': 1
        }
        
        times = []
        successful_predictions = 0
        
        # Warmup
        for _ in range(5):
            self.predictor.predict_single(test_question)
        
        # Actual benchmark
        for i in range(iterations):
            start_time = time.perf_counter()
            result = self.predictor.predict_single(test_question)
            end_time = time.perf_counter()
            
            prediction_time = end_time - start_time
            times.append(prediction_time)
            
            if result.get('success', False):
                successful_predictions += 1
        
        results = {
            'iterations': iterations,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / iterations,
            'average_time': np.mean(times),
            'median_time': np.median(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'percentile_95': np.percentile(times, 95),
            'percentile_99': np.percentile(times, 99),
            'meets_requirement': np.mean(times) < self.requirements['single_prediction_max_time']
        }
        
        print(f"   Average time: {results['average_time']:.4f}s")
        print(f"   95th percentile: {results['percentile_95']:.4f}s")
        print(f"   Success rate: {results['success_rate']:.1%}")
        print(f"   Requirement (<{self.requirements['single_prediction_max_time']}s): {'✅ PASS' if results['meets_requirement'] else '❌ FAIL'}")
        
        self.benchmark_results['single_prediction_speed'] = results
        return results
    
    def benchmark_batch_throughput(self, batch_sizes: list = None) -> dict:
        """Benchmark batch processing throughput."""
        if batch_sizes is None:
            batch_sizes = [10, 50, 100, 500, 1000]
        
        print(f"\n📊 Benchmarking batch throughput (sizes: {batch_sizes})...")
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Generate test questions
            test_questions = [
                {
                    'question_text': f'Batch test question {i} with moderate length for realistic testing',
                    'options': [f'Option {i}A', f'Option {i}B', f'Option {i}C', f'Option {i}D'],
                    'question_number': i
                }
                for i in range(batch_size)
            ]
            
            # Benchmark batch processing
            start_time = time.perf_counter()
            batch_results = self.predictor.predict_batch(test_questions, validate_each=False)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = batch_size / processing_time
            
            throughput_results[batch_size] = {
                'processing_time': processing_time,
                'throughput_qps': throughput,
                'success_rate': batch_results['batch_summary']['success_rate'],
                'meets_requirement': throughput >= self.requirements['batch_throughput_min']
            }
            
            print(f"      {throughput:.1f} questions/second (success rate: {batch_results['batch_summary']['success_rate']:.1%})")
        
        # Find optimal batch size
        optimal_batch = max(throughput_results.keys(), 
                           key=lambda x: throughput_results[x]['throughput_qps'])
        
        results = {
            'batch_results': throughput_results,
            'optimal_batch_size': optimal_batch,
            'max_throughput': throughput_results[optimal_batch]['throughput_qps'],
            'meets_requirement': any(r['meets_requirement'] for r in throughput_results.values())
        }
        
        print(f"   Optimal batch size: {optimal_batch} ({results['max_throughput']:.1f} q/s)")
        print(f"   Requirement (>{self.requirements['batch_throughput_min']} q/s): {'✅ PASS' if results['meets_requirement'] else '❌ FAIL'}")
        
        self.benchmark_results['batch_throughput'] = results
        return results
    
    def benchmark_memory_usage(self, max_batch_size: int = 2000) -> dict:
        """Benchmark memory usage patterns."""
        print(f"\n💾 Benchmarking memory usage (up to {max_batch_size} questions)...")
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results = {}
        batch_sizes = [100, 500, 1000, max_batch_size]
        
        for batch_size in batch_sizes:
            print(f"   Testing memory with batch size: {batch_size}")
            
            # Generate large test batch
            test_questions = [
                {
                    'question_text': f'Memory test question {i} ' * 10,  # Longer text for memory testing
                    'options': [f'Detailed option {i}A for memory testing', 
                               f'Detailed option {i}B for memory testing',
                               f'Detailed option {i}C for memory testing', 
                               f'Detailed option {i}D for memory testing'],
                    'question_number': i
                }
                for i in range(batch_size)
            ]
            
            # Measure memory before prediction
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run prediction
            batch_results = self.predictor.predict_batch(test_questions, validate_each=False)
            
            # Measure memory after prediction
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_increase = memory_after - baseline_memory
            
            memory_results[batch_size] = {
                'baseline_memory_mb': baseline_memory,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_increase,
                'memory_per_question_kb': (memory_increase * 1024) / batch_size if batch_size > 0 else 0,
                'success_rate': batch_results['batch_summary']['success_rate'],
                'meets_requirement': memory_after < self.requirements['memory_limit_mb']
            }
            
            print(f"      Memory increase: {memory_increase:.1f}MB ({memory_results[batch_size]['memory_per_question_kb']:.2f}KB per question)")
        
        max_memory = max(r['memory_after_mb'] for r in memory_results.values())
        
        results = {
            'baseline_memory_mb': baseline_memory,
            'memory_by_batch_size': memory_results,
            'max_memory_usage_mb': max_memory,
            'meets_requirement': max_memory < self.requirements['memory_limit_mb']
        }
        
        print(f"   Max memory usage: {max_memory:.1f}MB")
        print(f"   Requirement (<{self.requirements['memory_limit_mb']}MB): {'✅ PASS' if results['meets_requirement'] else '❌ FAIL'}")
        
        self.benchmark_results['memory_usage'] = results
        return results
    
    def benchmark_accuracy_consistency(self, test_size: int = 500) -> dict:
        """Benchmark accuracy and prediction consistency."""
        print(f"\n🎯 Benchmarking accuracy consistency ({test_size} predictions)...")
        
        # Generate diverse test questions
        test_questions = []
        for i in range(test_size):
            question_types = [
                ('What is the capital of {}?', ['Paris', 'London', 'Berlin', 'Madrid']),
                ('Which is larger: {} or {}?', ['Option A', 'Option B', 'Both equal', 'Cannot determine']),
                ('The result of {} + {} is:', ['10', '15', '20', '25']),
                ('Which statement is correct?', ['Statement A', 'Statement B', 'All of the above', 'None of the above'])
            ]
            
            question_type, options = question_types[i % len(question_types)]
            test_questions.append({
                'question_text': question_type.format(f'item{i}', f'item{i+1}'),
                'options': options,
                'question_number': i + 1
            })
        
        # Run batch prediction
        start_time = time.perf_counter()
        batch_results = self.predictor.predict_batch(test_questions, validate_each=True)
        end_time = time.perf_counter()
        
        # Analyze results
        successful_predictions = [p for p in batch_results['predictions'] if p.get('success', False)]
        
        if successful_predictions:
            confidences = [p['max_confidence'] for p in successful_predictions]
            option_distribution = {}
            
            for p in successful_predictions:
                option = p['predicted_option_letter']
                option_distribution[option] = option_distribution.get(option, 0) + 1
            
            results = {
                'total_predictions': test_size,
                'successful_predictions': len(successful_predictions),
                'success_rate': len(successful_predictions) / test_size,
                'average_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'option_distribution': option_distribution,
                'processing_time': end_time - start_time,
                'throughput_qps': test_size / (end_time - start_time),
                'meets_requirement': len(successful_predictions) / test_size >= 0.95  # 95% success rate
            }
        else:
            results = {
                'total_predictions': test_size,
                'successful_predictions': 0,
                'success_rate': 0,
                'meets_requirement': False
            }
        
        print(f"   Success rate: {results['success_rate']:.1%}")
        if 'average_confidence' in results:
            print(f"   Average confidence: {results['average_confidence']:.3f}")
            print(f"   Option distribution: {results['option_distribution']}")
        print(f"   Requirement (>95% success): {'✅ PASS' if results['meets_requirement'] else '❌ FAIL'}")
        
        self.benchmark_results['accuracy_consistency'] = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        print("\n📋 Generating Performance Report...")
        
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            },
            'requirements': self.requirements,
            'results': self.benchmark_results,
            'overall_assessment': self._assess_overall_performance()
        }
        
        # Save report
        report_path = Path(__file__).parent / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved: {report_path}")
        
        # Print summary
        self._print_performance_summary()
        
        return str(report_path)
    
    def _assess_overall_performance(self) -> dict:
        """Assess overall performance against requirements."""
        assessments = {}
        
        # Check each benchmark against requirements
        if 'single_prediction_speed' in self.benchmark_results:
            assessments['single_prediction'] = self.benchmark_results['single_prediction_speed']['meets_requirement']
        
        if 'batch_throughput' in self.benchmark_results:
            assessments['batch_throughput'] = self.benchmark_results['batch_throughput']['meets_requirement']
        
        if 'memory_usage' in self.benchmark_results:
            assessments['memory_usage'] = self.benchmark_results['memory_usage']['meets_requirement']
        
        if 'accuracy_consistency' in self.benchmark_results:
            assessments['accuracy_consistency'] = self.benchmark_results['accuracy_consistency']['meets_requirement']
        
        overall_pass = all(assessments.values()) if assessments else False
        
        return {
            'individual_assessments': assessments,
            'overall_pass': overall_pass,
            'performance_grade': 'EXCELLENT' if overall_pass else 'NEEDS_IMPROVEMENT'
        }
    
    def _print_performance_summary(self):
        """Print performance summary to console."""
        print("\n" + "=" * 60)
        print("🧪 TESTER: PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        
        assessment = self._assess_overall_performance()
        
        print(f"Overall Performance: {'✅ PASS' if assessment['overall_pass'] else '❌ FAIL'}")
        print(f"Performance Grade: {assessment['performance_grade']}")
        
        print(f"\nDetailed Results:")
        for test_name, passed in assessment['individual_assessments'].items():
            status = '✅ PASS' if passed else '❌ FAIL'
            print(f"  {test_name}: {status}")
        
        # Key metrics
        if 'single_prediction_speed' in self.benchmark_results:
            avg_time = self.benchmark_results['single_prediction_speed']['average_time']
            print(f"\nKey Metrics:")
            print(f"  Single prediction: {avg_time:.4f}s average")
        
        if 'batch_throughput' in self.benchmark_results:
            max_throughput = self.benchmark_results['batch_throughput']['max_throughput']
            print(f"  Max throughput: {max_throughput:.1f} questions/second")
        
        if 'memory_usage' in self.benchmark_results:
            max_memory = self.benchmark_results['memory_usage']['max_memory_usage_mb']
            print(f"  Peak memory: {max_memory:.1f}MB")
        
        print(f"\n🎯 System ready for production: {'YES' if assessment['overall_pass'] else 'NO'}")


def run_performance_benchmarks():
    """Run complete performance benchmark suite."""
    print("🧪 Tester: MCQ Bias Prediction - Performance Benchmark Suite")
    print("=" * 70)
    
    benchmark = PerformanceBenchmark()
    
    if not benchmark.setup():
        print("❌ Benchmark setup failed")
        return False
    
    try:
        # Run all benchmarks
        benchmark.benchmark_single_prediction_speed(iterations=200)
        benchmark.benchmark_batch_throughput(batch_sizes=[10, 50, 100, 500, 1000])
        benchmark.benchmark_memory_usage(max_batch_size=2000)
        benchmark.benchmark_accuracy_consistency(test_size=500)
        
        # Generate report
        report_path = benchmark.generate_performance_report()
        
        print(f"\n✅ Performance benchmarking completed successfully")
        print(f"📊 Detailed report: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return False


if __name__ == "__main__":
    success = run_performance_benchmarks()
    sys.exit(0 if success else 1)
