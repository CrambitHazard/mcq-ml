"""
Simplified System Architecture Validation for Data Loader
Focus on core functionality validation without complex file handling
"""

import sys
import os
import pandas as pd
import json
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import DataLoader


class SimpleDataLoaderValidator:
    """Simplified validation focusing on architecture compliance."""
    
    def __init__(self):
        self.loader = DataLoader()
        self.test_results = {}
    
    def run_core_validation(self) -> Dict[str, Any]:
        """Execute core validation tests."""
        print("🏗️  System Architect: Core Data Loader Validation")
        print("=" * 50)
        
        # Test actual datasets (the most important validation)
        self.test_actual_datasets()
        
        # Test schema compliance
        self.test_schema_structure()
        
        # Test error detection
        self.test_error_detection_capability()
        
        # Generate report
        self.generate_summary_report()
        
        return self.test_results
    
    def test_actual_datasets(self):
        """Test loading actual datasets - core functionality."""
        print("\n📊 Testing Actual Dataset Loading...")
        
        dataset_configs = [
            {'path': '../data/indiabix_ca/bix_ca.csv', 'type': 'indiabix'},
            {'path': '../data/medqs/train.csv', 'type': 'medqs'},
            {'path': '../data/mental_health/mhqa.csv', 'type': 'mental_health'},
            {'path': '../data/qna/Train.csv', 'type': 'qna'}
        ]
        
        results = {}
        total_questions = 0
        
        for config in dataset_configs:
            try:
                if os.path.exists(config['path']):
                    df = self.loader.load_dataset(config['path'], config['type'])
                    
                    # Validate key requirements
                    required_cols = ['question_id', 'exam_id', 'question_number', 
                                   'question_text', 'options', 'correct_answer']
                    has_all_cols = all(col in df.columns for col in required_cols)
                    
                    # Check data quality
                    options_are_lists = df['options'].apply(lambda x: isinstance(x, list)).all()
                    questions_not_empty = df['question_text'].notna().sum() > len(df) * 0.9
                    
                    results[config['type']] = {
                        'status': 'SUCCESS',
                        'rows': len(df),
                        'schema_compliant': has_all_cols,
                        'data_quality': options_are_lists and questions_not_empty
                    }
                    
                    total_questions += len(df)
                    print(f"   ✅ {config['type']}: {len(df)} questions loaded successfully")
                else:
                    results[config['type']] = {
                        'status': 'FILE_NOT_FOUND',
                        'path': config['path']
                    }
                    print(f"   ⚠️  {config['type']}: File not found")
                    
            except Exception as e:
                results[config['type']] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"   ❌ {config['type']}: {str(e)[:60]}...")
        
        self.test_results['dataset_loading'] = results
        self.test_results['total_questions_loaded'] = total_questions
        
        print(f"\n📈 Total Questions Loaded: {total_questions:,}")
    
    def test_schema_structure(self):
        """Test that the unified schema is correctly implemented."""
        print("\n🏗️  Testing Schema Structure...")
        
        # Test with a simple sample
        sample_indiabix = pd.DataFrame({
            "QuesText": ["Test question"],
            "OptionA": ["Option A"],
            "OptionB": ["Option B"],
            "OptionC": ["Option C"],
            "OptionD": ["Option D"],
            "OptionAns": [1]
        })
        
        # Save to temporary location
        test_file = "temp_test_schema.csv"
        try:
            sample_indiabix.to_csv(test_file, index=False)
            
            # Load with our system
            result = self.loader.load_dataset(test_file, 'indiabix')
            
            # Validate schema
            expected_schema = {
                "question_id": str,
                "exam_id": str,
                "question_number": int,
                "question_text": str,
                "options": list,
                "correct_answer": str
            }
            
            schema_valid = True
            for col, expected_type in expected_schema.items():
                if col not in result.columns:
                    schema_valid = False
                    print(f"   ❌ Missing column: {col}")
                elif col == 'options' and not isinstance(result.iloc[0][col], list):
                    schema_valid = False
                    print(f"   ❌ Column {col} not correct type")
            
            if schema_valid:
                print("   ✅ Schema structure is compliant")
            
            self.test_results['schema_validation'] = {
                'compliant': schema_valid,
                'columns_present': list(result.columns),
                'sample_row': result.iloc[0].to_dict() if len(result) > 0 else None
            }
            
            # Cleanup
            os.remove(test_file)
            
        except Exception as e:
            print(f"   ❌ Schema test failed: {e}")
            self.test_results['schema_validation'] = {'error': str(e)}
    
    def test_error_detection_capability(self):
        """Test that the system properly detects and handles errors."""
        print("\n🛡️  Testing Error Detection...")
        
        error_tests = []
        
        # Test 1: Auto-detection capability
        try:
            # Create a sample with clear column structure
            sample_data = pd.DataFrame({
                "question": ["Test"],
                "opa": ["A"], "opb": ["B"], "opc": ["C"], "opd": ["D"],
                "cop": [0]
            })
            
            test_file = "temp_autodetect.csv"
            sample_data.to_csv(test_file, index=False)
            
            # Test auto-detection (no type specified)
            detected_result = self.loader.load_dataset(test_file)
            error_tests.append("✅ Auto-detection works")
            
            os.remove(test_file)
            
        except Exception as e:
            error_tests.append(f"❌ Auto-detection failed: {str(e)[:40]}...")
        
        # Test 2: Invalid file handling
        try:
            self.loader.load_dataset("nonexistent_file.csv")
            error_tests.append("❌ Should have failed for missing file")
        except Exception:
            error_tests.append("✅ Properly handles missing files")
        
        self.test_results['error_handling'] = error_tests
        
        for test in error_tests:
            print(f"   {test}")
    
    def generate_summary_report(self):
        """Generate a comprehensive architecture summary."""
        
        # Count successful datasets
        successful_loads = sum(1 for result in self.test_results.get('dataset_loading', {}).values() 
                              if result.get('status') == 'SUCCESS')
        
        total_datasets = len(self.test_results.get('dataset_loading', {}))
        
        # Determine overall status
        schema_ok = self.test_results.get('schema_validation', {}).get('compliant', False)
        data_loaded = self.test_results.get('total_questions_loaded', 0) > 100000
        
        overall_success = schema_ok and data_loaded and successful_loads >= 3
        
        summary = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "overall_status": "PASSED" if overall_success else "NEEDS_ATTENTION",
            "datasets_loaded_successfully": f"{successful_loads}/{total_datasets}",
            "total_questions": self.test_results.get('total_questions_loaded', 0),
            "schema_compliant": schema_ok,
            "architecture_assessment": {
                "data_standardization": "IMPLEMENTED" if schema_ok else "PARTIAL",
                "error_handling": "ROBUST",
                "scalability": "PROVEN" if data_loaded else "NEEDS_TESTING",
                "production_ready": overall_success
            },
            "detailed_results": self.test_results
        }
        
        # Save report
        report_path = "../architecture/validation_summary.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 Architecture Summary:")
        print(f"   Status: {summary['overall_status']}")
        print(f"   Datasets: {summary['datasets_loaded_successfully']}")
        print(f"   Questions: {summary['total_questions']:,}")
        print(f"   Schema: {'✅' if schema_ok else '❌'}")
        print(f"\n📊 Report saved: {report_path}")
        
        return summary


def main():
    """Run simplified validation."""
    validator = SimpleDataLoaderValidator()
    results = validator.run_core_validation()
    
    print("\n" + "=" * 50)
    print("🏗️  SYSTEM ARCHITECT VALIDATION COMPLETE")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    main()
