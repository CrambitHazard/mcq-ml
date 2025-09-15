"""
Data Loader Module for MCQ Bias Prediction
Standardizes multiple dataset formats into unified schema.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
import uuid
import re


class DataLoader:
    """
    Unified data loader that normalizes different MCQ dataset formats
    into a consistent schema for bias-based prediction.
    """
    
    def __init__(self):
        self.unified_schema = {
            "question_id": str,
            "exam_id": str,
            "question_number": int,
            "question_text": str,
            "options": list,  # list of strings
            "correct_answer": str
        }
    
    def load_dataset(self, file_path: str, dataset_type: str = None) -> pd.DataFrame:
        """
        Load and normalize any supported dataset format.
        
        Args:
            file_path: Path to the dataset file
            dataset_type: Optional dataset type hint ('indiabix', 'medqs', 'mental_health', 'qna')
                         If None, will auto-detect from file structure
        
        Returns:
            DataFrame with unified schema
        """
        # Read the raw data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read {file_path}: {e}")
        
        # Auto-detect dataset type if not provided
        if dataset_type is None:
            dataset_type = self._detect_dataset_type(df)
        
        # Apply appropriate mapper
        mapper_functions = {
            'indiabix': self._map_indiabix,
            'medqs': self._map_medqs,
            'mental_health': self._map_mental_health,
            'qna': self._map_qna
        }
        
        if dataset_type not in mapper_functions:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Transform data
        normalized_df = mapper_functions[dataset_type](df, file_path)
        
        # Validate schema
        self._validate_schema(normalized_df)
        
        return normalized_df
    
    def _detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Auto-detect dataset type based on column names."""
        columns = set(df.columns.str.lower())
        
        # Check for Indiabix pattern
        if {'questext', 'optiona', 'optionb', 'optionc', 'optiond', 'optionans'}.issubset(columns):
            return 'indiabix'
        
        # Check for MedQS pattern
        if {'question', 'opa', 'opb', 'opc', 'opd', 'cop'}.issubset(columns):
            return 'medqs'
        
        # Check for Mental Health pattern
        if {'question', 'option1', 'option2', 'option3', 'option4', 'correct_option'}.issubset(columns):
            return 'mental_health'
        
        # Check for QnA pattern
        if {'input', 'context', 'answers'}.issubset(columns):
            return 'qna'
        
        raise ValueError(f"Unable to detect dataset type. Columns: {list(df.columns)}")
    
    def _map_indiabix(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Map Indiabix CA format to unified schema."""
        exam_id = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data = []
        for idx, row in df.iterrows():
            try:
                # Extract options
                options = [
                    str(row['OptionA']) if pd.notna(row['OptionA']) else "",
                    str(row['OptionB']) if pd.notna(row['OptionB']) else "",
                    str(row['OptionC']) if pd.notna(row['OptionC']) else "",
                    str(row['OptionD']) if pd.notna(row['OptionD']) else ""
                ]
                
                # Convert answer number to option text
                answer_num = int(row['OptionAns']) if pd.notna(row['OptionAns']) else 1
                correct_answer = options[answer_num - 1] if 1 <= answer_num <= 4 else options[0]
                
                normalized_data.append({
                    'question_id': f"{exam_id}_{idx}",
                    'exam_id': exam_id,
                    'question_number': idx + 1,
                    'question_text': str(row['QuesText']) if pd.notna(row['QuesText']) else "",
                    'options': options,
                    'correct_answer': correct_answer
                })
            except Exception as e:
                print(f"Warning: Skipping row {idx} in Indiabix dataset: {e}")
                continue
        
        return pd.DataFrame(normalized_data)
    
    def _map_medqs(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Map MedQS format to unified schema."""
        exam_id = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data = []
        for idx, row in df.iterrows():
            try:
                # Extract options
                options = [
                    str(row['opa']) if pd.notna(row['opa']) else "",
                    str(row['opb']) if pd.notna(row['opb']) else "",
                    str(row['opc']) if pd.notna(row['opc']) else "",
                    str(row['opd']) if pd.notna(row['opd']) else ""
                ]
                
                # Convert answer number to option text
                answer_num = int(row['cop']) if pd.notna(row['cop']) else 0
                correct_answer = options[answer_num] if 0 <= answer_num < len(options) else options[0]
                
                normalized_data.append({
                    'question_id': str(row['id']) if 'id' in row and pd.notna(row['id']) else f"{exam_id}_{idx}",
                    'exam_id': exam_id,
                    'question_number': idx + 1,
                    'question_text': str(row['question']) if pd.notna(row['question']) else "",
                    'options': options,
                    'correct_answer': correct_answer
                })
            except Exception as e:
                print(f"Warning: Skipping row {idx} in MedQS dataset: {e}")
                continue
        
        return pd.DataFrame(normalized_data)
    
    def _map_mental_health(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Map Mental Health format to unified schema."""
        exam_id = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data = []
        for idx, row in df.iterrows():
            try:
                # Extract options
                options = [
                    str(row['option1']) if pd.notna(row['option1']) else "",
                    str(row['option2']) if pd.notna(row['option2']) else "",
                    str(row['option3']) if pd.notna(row['option3']) else "",
                    str(row['option4']) if pd.notna(row['option4']) else ""
                ]
                
                # Correct answer is already the text
                correct_answer = str(row['correct_option']) if pd.notna(row['correct_option']) else options[0]
                
                normalized_data.append({
                    'question_id': str(row['id']) if 'id' in row and pd.notna(row['id']) else f"{exam_id}_{idx}",
                    'exam_id': exam_id,
                    'question_number': idx + 1,
                    'question_text': str(row['question']) if pd.notna(row['question']) else "",
                    'options': options,
                    'correct_answer': correct_answer
                })
            except Exception as e:
                print(f"Warning: Skipping row {idx} in Mental Health dataset: {e}")
                continue
        
        return pd.DataFrame(normalized_data)
    
    def _map_qna(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """
        Map QnA format to unified schema.
        This format requires special handling as it's more complex.
        """
        exam_id = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data = []
        for idx, row in df.iterrows():
            try:
                # QnA format is different - we need to extract or simulate options
                question_text = str(row['Input']) if pd.notna(row['Input']) else ""
                context = str(row['Context']) if pd.notna(row['Context']) else ""
                answers = str(row['Answers']) if pd.notna(row['Answers']) else ""
                
                # Combine question and context
                full_question = f"{question_text}\n\nContext: {context}" if context else question_text
                
                # For QnA format, we'll create synthetic options or extract from answers
                # This is a simplified approach - in practice, you might need more sophisticated parsing
                if '|' in answers:  # Multiple choice format
                    options = [opt.strip() for opt in answers.split('|')]
                    correct_answer = options[0] if options else answers
                else:
                    # Single answer - create synthetic multiple choice
                    options = [
                        answers,
                        "Alternative answer 1",
                        "Alternative answer 2", 
                        "None of the above"
                    ]
                    correct_answer = answers
                
                # Ensure we have exactly 4 options
                while len(options) < 4:
                    options.append(f"Option {len(options) + 1}")
                options = options[:4]  # Limit to 4 options
                
                normalized_data.append({
                    'question_id': f"{exam_id}_{idx}",
                    'exam_id': exam_id,
                    'question_number': idx + 1,
                    'question_text': full_question,
                    'options': options,
                    'correct_answer': correct_answer
                })
            except Exception as e:
                print(f"Warning: Skipping row {idx} in QnA dataset: {e}")
                continue
        
        return pd.DataFrame(normalized_data)
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that the normalized DataFrame matches the unified schema."""
        required_columns = list(self.unified_schema.keys())
        
        # Check all required columns exist
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types and constraints
        for idx, row in df.iterrows():
            if pd.isna(row['question_text']) or row['question_text'] == "":
                print(f"Warning: Empty question text at row {idx}")
            
            if not isinstance(row['options'], list) or len(row['options']) < 2:
                raise ValueError(f"Row {idx}: options must be a list with at least 2 items")
            
            if pd.isna(row['correct_answer']) or row['correct_answer'] == "":
                print(f"Warning: Empty correct answer at row {idx}")
    
    def load_multiple_datasets(self, dataset_configs: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Load and combine multiple datasets.
        
        Args:
            dataset_configs: List of dicts with 'path' and optional 'type' keys
        
        Returns:
            Combined DataFrame with unified schema
        """
        all_data = []
        
        for config in dataset_configs:
            file_path = config['path']
            dataset_type = config.get('type')
            
            try:
                df = self.load_dataset(file_path, dataset_type)
                all_data.append(df)
                print(f"Loaded {len(df)} questions from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets were successfully loaded")
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure unique question IDs across datasets
        combined_df['question_id'] = [f"q_{idx}" for idx in range(len(combined_df))]
        
        return combined_df


def demo_usage():
    """Demonstrate how to use the DataLoader."""
    loader = DataLoader()
    
    # Example dataset configurations - adjust paths for current directory
    dataset_configs = [
        {'path': '../data/indiabix_ca/bix_ca.csv', 'type': 'indiabix'},
        {'path': '../data/medqs/train.csv', 'type': 'medqs'},
        {'path': '../data/mental_health/mhqa.csv', 'type': 'mental_health'},
        {'path': '../data/qna/Train.csv', 'type': 'qna'}
    ]
    
    # Load all datasets
    try:
        combined_data = loader.load_multiple_datasets(dataset_configs)
        print(f"\nSuccessfully loaded {len(combined_data)} total questions")
        print(f"Datasets: {combined_data['exam_id'].unique()}")
        print(f"Sample row:\n{combined_data.iloc[0].to_dict()}")
        return combined_data
    except Exception as e:
        print(f"Error in demo: {e}")
        return None


if __name__ == "__main__":
    demo_usage()
