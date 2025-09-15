# API Reference Documentation

Complete API reference for the MCQ Bias Prediction Model components.

## 📚 Table of Contents

1. [MCQBiasPredictor](#mcqbiasPredictor)
2. [MCQBiasTrainer](#mcqbiastrainer)
3. [MCQBiasEvaluator](#mcqbiasevaluator)
4. [DataLoader](#dataloader)
5. [MCQBiasFeatureExtractor](#mcqbiasfeatureextractor)
6. [MCQBiasPipeline](#mcqbiaspipeline)
7. [Data Types](#data-types)
8. [Error Classes](#error-classes)

---

## MCQBiasPredictor

**Module**: `mcq_bias_model.predict`

Main inference class for predicting MCQ answers using trained bias models.

### Constructor

```python
MCQBiasPredictor(model_path: str = "mcq_bias_model.pkl")
```

**Parameters:**
- `model_path` (str, optional): Path to trained model file. Defaults to "mcq_bias_model.pkl"

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `ValueError`: If model file is corrupted

**Example:**
```python
predictor = MCQBiasPredictor()
predictor = MCQBiasPredictor("custom_model.pkl")
```

### Properties

#### `is_loaded`
```python
@property
is_loaded -> bool
```
Returns whether the model is successfully loaded.

#### `model_path`
```python
@property
model_path -> str
```
Returns the path to the loaded model file.

### Methods

#### `predict_single`
```python
predict_single(question: Dict[str, Any]) -> Dict[str, Any]
```

Predict answer for a single MCQ question.

**Parameters:**
- `question` (dict): Question dictionary with required keys:
  - `question_text` (str): The question content
  - `options` (List[str]): List of answer options (2-6 options)

**Returns:**
- `dict`: Prediction result containing:
  - `predicted_option` (str): The predicted answer text
  - `predicted_index` (int): Index of predicted option (0-based)
  - `confidence` (float): Prediction confidence (0.0-1.0)
  - `success` (bool): Whether prediction succeeded
  - `processing_time` (float): Time taken in seconds
  - `feature_count` (int): Number of features used
  - `error` (str, optional): Error message if `success=False`

**Example:**
```python
question = {
    "question_text": "What is 2+2?",
    "options": ["3", "4", "5", "6"]
}
result = predictor.predict_single(question)
# {'predicted_option': '4', 'predicted_index': 1, 'confidence': 0.68, 'success': True, ...}
```

#### `predict_batch`
```python
predict_batch(questions: List[Dict[str, Any]], 
              batch_size: int = 100,
              show_progress: bool = True) -> List[Dict[str, Any]]
```

Predict answers for multiple questions in batches.

**Parameters:**
- `questions` (List[dict]): List of question dictionaries
- `batch_size` (int, optional): Questions per batch. Defaults to 100
- `show_progress` (bool, optional): Show progress indicators. Defaults to True

**Returns:**
- `List[dict]`: List of prediction results (same format as `predict_single`)

**Example:**
```python
questions = [
    {"question_text": "Q1?", "options": ["A", "B", "C", "D"]},
    {"question_text": "Q2?", "options": ["X", "Y", "Z"]}
]
results = predictor.predict_batch(questions)
```

#### `predict_from_file`
```python
predict_from_file(file_path: str, 
                  dataset_type: str,
                  sample_size: Optional[int] = None,
                  save_results: bool = False) -> List[Dict[str, Any]]
```

Load questions from file and predict answers.

**Parameters:**
- `file_path` (str): Path to dataset file
- `dataset_type` (str): Dataset format ("mental_health", "medqs", "indiabix_ca", "qna")
- `sample_size` (int, optional): Limit number of questions
- `save_results` (bool, optional): Save results to JSON file

**Returns:**
- `List[dict]`: Prediction results

**Example:**
```python
results = predictor.predict_from_file(
    "../data/test.csv", 
    "mental_health",
    sample_size=100,
    save_results=True
)
```

#### `validate_accuracy`
```python
validate_accuracy(data: Union[pd.DataFrame, str],
                  dataset_type: Optional[str] = None) -> Dict[str, Any]
```

Validate model accuracy on labeled dataset.

**Parameters:**
- `data` (DataFrame or str): Either loaded DataFrame or file path
- `dataset_type` (str, optional): Required if `data` is file path

**Returns:**
- `dict`: Validation results containing:
  - `accuracy_metrics` (dict): Overall accuracy, top-2 accuracy, improvement
  - `prediction_stats` (dict): Success rate, processing time, etc.
  - `detailed_results` (dict, optional): Per-question results

**Example:**
```python
validation = predictor.validate_accuracy("test.csv", "mental_health")
print(f"Accuracy: {validation['accuracy_metrics']['overall_accuracy']:.1%}")
```

---

## MCQBiasTrainer

**Module**: `mcq_bias_model.train`

Training class for building MCQ bias prediction models.

### Constructor

```python
MCQBiasTrainer(model_params: Optional[Dict] = None)
```

**Parameters:**
- `model_params` (dict, optional): LightGBM parameters. Uses defaults if None

**Example:**
```python
trainer = MCQBiasTrainer()
trainer = MCQBiasTrainer({'n_estimators': 200, 'max_depth': 8})
```

### Methods

#### `train`
```python
train(dataset_paths: List[str],
      validate: bool = True,
      test_size: float = 0.2,
      random_state: int = 42) -> Dict[str, Any]
```

Train the MCQ bias model on specified datasets.

**Parameters:**
- `dataset_paths` (List[str]): Paths to training datasets
- `validate` (bool, optional): Perform cross-validation. Defaults to True
- `test_size` (float, optional): Test set proportion. Defaults to 0.2
- `random_state` (int, optional): Random seed. Defaults to 42

**Returns:**
- `dict`: Training results containing:
  - `train_accuracy` (float): Training set accuracy
  - `validation_accuracy` (float): CV accuracy (if validate=True)
  - `test_accuracy` (float): Test set accuracy
  - `training_time` (float): Time taken in seconds
  - `total_questions` (int): Number of questions trained on
  - `total_option_examples` (int): Number of option examples
  - `feature_importance` (List[Tuple]): Feature names and importance scores

**Example:**
```python
results = trainer.train([
    "../data/train1.csv",
    "../data/train2.csv"
], validate=True)
```

#### `cross_validate`
```python
cross_validate(X: np.ndarray, 
               y: np.ndarray,
               groups: np.ndarray,
               cv_folds: int = 5) -> Dict[str, Any]
```

Perform grouped cross-validation.

**Parameters:**
- `X` (ndarray): Feature matrix
- `y` (ndarray): Target labels
- `groups` (ndarray): Group identifiers for GroupKFold
- `cv_folds` (int, optional): Number of CV folds. Defaults to 5

**Returns:**
- `dict`: CV results with mean/std accuracy and fold scores

#### `predict`
```python
predict(questions: List[Dict[str, Any]]) -> List[int]
```

Make predictions using trained model.

**Parameters:**
- `questions` (List[dict]): Questions to predict

**Returns:**
- `List[int]`: Predicted option indices

#### `evaluate_model`
```python
evaluate_model(X_test: np.ndarray, 
               y_test: np.ndarray,
               groups_test: np.ndarray) -> Dict[str, float]
```

Evaluate model performance on test data.

**Returns:**
- `dict`: Metrics including accuracy, top-2 accuracy, etc.

#### `save_model`
```python
save_model(file_path: str) -> None
```

Save trained model to file.

#### `load_model`
```python
load_model(file_path: str) -> None
```

Load model from file.

#### `get_feature_importance`
```python
get_feature_importance(top_n: int = 20) -> Dict[str, float]
```

Get feature importance scores.

**Parameters:**
- `top_n` (int, optional): Number of top features. Defaults to 20

**Returns:**
- `dict`: Feature names mapped to importance scores

---

## MCQBiasEvaluator

**Module**: `mcq_bias_model.evaluate`

Comprehensive evaluation system for analyzing model performance.

### Constructor

```python
MCQBiasEvaluator(model_path: str = "mcq_bias_model.pkl")
```

### Methods

#### `evaluate_overall_accuracy`
```python
evaluate_overall_accuracy(test_datasets: List[Dict[str, str]],
                          sample_size: Optional[int] = None) -> Dict[str, Any]
```

Evaluate overall model accuracy across multiple datasets.

**Parameters:**
- `test_datasets` (List[dict]): List of dataset configs with 'path' and 'type' keys
- `sample_size` (int, optional): Limit questions per dataset

**Returns:**
- `dict`: Overall accuracy results with per-dataset and combined metrics

**Example:**
```python
datasets = [
    {'path': 'test1.csv', 'type': 'mental_health'},
    {'path': 'test2.csv', 'type': 'medqs'}
]
results = evaluator.evaluate_overall_accuracy(datasets, sample_size=500)
```

#### `evaluate_accuracy_by_question_type`
```python
evaluate_accuracy_by_question_type(test_datasets: List[Dict[str, str]]) -> Dict[str, Any]
```

Evaluate accuracy by different question characteristics.

**Returns:**
- `dict`: Accuracy breakdown by dataset type, question characteristics, and option patterns

#### `evaluate_feature_ablation`
```python
evaluate_feature_ablation(test_data_path: str, 
                          dataset_type: str) -> Dict[str, Any]
```

Perform ablation study by removing feature groups.

**Parameters:**
- `test_data_path` (str): Path to test dataset
- `dataset_type` (str): Dataset format type

**Returns:**
- `dict`: Feature importance and ablation analysis results

#### `generate_comprehensive_report`
```python
generate_comprehensive_report(output_path: Optional[str] = None) -> Dict[str, Any]
```

Generate complete evaluation report.

**Parameters:**
- `output_path` (str, optional): Path to save JSON report

**Returns:**
- `dict`: Complete evaluation report with findings and recommendations

---

## DataLoader

**Module**: `mcq_bias_model.data_loader`

Data loading and standardization component.

### Constructor

```python
DataLoader()
```

### Methods

#### `load_dataset`
```python
load_dataset(file_path: str, 
             dataset_type: str) -> pd.DataFrame
```

Load and standardize a single dataset.

**Parameters:**
- `file_path` (str): Path to dataset file
- `dataset_type` (str): Format type ("mental_health", "medqs", "indiabix_ca", "qna")

**Returns:**
- `DataFrame`: Standardized dataset with unified schema

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If dataset format is invalid

#### `load_multiple_datasets`
```python
load_multiple_datasets(dataset_configs: List[Tuple[str, str]]) -> pd.DataFrame
```

Load and combine multiple datasets.

**Parameters:**
- `dataset_configs` (List[Tuple]): List of (file_path, dataset_type) tuples

**Returns:**
- `DataFrame`: Combined standardized dataset

#### `get_dataset_info`
```python
get_dataset_info(file_path: str, 
                 dataset_type: str) -> Dict[str, Any]
```

Get information about dataset without loading.

**Returns:**
- `dict`: Dataset metadata (row count, columns, sample data)

---

## MCQBiasFeatureExtractor

**Module**: `mcq_bias_model.features`

Feature extraction component for bias analysis.

### Constructor

```python
MCQBiasFeatureExtractor()
```

### Methods

#### `extract_features`
```python
extract_features(questions_data: List[Dict[str, Any]]) -> pd.DataFrame
```

Extract bias features from questions.

**Parameters:**
- `questions_data` (List[dict]): Question data with unified schema

**Returns:**
- `DataFrame`: Feature matrix with 86 bias features per question

#### `get_feature_names`
```python
get_feature_names() -> List[str]
```

Get list of all feature names.

**Returns:**
- `List[str]`: 86 feature names

#### `get_feature_categories`
```python
get_feature_categories() -> Dict[str, List[str]]
```

Get features grouped by category.

**Returns:**
- `dict`: Feature categories mapped to feature name lists

---

## MCQBiasPipeline

**Module**: `mcq_bias_model.pipeline`

Integrated ML pipeline combining data loading and feature extraction.

### Constructor

```python
MCQBiasPipeline(cache_features: bool = False)
```

**Parameters:**
- `cache_features` (bool, optional): Enable feature caching. Defaults to False

### Methods

#### `load_datasets`
```python
load_datasets(dataset_configs: List[Tuple[str, str]]) -> pd.DataFrame
```

Load and standardize datasets.

#### `extract_features`
```python
extract_features(data: pd.DataFrame) -> pd.DataFrame
```

Extract features with optional caching.

#### `generate_feature_matrix`
```python
generate_feature_matrix(features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Convert features to ML-ready format.

**Returns:**
- `Tuple`: (X, y, groups) for machine learning

#### `create_ml_dataset`
```python
create_ml_dataset(dataset_configs: List[Tuple[str, str]],
                  test_size: float = 0.2,
                  val_size: float = 0.1) -> Dict[str, Any]
```

Create complete train/val/test splits.

**Returns:**
- `dict`: ML-ready datasets with splits

---

## Data Types

### Question Dictionary

```python
QuestionDict = {
    "question_text": str,      # Required: Question content
    "options": List[str],      # Required: 2-6 answer options
    "question_number": int,    # Optional: Question index
    "question_id": str,        # Optional: Unique identifier
    "exam_id": str,           # Optional: Dataset identifier
    "correct_answer": str      # Optional: For validation only
}
```

### Prediction Result

```python
PredictionResult = {
    "predicted_option": str,       # Predicted answer text
    "predicted_index": int,        # Option index (0-based)
    "confidence": float,           # Confidence score (0.0-1.0)
    "success": bool,              # Whether prediction succeeded
    "processing_time": float,      # Time taken in seconds
    "feature_count": int,         # Number of features used
    "error": Optional[str]        # Error message if failed
}
```

### Training Result

```python
TrainingResult = {
    "train_accuracy": float,           # Training accuracy
    "validation_accuracy": float,     # CV accuracy
    "test_accuracy": float,          # Test accuracy
    "training_time": float,          # Training time in seconds
    "total_questions": int,          # Number of questions
    "total_option_examples": int,    # Number of option examples
    "feature_importance": List[Tuple[str, float]]  # Feature importance
}
```

### Evaluation Result

```python
EvaluationResult = {
    "accuracy_metrics": {
        "overall_accuracy": float,     # Question-level accuracy
        "top2_accuracy": float,        # Top-2 accuracy
        "improvement_over_random": float  # % improvement over 25%
    },
    "prediction_stats": {
        "total_questions": int,
        "prediction_success_rate": float,
        "average_processing_time": float
    },
    "detailed_results": Optional[Dict]  # Per-question results
}
```

## Error Classes

### ValidationError

```python
class ValidationError(ValueError):
    """Raised when data validation fails."""
    pass
```

### PredictionError

```python
class PredictionError(RuntimeError):
    """Raised when prediction fails."""
    pass
```

### ModelError

```python
class ModelError(RuntimeError):
    """Raised when model loading/training fails."""
    pass
```

## Constants

### Dataset Types

```python
SUPPORTED_DATASET_TYPES = [
    "mental_health",
    "medqs", 
    "indiabix_ca",
    "qna"
]
```

### Feature Categories

```python
FEATURE_CATEGORIES = {
    "length_features": 17,      # Character/word counts
    "keyword_features": 25,     # Bias keywords
    "numeric_features": 10,     # Numeric patterns
    "context_features": 7,      # Question context
    "rule_based_features": 5,   # Rule-based scores
    "overlap_features": 22      # Option overlap
}
```

### Performance Limits

```python
PERFORMANCE_LIMITS = {
    "max_questions_per_batch": 1000,
    "max_options_per_question": 6,
    "min_options_per_question": 2,
    "max_question_length": 10000,
    "max_option_length": 1000
}
```

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Compatibility**: Python 3.8+
