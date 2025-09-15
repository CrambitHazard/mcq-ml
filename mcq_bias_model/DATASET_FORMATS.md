# Dataset Format Documentation

This document provides comprehensive information about supported dataset formats and requirements for the MCQ Bias Prediction Model.

## 📋 Overview

The MCQ Bias Model supports multiple dataset formats through automatic standardization. All datasets are converted to a unified schema internally, allowing seamless processing of diverse question sources.

## 🎯 Unified Schema (Target Format)

All datasets are standardized to this format:

```python
{
    "question_id": str,        # Unique identifier (auto-generated if missing)
    "exam_id": str,           # Dataset/exam identifier 
    "question_number": int,    # Sequential question number
    "question_text": str,     # The actual question content
    "options": [str],         # List of answer options (typically 4)
    "correct_answer": str     # The correct option text
}
```

### Field Requirements

| Field | Type | Required | Auto-Generated | Description |
|-------|------|----------|----------------|-------------|
| `question_id` | string | ✅ | ✅ | Unique question identifier |
| `exam_id` | string | ✅ | ✅ | Dataset/source identifier |
| `question_number` | integer | ✅ | ✅ | Sequential numbering |
| `question_text` | string | ✅ | ❌ | Question content |
| `options` | list[string] | ✅ | ❌ | Answer choices |
| `correct_answer` | string | ✅ | ❌ | Correct option |

## 📊 Supported Dataset Formats

### 1. Indiabix CA Format

**File Pattern**: `bix_ca.csv`
**Source**: Indiabix Competitive Aptitude questions

**Required Columns**:
```csv
QuesText,OptionA,OptionB,OptionC,OptionD,OptionAns
```

**Example**:
```csv
QuesText,OptionA,OptionB,OptionC,OptionD,OptionAns
"What is the square root of 64?","6","8","10","12","8"
"Which planet is closest to the Sun?","Venus","Earth","Mercury","Mars","Mercury"
```

**Mapping Rules**:
- `QuesText` → `question_text`
- `[OptionA, OptionB, OptionC, OptionD]` → `options` (list)
- `OptionAns` → `correct_answer`
- `exam_id` = "indiabix_ca"

### 2. MedQS Format

**File Pattern**: `train.csv`, `test.csv`, `validation.csv`
**Source**: Medical Questions dataset

**Required Columns**:
```csv
question,opa,opb,opc,opd,cop
```

**Example**:
```csv
question,opa,opb,opc,opd,cop
"Which medication is used for hypertension?","Aspirin","Lisinopril","Insulin","Acetaminophen","Lisinopril"
"What is the normal range for blood pressure?","90/60 - 120/80","140/90 - 160/100","80/50 - 100/70","120/80 - 140/90","90/60 - 120/80"
```

**Mapping Rules**:
- `question` → `question_text`
- `[opa, opb, opc, opd]` → `options` (list)
- `cop` → `correct_answer`
- `exam_id` = "medqs"

### 3. Mental Health Format

**File Pattern**: `mhqa.csv`, `mhqa-b.csv`
**Source**: Mental Health Questions dataset

**Required Columns**:
```csv
question,option1,option2,option3,option4,correct_option
```

**Example**:
```csv
question,option1,option2,option3,option4,correct_option
"What is a common symptom of depression?","Increased energy","Persistent sadness","Improved appetite","Enhanced focus","Persistent sadness"
"Which therapy is effective for anxiety?","Cognitive Behavioral Therapy","Surgery","Medication only","Isolation","Cognitive Behavioral Therapy"
```

**Mapping Rules**:
- `question` → `question_text`
- `[option1, option2, option3, option4]` → `options` (list)
- `correct_option` → `correct_answer`
- `exam_id` = "mental_health"

### 4. QnA Format

**File Pattern**: `Train.csv`, `Test.csv`
**Source**: General Q&A dataset with context

**Required Columns**:
```csv
Input,Context,Answers
```

**Example**:
```csv
Input,Context,Answers
"What causes earthquakes?","Geological processes","Tectonic plate movement;Volcanic activity;Human activity;Magnetic fields"
"How do plants make food?","Biological processes","Photosynthesis;Respiration;Digestion;Absorption"
```

**Mapping Rules**:
- `Input` → `question_text`
- `Context` → Additional context (may be appended to question)
- `Answers` → Split by `;` to create `options` list
- `correct_answer` = First option (index 0)
- `exam_id` = "qna"

**Special Handling**:
- Answers are semicolon-separated in a single field
- First answer is assumed to be correct
- Context may be integrated into question text

## 🔧 Data Loading Implementation

### DataLoader Class Usage

```python
from mcq_bias_model.data_loader import DataLoader

# Initialize loader
loader = DataLoader()

# Load single dataset
data = loader.load_dataset("path/to/dataset.csv", "mental_health")

# Load multiple datasets
datasets = [
    ("data/mental_health/mhqa.csv", "mental_health"),
    ("data/medqs/train.csv", "medqs"),
    ("data/indiabix_ca/bix_ca.csv", "indiabix_ca")
]

combined_data = loader.load_multiple_datasets(datasets)
print(f"Loaded {len(combined_data)} total questions")
```

### Automatic Type Detection

The system can auto-detect dataset types based on column names:

```python
# Auto-detection based on columns
data = loader.load_dataset("unknown_format.csv", "auto")
```

**Detection Rules**:
1. If columns match `[QuesText, OptionA, ...]` → "indiabix_ca"
2. If columns match `[question, opa, opb, ...]` → "medqs"  
3. If columns match `[question, option1, option2, ...]` → "mental_health"
4. If columns match `[Input, Context, Answers]` → "qna"
5. Otherwise → Raise format error

## ✅ Data Validation

### Required Validations

1. **Column Presence**: All required columns must exist
2. **Non-empty Questions**: `question_text` cannot be empty
3. **Valid Options**: Must have 2-6 options per question
4. **Correct Answer**: Must match one of the provided options
5. **Encoding**: Must be UTF-8 compatible

### Validation Example

```python
# Validation is automatic during loading
try:
    data = loader.load_dataset("problematic.csv", "mental_health")
    print("✅ Dataset validation passed")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Missing required column: question` | Column name mismatch | Check column names against format spec |
| `Empty question text at row 5` | Blank question | Remove or fix empty rows |
| `Correct answer not in options` | Answer mismatch | Verify answer matches option text exactly |
| `Invalid option count: 2` | Too few options | Ensure 3-4 options per question |

## 🔄 Data Preprocessing

### Automatic Preprocessing Steps

1. **Text Cleaning**:
   - Strip whitespace from all text fields
   - Normalize unicode characters
   - Remove extra spaces and line breaks

2. **Option Standardization**:
   - Convert all options to strings
   - Ensure consistent list format
   - Handle missing options gracefully

3. **ID Generation**:
   - Auto-generate `question_id` if missing
   - Format: `{dataset_type}_{row_index}_{timestamp}`
   - Auto-generate `exam_id` from filename if missing

4. **Answer Validation**:
   - Verify correct_answer exists in options
   - Handle case sensitivity issues
   - Resolve partial matches when possible

### Preprocessing Configuration

```python
# Custom preprocessing options
loader = DataLoader(
    clean_text=True,           # Clean whitespace and formatting
    normalize_unicode=True,    # Normalize unicode characters
    strict_validation=False,   # Allow minor format deviations
    auto_fix_answers=True      # Attempt to fix answer mismatches
)
```

## 📁 File Organization

### Recommended Directory Structure

```
data/
├── indiabix_ca/
│   └── bix_ca.csv
├── medqs/
│   ├── train.csv
│   ├── test.csv
│   └── validation.csv
├── mental_health/
│   ├── mhqa.csv
│   └── mhqa-b.csv
└── qna/
    ├── Train.csv
    └── Test.csv
```

### File Naming Conventions

- Use descriptive folder names for dataset types
- Keep original filenames when possible
- Use lowercase with underscores for consistency
- Include version numbers for dataset updates

## 🚀 Performance Considerations

### Loading Performance

| Dataset Size | Load Time | Memory Usage |
|-------------|-----------|--------------|
| 1K questions | <1 second | ~10MB |
| 10K questions | ~2 seconds | ~50MB |
| 100K questions | ~15 seconds | ~300MB |

### Optimization Tips

1. **Batch Loading**: Load multiple small files together
2. **Caching**: Enable caching for repeated loads
3. **Sampling**: Use `sample_size` parameter for testing
4. **Memory**: Process large datasets in chunks

```python
# Optimized loading for large datasets
data = loader.load_dataset(
    "large_dataset.csv", 
    "mental_health",
    sample_size=1000,  # Limit for testing
    cache_result=True  # Cache for reuse
)
```

## 🛠️ Custom Format Support

### Adding New Formats

To support a new dataset format:

1. **Define Column Mapping**:
```python
def _load_custom_format(self, file_path: str) -> pd.DataFrame:
    """Load custom format dataset."""
    df = pd.read_csv(file_path)
    
    # Map columns to unified schema
    standardized = []
    for idx, row in df.iterrows():
        question_data = {
            'question_id': f"custom_{idx}",
            'exam_id': 'custom_dataset',
            'question_number': idx + 1,
            'question_text': row['custom_question_col'],
            'options': [row['choice_a'], row['choice_b'], row['choice_c'], row['choice_d']],
            'correct_answer': row['answer_col']
        }
        standardized.append(question_data)
    
    return pd.DataFrame(standardized)
```

2. **Register Format**:
```python
# Add to DataLoader._load_dataset method
elif dataset_type == 'custom_format':
    return self._load_custom_format(file_path)
```

3. **Add Validation Rules**:
```python
# Add validation in _validate_schema method
elif dataset_type == 'custom_format':
    required_cols = ['custom_question_col', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'answer_col']
    # ... validation logic
```

## 🧪 Testing Dataset Formats

### Validation Script

```python
# Test all supported formats
from mcq_bias_model.data_loader import DataLoader

def test_all_formats():
    loader = DataLoader()
    
    test_files = [
        ("data/indiabix_ca/bix_ca.csv", "indiabix_ca"),
        ("data/medqs/train.csv", "medqs"),
        ("data/mental_health/mhqa.csv", "mental_health"),
        ("data/qna/Train.csv", "qna")
    ]
    
    for file_path, dataset_type in test_files:
        try:
            data = loader.load_dataset(file_path, dataset_type)
            print(f"✅ {dataset_type}: {len(data)} questions loaded")
        except Exception as e:
            print(f"❌ {dataset_type}: {e}")

if __name__ == "__main__":
    test_all_formats()
```

## 📞 Troubleshooting

### Common Issues

1. **Encoding Problems**: Ensure files are UTF-8 encoded
2. **Column Name Mismatches**: Check exact column names and spelling
3. **Empty Fields**: Remove rows with missing required data
4. **Answer Mismatches**: Verify correct answers exist in options
5. **Large Files**: Use sampling for initial testing

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Load with debugging
data = loader.load_dataset("problematic.csv", "mental_health")
```

This will provide detailed information about the loading process and any issues encountered.
