# MCQ Bias Prediction Model - System Architecture

## Overview
This document outlines the system architecture for the MCQ Bias Prediction Model, designed to predict correct multiple-choice question answers based on quiz-writing biases rather than domain knowledge.

## Architecture Principles

### 1. Modular Design
- **Data Standardization Layer**: Unified interface for diverse dataset formats
- **Feature Engineering Pipeline**: Extracting bias-based features from question structures
- **Model Training Layer**: Machine learning pipeline for bias pattern recognition
- **Evaluation Framework**: Comprehensive metrics and validation system

### 2. Scalability & Performance
- **Target**: Process 188K+ questions in <2 hours
- **Memory Efficient**: CPU/GPU compatible (6GB VRAM max)
- **Parallel Processing**: Multi-dataset loading capability
- **Incremental Processing**: Support for new dataset types

### 3. Robustness & Error Handling
- **Graceful Degradation**: Continues processing despite malformed data
- **Auto-Detection**: Automatically identifies dataset formats
- **Comprehensive Logging**: Detailed warnings and error reporting
- **Validation Pipeline**: Multi-level data quality checks

## System Components

### Data Standardization Layer (`data_loader.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loader Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  Input Datasets                    │  Unified Schema Output  │
│  ┌─────────────────┐              │  ┌─────────────────────┐│
│  │ Indiabix CA     │─────┐        │  │ question_id: str    ││
│  │ QuesText, etc.  │     │        │  │ exam_id: str        ││
│  └─────────────────┘     │        │  │ question_number: int││
│                           ▼        │  │ question_text: str  ││
│  ┌─────────────────┐  ┌─────────┐ │  │ options: [str]      ││
│  │ MedQS           │─▶│ Mappers │─┼─▶│ correct_answer: str ││
│  │ question, etc.  │  └─────────┘ │  └─────────────────────┘│
│  └─────────────────┘              │                         │
│                                   │  Features:              │
│  ┌─────────────────┐              │  • Auto-detection       │
│  │ Mental Health   │─────┐        │  • Error handling       │
│  │ option1-4, etc. │     │        │  • Schema validation    │
│  └─────────────────┘     │        │  • ID generation        │
│                           │        │                         │
│  ┌─────────────────┐     │        │                         │
│  │ QnA Format      │─────┘        │                         │
│  │ Input/Context   │              │                         │
│  └─────────────────┘              │                         │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features:
- **Format Detection**: Automatic identification of dataset structure
- **Flexible Mapping**: Separate mapper functions for each dataset type
- **Error Recovery**: Continues processing with warnings for malformed rows
- **Batch Processing**: Efficient handling of large datasets

### Feature Engineering Pipeline (Planned)

```
┌─────────────────────────────────────────────────────────────┐
│                Feature Extraction Architecture              │
├─────────────────────────────────────────────────────────────┤
│  Unified Data    │  Feature Types           │  ML Features  │
│  ┌─────────────┐ │  ┌─────────────────────┐ │  ┌──────────┐ │
│  │ Question +  │ │  │ Option Length       │ │  │ Feature  │ │
│  │ 4 Options   │─┼─▶│ • Character count   │─┼─▶│ Matrix   │ │
│  │ + Answer    │ │  │ • Word count        │ │  │ (N x F)  │ │
│  └─────────────┘ │  └─────────────────────┘ │  └──────────┘ │
│                  │                          │               │
│                  │  ┌─────────────────────┐ │               │
│                  │  │ Keyword Detection   │ │               │
│                  │  │ • "all of above"    │ │               │
│                  │  │ • "none"            │ │               │
│                  │  │ • "both A and B"    │ │               │
│                  │  └─────────────────────┘ │               │
│                  │                          │               │
│                  │  ┌─────────────────────┐ │               │
│                  │  │ Numeric Patterns    │ │               │
│                  │  │ • Rank in sorted    │ │               │
│                  │  │ • Min/max/middle    │ │               │
│                  │  │ • Mathematical      │ │               │
│                  │  └─────────────────────┘ │               │
│                  │                          │               │
│                  │  ┌─────────────────────┐ │               │
│                  │  │ Context Features    │ │               │
│                  │  │ • Question position │ │               │
│                  │  │ • Answer patterns   │ │               │
│                  │  │ • Option overlap    │ │               │
│                  │  └─────────────────────┘ │               │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture (Planned)

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  Features        │  Training Process       │  Inference     │
│  ┌─────────────┐ │  ┌─────────────────────┐│  ┌──────────┐  │
│  │ Per-Option  │ │  │ LightGBM/XGBoost    ││  │ Question │  │
│  │ Feature     │─┼─▶│ Binary Classifier   ││  │ Level    │  │
│  │ Vectors     │ │  │ (is_correct: 0/1)   ││  │ Argmax   │  │
│  └─────────────┘ │  └─────────────────────┘│  └──────────┘  │
│                  │                         │               │
│                  │  ┌─────────────────────┐│               │
│                  │  │ Cross-Validation    ││               │
│                  │  │ Grouped by exam_id  ││               │
│                  │  │ (Prevents leakage)  ││               │
│                  │  └─────────────────────┘│               │
│                  │                         │               │
│                  │  ┌─────────────────────┐│               │
│                  │  │ Multi-Dataset       ││               │
│                  │  │ Training Support    ││               │
│                  │  └─────────────────────┘│               │
└─────────────────────────────────────────────────────────────┘
```

## Validation Results

### ✅ **Data Standardization Validation - PASSED**

| Dataset Type | Status | Questions Loaded | Schema Compliant |
|-------------|--------|------------------|------------------|
| Indiabix CA | ✅ SUCCESS | 1,999 | ✅ YES |
| MedQS | ✅ SUCCESS | 182,822 | ✅ YES |
| Mental Health | ✅ SUCCESS | 2,475 | ✅ YES |
| QnA | ✅ SUCCESS | 1,550 | ✅ YES |
| **Total** | **✅ 4/4** | **188,846** | **✅ 100%** |

### ✅ **Architecture Compliance Assessment**

| Component | Status | Details |
|-----------|--------|---------|
| **Unified Schema** | ✅ IMPLEMENTED | All datasets successfully normalized |
| **Error Handling** | ✅ ROBUST | Graceful handling of malformed data |
| **Performance** | ✅ MEETS REQUIREMENTS | 188K questions processed efficiently |
| **Scalability** | ✅ DESIGNED FOR GROWTH | Modular mapper architecture |
| **Auto-Detection** | ✅ FUNCTIONAL | Correctly identifies dataset formats |

### 🔍 **Data Quality Insights**
- **Clean Data**: ~97% of questions have valid structure
- **Warning Handling**: System properly flags empty answers (~3% of records)
- **Format Diversity**: Successfully handles 4 distinct dataset structures
- **Unicode Support**: Handles special characters and international text

## Technical Specifications

### Dependencies
```yaml
Core Libraries:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computing
  - scikit-learn: Machine learning framework
  
Optional ML Libraries:
  - lightgbm: Gradient boosting (recommended)
  - xgboost: Alternative gradient boosting

System Requirements:
  - Python 3.7+
  - Memory: 4GB+ RAM
  - Storage: 500MB for datasets
  - GPU: Optional (6GB VRAM max)
```

### File Structure
```
mcq_bias_model/
├── data_loader.py              # ✅ Implemented
├── test_data_loader_simple.py  # ✅ Validation suite
├── features.py                 # 🔄 Next phase
├── train.py                    # 🔄 Next phase
├── predict.py                  # 🔄 Next phase
└── evaluate.py                 # 🔄 Next phase

architecture/
├── system_architecture.md      # ✅ This document
├── validation_summary.json     # ✅ Test results
└── data_loader_validation_report.json  # ✅ Detailed report
```

## Next Phase Recommendations

### Phase 2: Feature Engineering (45 minutes)
1. **Implement `features.py`** with bias-detection algorithms
2. **Create feature pipeline** integration with data loader
3. **Validate feature extraction** on sample datasets

### Phase 3: Model Development (30 minutes)
1. **Build training pipeline** with LightGBM
2. **Implement cross-validation** grouped by exam_id
3. **Create prediction interface** with argmax selection

### Phase 4: Evaluation Framework (15 minutes)
1. **Comprehensive metrics** (accuracy, top-2, ablation studies)
2. **Performance benchmarking** against random baseline
3. **Final documentation** and usage examples

## Risk Mitigation Strategies

### ✅ **Implemented Mitigations**
- **Data Format Variations**: Flexible mapper architecture handles diverse formats
- **Missing Data**: Graceful degradation with warning systems
- **Scale Requirements**: Proven to handle 188K+ questions efficiently
- **Error Recovery**: Robust exception handling prevents pipeline failures

### 🔄 **Planned Mitigations**
- **Feature Engineering Complexity**: Start with basic features, expand incrementally
- **Model Performance**: Use proven algorithms (LightGBM) for reliability
- **Time Constraints**: Prioritize core functionality over advanced features

## Conclusion

The data standardization layer has been successfully implemented and validated. The architecture provides a solid foundation for the remaining phases, with proven scalability and robustness for handling diverse MCQ datasets.

**Architecture Status: ✅ READY FOR PHASE 2**
