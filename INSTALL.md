# Installation & Setup Guide

Comprehensive installation and setup guide for the MCQ Bias Prediction Model.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependency Management](#dependency-management)
4. [Verification & Testing](#verification--testing)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Development Setup](#development-setup)

## 🖥️ System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+) |
| **Python Version** | 3.8 or higher |
| **RAM** | 2GB minimum, 4GB recommended |
| **Storage** | 1GB free space |
| **CPU** | Any modern x64 processor |

### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| **RAM** | 8GB+ for large datasets |
| **Storage** | SSD with 5GB+ free space |
| **CPU** | Multi-core processor for faster training |

### Python Version Compatibility

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Recommended |
| 3.10 | ✅ Supported | Fully tested |
| 3.11 | ✅ Supported | Latest features |
| 3.12 | ⚠️ Experimental | May work but not tested |

## 🚀 Installation Methods

### Method 1: Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/mcq-bias-model.git
cd mcq-bias-model

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from mcq_bias_model.predict import MCQBiasPredictor; print('✅ Installation successful')"
```

### Method 2: Virtual Environment Installation

```bash
# Create virtual environment
python -m venv mcq_env

# Activate virtual environment
# On Windows:
mcq_env\Scripts\activate
# On macOS/Linux:
source mcq_env/bin/activate

# Clone and install
git clone https://github.com/your-repo/mcq-bias-model.git
cd mcq-bias-model
pip install -r requirements.txt
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n mcq_env python=3.9
conda activate mcq_env

# Clone repository
git clone https://github.com/your-repo/mcq-bias-model.git
cd mcq-bias-model

# Install dependencies via conda (preferred) or pip
conda install pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm

# Or use pip for all dependencies
pip install -r requirements.txt
```

### Method 4: Docker Installation

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up environment
ENV PYTHONPATH=/app
CMD ["python", "-c", "from mcq_bias_model.predict import MCQBiasPredictor; print('MCQ Bias Model ready')"]
```

```bash
# Build and run Docker container
docker build -t mcq-bias-model .
docker run -it mcq-bias-model
```

## 📦 Dependency Management

### Core Dependencies

Create `requirements.txt`:

```txt
# Core ML libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0

# Data visualization (optional)
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
```

### Optional Dependencies

For enhanced functionality:

```txt
# Performance optimization
numba>=0.56.0

# Advanced visualization
plotly>=5.0.0

# Jupyter notebook support
jupyter>=1.0.0
ipywidgets>=7.6.0

# Testing
pytest>=6.0.0
pytest-cov>=3.0.0
```

### Development Dependencies

For development and testing:

```txt
# Code quality
black>=22.0.0
flake8>=4.0.0
mypy>=0.910

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0

# Git hooks
pre-commit>=2.15.0
```

## ✅ Verification & Testing

### Basic Verification

```bash
# Test Python imports
python -c "
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
print('✅ All core dependencies imported successfully')
"
```

### Component Testing

```bash
# Test data loading
python -c "
from mcq_bias_model.data_loader import DataLoader
loader = DataLoader()
print('✅ DataLoader initialized')
"

# Test feature extraction
python -c "
from mcq_bias_model.features import MCQBiasFeatureExtractor
extractor = MCQBiasFeatureExtractor()
print('✅ FeatureExtractor initialized')
"

# Test predictor (requires trained model)
python -c "
from mcq_bias_model.predict import MCQBiasPredictor
try:
    predictor = MCQBiasPredictor()
    print('✅ Predictor loaded successfully')
except FileNotFoundError:
    print('⚠️ Predictor ready (no trained model found)')
"
```

### Full System Test

```bash
# Run comprehensive test
python -c "
from mcq_bias_model.data_loader import DataLoader
from mcq_bias_model.features import MCQBiasFeatureExtractor
from mcq_bias_model.pipeline import MCQBiasPipeline

# Test pipeline
pipeline = MCQBiasPipeline()
print('✅ Pipeline initialized')

# Test with sample data
sample_data = [{
    'question_text': 'Test question?',
    'options': ['A', 'B', 'C', 'D'],
    'question_number': 1,
    'correct_answer': 'B'
}]

extractor = MCQBiasFeatureExtractor()
features = extractor.extract_features(sample_data)
print(f'✅ Feature extraction working: {features.shape[1]} features')
print('🎉 Full system test passed')
"
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file (optional):

```bash
# Model configuration
MCQ_MODEL_PATH=mcq_bias_model.pkl
MCQ_CACHE_DIR=.cache
MCQ_LOG_LEVEL=INFO

# Performance settings
MCQ_BATCH_SIZE=100
MCQ_MAX_WORKERS=4
MCQ_MEMORY_LIMIT=4GB

# Dataset paths (optional)
MCQ_DATA_DIR=./data
MCQ_MENTAL_HEALTH_PATH=./data/mental_health/mhqa.csv
MCQ_MEDQS_PATH=./data/medqs/train.csv
```

### Configuration File

Create `config.json`:

```json
{
  "model": {
    "path": "mcq_bias_model.pkl",
    "cache_features": true,
    "batch_size": 100
  },
  "datasets": {
    "data_dir": "./data",
    "supported_types": ["mental_health", "medqs", "indiabix_ca", "qna"]
  },
  "training": {
    "random_state": 42,
    "cv_folds": 5,
    "test_size": 0.2
  },
  "evaluation": {
    "sample_size": 1000,
    "generate_reports": true,
    "report_dir": "./reports"
  }
}
```

### Directory Structure Setup

```bash
# Create recommended directory structure
mkdir -p data/{mental_health,medqs,indiabix_ca,qna}
mkdir -p models
mkdir -p reports
mkdir -p cache
mkdir -p logs

# Directory layout:
# mcq-bias-model/
# ├── data/                 # Dataset files
# │   ├── mental_health/
# │   ├── medqs/
# │   ├── indiabix_ca/
# │   └── qna/
# ├── models/               # Trained models
# ├── reports/              # Evaluation reports
# ├── cache/               # Feature cache
# ├── logs/                # Log files
# └── mcq_bias_model/      # Source code
```

## 🐛 Troubleshooting

### Common Installation Issues

#### Issue 1: LightGBM Installation Fails

**Error:**
```
ERROR: Failed building wheel for lightgbm
```

**Solutions:**

**Windows:**
```bash
# Install Visual Studio Build Tools first
# Then install lightgbm
pip install lightgbm --install-option=--mpi
```

**macOS:**
```bash
# Install via Homebrew if pip fails
brew install libomp
pip install lightgbm
```

**Linux:**
```bash
# Install build essentials
sudo apt-get install build-essential cmake
pip install lightgbm
```

#### Issue 2: Pandas Version Conflicts

**Error:**
```
ImportError: cannot import name 'DataFrame' from 'pandas'
```

**Solution:**
```bash
# Upgrade pandas
pip install --upgrade pandas>=1.3.0

# Or use conda
conda update pandas
```

#### Issue 3: Memory Issues

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
```python
# Reduce batch size in configuration
MCQ_BATCH_SIZE=50

# Or use sampling for large datasets
sample_size = 1000  # Instead of full dataset
```

#### Issue 4: File Permission Issues

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# Fix permissions
chmod +x mcq-bias-model/
chmod 644 mcq-bias-model/*.py

# Or run with sudo (not recommended)
sudo python setup.py install
```

### Platform-Specific Issues

#### Windows Issues

1. **Long Path Names:**
```bash
# Enable long path support
git config --system core.longpaths true
```

2. **PowerShell Execution Policy:**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS Issues

1. **Xcode Command Line Tools:**
```bash
# Install if missing
xcode-select --install
```

2. **Homebrew Dependencies:**
```bash
# Install required packages
brew install cmake libomp
```

#### Linux Issues

1. **Build Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake python3-devel
```

### Performance Optimization

#### Memory Optimization

```python
# Optimize for low memory systems
import pandas as pd

# Use efficient data types
def optimize_datatypes(df):
    for col in df.select_dtypes(include=['int64']):
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
    return df
```

#### CPU Optimization

```python
# Use all available cores
import multiprocessing
n_jobs = multiprocessing.cpu_count()

# Configure LightGBM for multiple cores
lgb_params = {
    'n_jobs': n_jobs,
    'force_col_wise': True  # Better for small datasets
}
```

## 🔧 Development Setup

### For Contributors

```bash
# Clone development version
git clone -b develop https://github.com/your-repo/mcq-bias-model.git
cd mcq-bias-model

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Check code quality
flake8 mcq_bias_model/
black mcq_bias_model/ --check
mypy mcq_bias_model/
```

### IDE Setup

#### VS Code Configuration

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./mcq_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm Configuration

1. Set interpreter to virtual environment
2. Enable pytest as test runner
3. Configure black as formatter
4. Set source root to project directory

### Testing Setup

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run full test suite
pytest tests/ --cov=mcq_bias_model --cov-report=html

# Run specific tests
pytest tests/test_predict.py -v

# Run with performance profiling
pytest tests/ --profile
```

## 📚 Additional Resources

### Documentation

- [API Reference](mcq_bias_model/API_REFERENCE.md)
- [Usage Examples](mcq_bias_model/USAGE_EXAMPLES.md)
- [Dataset Formats](mcq_bias_model/DATASET_FORMATS.md)

### Community

- **Issues**: [GitHub Issues](https://github.com/your-repo/mcq-bias-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/mcq-bias-model/discussions)
- **Wiki**: [Project Wiki](https://github.com/your-repo/mcq-bias-model/wiki)

### Support

- **Bug Reports**: Use GitHub Issues with bug label
- **Feature Requests**: Use GitHub Issues with enhancement label
- **Questions**: Use GitHub Discussions

---

**Need Help?** If you encounter issues not covered here, please [open an issue](https://github.com/your-repo/mcq-bias-model/issues) with:
1. Your operating system and Python version
2. Complete error message
3. Steps to reproduce the issue
4. Your requirements.txt file
