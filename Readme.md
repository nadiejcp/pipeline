# 🚗 US Accidents Severity Prediction Pipeline

A robust, configurable machine learning pipeline designed to predict the severity of traffic accidents using the [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

This project features a modular architecture with support for multiple model types (Logistic Regression, Random Forest, MLP), automated feature engineering, and comprehensive experiment tracking.

---

## 🚀 Quick Start (Zero-Setup)

This project is designed to be **zero-setup** on Windows.

### Requirements
* **Windows**
* **Python 3.12+** installed and available in PATH
* **Data**: Download `US_Accidents_March23.csv` and place it in `data/raw/`.

### How to Run
Simply double-click or execute:

```bat
start.bat
```

The script automatically:
1. Creates a Python virtual environment (`.venv`) if one doesn't exist.
2. Installs dependencies from `requirements.txt`.
3. Runs the pipeline (`run.py`).

---

## 📂 Project Structure

```
pipeline/
├── config/             # Configuration files
│   └── config.yaml     # Main pipeline configuration (models, features, paths)
├── core/               # Core data processing modules
│   ├── data_loader.py      # Loads and splits raw data
│   ├── preprocessor.py     # Handles missing values and scaling
│   ├── feature_engineer.py # transform cyclic temporal features (sin/cos)
│   └── evaluator.py        # Computes metrics (Accuracy, F1)
├── models/             # Model implementations
│   ├── base.py         # Abstract base class for all models
│   ├── factory.py      # Model factory pattern for instantiation
│   ├── baseline.py     # Logistic Regression
│   ├── random_forest.py# Random Forest Classifier
│   └── mlp.py          # Multi-Layer Perceptron (Neural Network)
├── data/               # Data storage
│   ├── raw/            # Place raw CSV here
│   └── processed/      # Cached processed data (.joblib)
├── artifacts/          # Outputs (Saved models, metrics, preprocessors)
├── run.py              # Application entry point
├── pipeline_controller.py # Main orchestration logic
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Configuration

The pipeline is fully data-driven via `config/config.yaml`.

### Data & Features
You can configure which columns to use as features without changing code:

```yaml
features:
  temporal: Start_Time
  temporal_features: [minute, hour, dayofweek, month] # These get cyclic encoding
  numerical: [Temperature(F), Visibility(mi), ...]
  categorical: [Weather_Condition, Junction, ...]
  target: Severity
```

### Models
Adjust hyperparameters or enable/disable models:

```yaml
models:
  baseline:
    type: logistic_regression
  medium:
    type: random_forest
    n_estimators: 200
  advanced:
    type: mlp
    hidden_layers: [128, 64]
    dropout: 0.3
```

---

## 🛠️ Pipeline Details

### 1. Data Loader
* Loads the raw CSV dataset.
* Splits data into **Train (70%)**, **Validation (10%)**, and **Test (20%)** sets (configurable).

### 2. Feature Engineering
* **Temporal Features**: Converts cyclic time features (e.g., Hour 0-23, Month 1-12) into **Sine/Cosine** components to preserve their cyclic nature.
* **Selection**: Filters datasets to only include configured numerical and categorical columns.

### 3. Preprocessing
* **Imputation**: Handles missing values (e.g., mean/median for numerical, mode for categorical).
* **Scaling**: Standardizes numerical features.
* **Encoding**: One-hot encodes categorical variables.

### 4. Training & Evaluation
* Trains all models defined in `config.yaml`.
* Evaluates using **Accuracy**, **Macro F1-Score**, and **Weighted F1-Score**.
* **Caching**: Automatically caches processed data to `data/processed/` to speed up subsequent runs. To force a full rebuild, delete the cached files.

---

## 🧩 Extending the Pipeline

### Adding a New Model
1. Create a new class in `models/` that inherits from `BaseModel`.
2. Implement `fit`, `predict`, and `predict_proba`.
3. Register the new model type in `models/factory.py`.
4. Add the model configuration to `config.yaml`.

### output
Results are saved in the `artifacts/` directory:
* **Models**: `artifacts/models/*.joblib`
* **Metrics**: `artifacts/metrics/*.json`
* **Preprocessors**: `artifacts/*.joblib`
