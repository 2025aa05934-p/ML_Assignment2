# 🎯 ML Classification Playground - BITS Pilani

---

## a. Problem Statement

The objective of this project is to develop a comprehensive and interactive Machine Learning classification platform that enables the comparison and analysis of multiple classification algorithms. The system is designed to:
- Take a classification dataset as input (via CSV upload or Kaggle integration).
- Automatically preprocess the data (handling missing values and categorical encoding).
- Train and evaluate six different classification models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost).
- Calculate and compare six key performance metrics: Accuracy, ROC-AUC, Precision, Recall, F1-Score, and MCC.
- Provide professional visualizations, including performance charts and confusion matrices, to facilitate model selection and analysis.

This tool is specifically engineered for the **BITS Pilani M.Tech (AIML)** program to benchmark classical and ensemble learning techniques on various structured datasets.

---

## b. Dataset Description

The application is designed to handle structured classification datasets with the following characteristics:
- **Input Type**: Tabular data in CSV format.
- **Features**: Supports both numerical and categorical features (automatically encoded using `LabelEncoder`).
- **Target Variable**: Supports binary and multi-class classification labels (automatically normalized to 0-indexed integers).
- **Processing**: 
    - **Missing Values**: Automatically detects and drops rows with missing data to ensure model stability.
    - **Data Splitting**: Configurable train-test split (typically 80/20) to validate model generalization.
- **Example Data Sources**: 
    - Custom user-uploaded CSV files.
    - Standard academic datasets (Iris, Zoo Animal, Adult Census, etc.) integrated via Kaggle.

---

## c. Models Used (Comparison Table)

The table below summarizes the performance metrics for the six classification models supported by the studio. Note that exact values depend on the specific dataset and test-split configuration used during execution.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| **Logistic Regression** | - | - | - | - | - | - |
| **Decision Tree** | - | - | - | - | - | - |
| **K-Nearest Neighbors** | - | - | - | - | - | - |
| **Naive Bayes** | - | - | - | - | - | - |
| **Random Forest** | - | - | - | - | - | - |
| **XGBoost** | - | - | - | - | - | - |

*(Note: These values are populated dynamically within the application UI after training on your specific dataset.)*

---

## d. Observations

Detailed performance observations for each model in the suite:

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Excellent baseline for linear classification. Fast to train and interprets well for high-dimensional data. Best suited for linearly separable datasets. |
| **Decision Tree** | Captures complex non-linear relationships through recursive partitioning. Highly interpretable but prone to high variance (overfitting) if not pruned. |
| **K-Nearest Neighbors** | Powerful instance-based learner. Performance is sensitive to the choice of 'k' and distance metric. Requires scaled features for optimal results. |
| **Naive Bayes** | Highly efficient probabilistic model based on feature independence. Performs exceptionally well on categorical data even with smaller sample sizes. |
| **Random Forest** | An ensemble of decision trees that reduces variance through bagging. Consistently high accuracy and robust against noise and outliers. |
| **XGBoost** | State-of-the-art gradient boosting implementation. Usually provides the highest accuracy through sequential correction of errors and L1/L2 regularization. |
## 📊 Pre-trained Performance (Benchmark: Credit Card Fraud)

The studio includes pre-trained benchmarks automatically displayed on the landing page. These were generated via `train_models.py` using a truncated test split from the `chetanmittal033/credit-card-fraud-data` Kaggle repository.

| ML Model Name | Accuracy | AUC Score | F1 Score | MCC Score |
|---------------|----------|-----------|----------|-----------|
| **XGBoost** | 0.9975 | 0.8918 | 0.9970 | 0.4354 |
| **Random Forest** | 0.9971 | 0.9948 | 0.9967 | 0.3766 |
| **Log. Regression** | 0.9967 | 0.8016 | 0.9954 | -0.0011 |
---

## 🚀 Key Features & Implementation Details

### Core Functionality
- ✅ **Automated Workflow**: Handles data ingestion, cleaning, training, and evaluation in one click.
- ✅ **Kaggle Integration**: Direct dataset retrieval using `kagglehub` for benchmarking.
- ✅ **Multi-Metric Evaluation**: Goes beyond accuracy to provide a holistic view of model performance.
- ✅ **Visual Analytics**: Interactive confusion matrices and bar charts using `Seaborn` and `Matplotlib`.

### Technical Stack
- **Framework**: Streamlit (Professional Academic Studio Style)
- **Engine**: Scikit-Learn | XGBoost | Joblib
- **Data Handling**: Pandas | NumPy | KaggleHub
- **Environment**: Python 3.13+ (Optimized for macOS)

---

## 🛠️ Installation & Execution

### 1. Unified Setup
Ensure all dependencies are installed in your environment:
```bash
pip install -r requirements.txt
```

### 2. Prepare Pre-trained Weights (Optional but Recommended)
The studio uses a professional offline training engine to prepare model weights before UI launch. This ensures instant performance for benchmark datasets:
```bash
python train_models.py
```
*This generates `trained_models.joblib` containing models, label encoders, and benchmark metrics.*

### 3. Launch Inference Studio
```bash
streamlit run app.py
```

### 4. How to Use
- **Scenario A (Pre-trained)**: Select "Upload CSV" or "Load from Kaggle", then click **🧪 Run Pre-trained** for instant inference results.
- **Scenario B (On-the-fly)**: Click **🚀 Train & Evaluate** to train fresh models on a specific data sample (supports custom datasets).
- **Optimization**: All datasets are automatically truncated to 12,000 rows to ensure optimal performance on consumer-grade machines.


---

**Last Updated**: February 2026
