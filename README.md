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

---

## 🚀 Key Features & Implementation Details

### Core Functionality
- ✅ **Automated Workflow**: Handles data ingestion, cleaning, training, and evaluation in one click.
- ✅ **Kaggle Integration**: Direct dataset retrieval using `kagglehub` for benchmarking.
- ✅ **Multi-Metric Evaluation**: Goes beyond accuracy to provide a holistic view of model performance.
- ✅ **Visual Analytics**: Interactive confusion matrices and bar charts using `Seaborn` and `Matplotlib`.

### Technical Stack
- **Framework**: Streamlit (Web UI)
- **ML Libraries**: Scikit-Learn, XGBoost
- **Data Handling**: Pandas, NumPy
- **Environment**: Python 3.10+ (Optimized for macOS with `libomp` support)

---

## 🛠️ Installation & Execution

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup XGBoost (Mac Users)**:
   ```bash
   brew install libomp
   ```
3. **Run Application**:
   ```bash
   streamlit run app.py
   ```


---

**Last Updated**: February 2026
