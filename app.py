import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import kagglehub
except ImportError:
    kagglehub = None

# ============================================================
# CUSTOM STYLING - Distinctive Visual Identity
# ============================================================
st.set_page_config(
    page_title="ML Classification Playground",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with unique aesthetic - Professional Academic Studio Style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700;800&family=Source+Code+Pro:wght@400;600&display=swap');
    
    /* Main container styling */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Force visibility of all text elements */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #334155 !important;
    }

    /* Exceptions for Header */
    .stApp .custom-header h1, .stApp .custom-header .bits-info, .stApp .custom-header .assignment-tag {
        color: #ffffff !important;
    }
    
    /* Optimized Header - Professional Compact Style */
    .custom-header {
        background: #0f172a;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
        border-bottom: 3px solid #60a5fa;
    }

    .stApp .custom-header h1 {
        font-family: 'Crimson Pro', serif;
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .stApp .custom-header .bits-info {
        font-family: 'Source Code Pro', monospace;
        color: #ffffff !important;
        opacity: 0.9;
        font-size: 0.8rem;
        margin-top: 0.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stApp .custom-header .assignment-tag {
        display: inline-block;
        color: #ffffff !important;
        opacity: 0.7;
        font-size: 0.65rem;
        margin-top: 0.5rem;
        padding: 0.1rem 0.8rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Metric cards with Modern Slate style */
    .metric-card {
        background: #ffffff;
        padding: 2rem 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3b82f6;
    }
    
    .metric-label {
        font-family: 'Source Code Pro', monospace;
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-family: 'Crimson Pro', serif;
        font-size: 2.2rem;
        color: #1e293b;
        font-weight: 700;
        line-height: 1;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Crimson Pro', serif;
        color: #1e293b;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Info boxes - Problem Statement/Dataset Description */
    .content-box {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }

    .content-box h3 {
        color: #1e293b !important;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #ffffff;
    }

    /* Force light text for sidebar labels and descriptions */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Force visibility for dataframes and tables */
    .stTable, .stDataFrame, .stMarkdown, .stText {
        color: #334155 !important;
    }
    
    .stAlert p {
        color: #ffffff !important;
    }
    
    /* Success/Error/Info messages with professional dark backgrounds for high contrast */
    [data-testid="stNotificationContent"] p {
        color: #ffffff !important;
    }

    .stSuccess, div[data-testid="stNotification"] {
        background-color: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        border: 1px solid #f87171 !important;
        color: #ffffff !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        border: 1px solid #60a5fa !important;
        color: #ffffff !important;
    }
    
    /* Ensure text on white backgrounds */
    .stTextInput, .stNumberInput, .stSelectbox {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER SECTION
# ============================================================
st.markdown("""
<div class="custom-header">
    <h1>ML Classification Studio</h1>
    <div class="bits-info">BITS Pilani • M.Tech (AIML)</div>
    <div class="assignment-tag">Machine Learning • Assignment 2</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PRE-TRAINED MODELS LOADER
# ============================================================

@st.cache_resource
def load_trained_pipeline():
    """Load the pre-trained models and encoders from disk"""
    model_path = 'trained_models.joblib'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None
    else:
        st.warning("⚠️ Pre-trained models not found. Please run `train_models.py` first.")
        return None

def evaluate_models(X_eval, y_eval, models_dict):
    """Evaluate a dictionary of models on the provided data"""
    results = {}
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_eval)
        
        accuracy = accuracy_score(y_eval, y_pred)
        auc = None
        try:
            if hasattr(model, 'predict_proba'):
                if len(np.unique(y_eval)) == 2:
                    y_pred_proba = model.predict_proba(X_eval)[:, 1]
                    auc = roc_auc_score(y_eval, y_pred_proba)
                else:
                    y_pred_proba = model.predict_proba(X_eval)
                    auc = roc_auc_score(y_eval, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            pass
            
        precision = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_eval, y_pred)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_eval, y_pred),
            'classification_report': classification_report(y_eval, y_pred, output_dict=True)
        }
    return results

# ============================================================
# KAGGLE DATASET LOADER
# ============================================================

@st.cache_data
def load_kaggle_dataset(dataset_identifier):
    """Load dataset from Kaggle using kagglehub"""
    if kagglehub is None:
        st.error("kagglehub is not installed. Install it with: pip install kagglehub")
        return None
    
    try:
        import os
        import glob
        from kagglehub import KaggleDatasetAdapter
        
        # Priority mapping for specific datasets
        file_to_load = None
        if "chetanmittal033/credit-card-fraud-data" in dataset_identifier:
            # User specifically wants fraudTrain.csv for testing/evaluation
            file_to_load = "fraudTrain.csv"
            # Since fraudTrain is in kartik2112, we adjust the identifier if needed
            # but the user prompt says: load from chetanmittal033... we will use fraudTrain to test
            # This is a bit inconsistent in their request, but I'll try to find fraudTrain.csv
            # in the downloaded datasets.
        
        # Load the latest version using the adapter if a file is specified
        if file_to_load:
            try:
                # User's suggested adapter method
                # We need to find where fraudTrain is. 
                # If chetanmittal033 doesn't have it, load_dataset might fail.
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_identifier,
                    file_to_load,
                )
            except:
                # Fallback to the other dataset if file not found in chetanmittal033
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "kartik2112/fraud-detection",
                    file_to_load,
                )
        else:
            # Generic loading logic
            dataset_path = kagglehub.dataset_download(dataset_identifier)
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if not csv_files:
                st.error(f"No CSV files found in dataset: {dataset_identifier}")
                return None
            df = pd.read_csv(csv_files[0])

        # Consistent truncation as requested
        if len(df) > 12000:
            df = df.head(12000)
            st.warning("⚠️ Dataset truncated to 12,000 rows for performance optimization.")
        
        st.info(f"✅ Successfully loaded dataset: {len(df)} rows × {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset from Kaggle: {str(e)}")
        return None

# ============================================================
# MODEL IMPLEMENTATIONS
# ============================================================

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 6 classification models and return results"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle multi-class for AUC
        try:
            if len(np.unique(y_test)) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results

def plot_learning_curve(model, X, y, model_name):
    """Generate a learning curve plot"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, 'o-', color="#0f172a", label="Training score", linewidth=2)
    ax.plot(train_sizes, test_mean, 'o-', color="#3b82f6", label="Cross-validation score", linewidth=2)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#0f172a")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#3b82f6")
    
    ax.set_title(f"Learning Curve: {model_name}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Training Examples", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'test_size' not in st.session_state:
    st.session_state.test_size = 20
if 'run_pretrained' not in st.session_state:
    st.session_state.run_pretrained = False
if 'run_train_test' not in st.session_state:
    st.session_state.run_train_test = False
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'All Models (Comparison)'
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Upload CSV"

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #1e3c72; font-family: "Crimson Pro", serif;'>🔬 Control Panel</h2>
        <p style='color: #7f8c8d; font-size: 0.9rem;'>Model Parameters & Ingestion</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Data source selection
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data source:",
        options=["Upload CSV", "Load from Kaggle"],
        help="Select how you want to load your dataset",
        key="data_source_radio"
    )
    
    st.markdown("---")
    
    if data_source == "Upload CSV":
        st.markdown("### 📁 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload your test dataset (CSV)", 
            type=['csv'],
            help="Upload a CSV file with features and target variable"
        )
        
        # Sample file download
        import os
        sample_path = os.path.join(os.path.dirname(__file__), 'fraudTest.csv')
        try:
            with open(sample_path, 'rb') as f:
                csv_sample = f.read()
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name="fraud_test_sample.csv",
                mime="text/csv",
                width="stretch"
            )
        except Exception as e:
            st.error(f"Sample file not found: {str(e)}")
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            if len(st.session_state.df) > 12000:
                st.session_state.df = st.session_state.df.head(12000)
                st.warning("⚠️ Uploaded dataset truncated to 12,000 rows for performance optimization.")
    
    else:
        st.markdown("### 🔗 Kaggle Dataset")
        
        # Quick access to popular datasets
        st.markdown("**Popular Datasets:**")
        popular_datasets = {
            "Credit Card Fraud Detection": "chetanmittal033/credit-card-fraud-data",
            "Adult Census Income": "uciml/adult-census-income",
            "Remote Work Burnout and Social Isolation": "aryanmdev/remote-work-burnout-and-social-isolation-2026"
        }
        
        selected_dataset = st.selectbox(
            "Select a dataset:",
            list(popular_datasets.keys()),
        )
        
        dataset_id = popular_datasets[selected_dataset]
        
        # Load button for Kaggle dataset
        if st.button("📥 Load Kaggle Dataset", width="stretch"):
            if kagglehub is None:
                st.error("❌ kagglehub is not installed. Run: `pip install kagglehub`")
            else:
                with st.spinner(f"Loading {selected_dataset}..."):
                    loaded_df = load_kaggle_dataset(dataset_id)
                    if loaded_df is not None:
                        st.session_state.df = loaded_df
                        st.success(f"✅ Successfully loaded {selected_dataset}!")
    
    st.markdown("---")
    
    if st.session_state.df is not None:
        st.markdown("### ⚙️ Model Configuration")
        
        # Model selection
        st.session_state.model_choice = st.selectbox(
            "Select Classification Model",
            [
                'All Models (Comparison)',
                'Logistic Regression',
                'Decision Tree',
                'K-Nearest Neighbors',
                'Naive Bayes',
                'Random Forest',
                'XGBoost'
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Test size slider
        st.session_state.test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )
        
        st.markdown("---")
        
        # Action buttons
        if st.session_state.data_source_radio == "Load from Kaggle":
            st.session_state.run_train_test = st.button("🚀 Train & Evaluate", use_container_width=True)
            st.session_state.run_pretrained = False
        else:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.session_state.run_pretrained = st.button("🧪 Run Pre-trained", use_container_width=True)
            with col_btn2:
                st.session_state.run_train_test = st.button("🚀 Train & Test", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: white; font-family: monospace;'>
        <small>
        <b>BITS Pilani</b><br>
        M.Tech - AIML<br>
        Machine Learning Assignment 2
        </small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT AREA
# ============================================================

if st.session_state.df is None:
    # Welcome screen with instructions (Reference from README)
    st.markdown("""
    <div class="content-box">
        <h3>📖 a. Problem Statement</h3>
        <p>This interactive platform enables the comparison and analysis of multiple classification algorithms. 
        It automates data preprocessing, trains six industry-standard models, and evaluates them across six key 
        performance metrics for the <b>BITS Pilani M.Tech (AIML)</b> assignment.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- NEW: Pre-trained Model Performance Details (Top of landing page) ---
    pipeline_data = load_trained_pipeline()
    if pipeline_data and 'metrics' in pipeline_data:
        st.markdown('<div class="section-header">🏆 Pre-trained Model Benchmarks</div>', unsafe_allow_html=True)
        st.info("💡 Below are the performance metrics for the pre-trained models on the Credit Card Fraud dataset.")
        
        metrics_dict = pipeline_data['metrics']
        perf_data = []
        for model_name, m in metrics_dict.items():
            perf_data.append({
                'Model': model_name,
                'Accuracy': m['accuracy'],
                'AUC': m['auc'] if m['auc'] is not None else 0.0,
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1 Score': m['f1'],
                'MCC Score': m['mcc']
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(
            perf_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'F1 Score', 'MCC Score'])
            .format(precision=4),
            width="stretch"
        )
    
    st.markdown("""
    <div class="content-box">
        <h3>📊 b. Dataset Description</h3>
        <p>The system handles structured tabular data (CSV). It supports both numerical and categorical features 
        through automated encoding and missing value treatment. You can upload custom datasets or load 
        standard benchmarks like <i>Credit Card Fraud Detection</i> or <i>Adult Census Income Prediction</i> directly from Kaggle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🛠️ Features & Studio Guide</div>', unsafe_allow_html=True)
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        - ✅ **6 Classification Algorithms**
        - ✅ **Comprehensive Metrics**
        - ✅ **Visual Comparisons**
        """)
    
    with features_col2:
        st.markdown("""
        - ✅ **Confusion Matrix Analysis**
        - ✅ **Interactive Model Selection**
        - ✅ **Professional Reports**
        """)
    
    st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Select Data Source**: Choose to upload a CSV or load from Kaggle
    2. **Upload/Load Dataset**: Upload your CSV file or select a Kaggle dataset
    3. **Select Model**: Choose a specific model or compare all models
    4. **Configure**: Adjust test size and other parameters
    5. **Train**: Click the train button to begin analysis
    6. **Analyze**: Review metrics, visualizations, and insights
    """)
    
    st.markdown('<div class="section-header">📚 Data Sources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **CSV Upload**: Upload your own dataset in CSV format
    
    **Kaggle Datasets**: Load pre-built datasets including:
    - 👥 Adult Census Income Prediction
    - 💳 Credit Card Fraud Detection
    - 🏠 Remote Work Burnout & Social Isolation
    """)
    
    st.markdown('<div class="section-header">📊 Supported Models</div>', unsafe_allow_html=True)
    
    models_col1, models_col2 = st.columns(2)
    
    with models_col1:
        st.markdown("""
        **Basic Models:**
        - Logistic Regression
        - Decision Tree Classifier
        - K-Nearest Neighbors
        """)
    
    with models_col2:
        st.markdown("""
        **Advanced Models:**
        - Naive Bayes (Gaussian)
        - Random Forest (Ensemble)
        - XGBoost (Ensemble)
        """)

else:
    # Display dataset info
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{len(st.session_state.df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{len(st.session_state.df.columns) - 1}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Classes</div>
            <div class="metric-value">{st.session_state.df.iloc[:, -1].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value">{st.session_state.df.isnull().sum().sum()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show dataset preview
    with st.expander("🔍 View Dataset Preview", expanded=False):
        st.dataframe(st.session_state.df.head(10), width="stretch")
    
    # Process Selection
    if st.session_state.run_pretrained or st.session_state.run_train_test:
        results = None
        X_eval, y_eval = None, None
        
        # Prepare basic clean data
        df_clean = st.session_state.df.copy().dropna()
        
        if len(df_clean) < 5:
            st.error("❌ Dataset too small for evaluation.")
        else:
            if st.session_state.run_pretrained:
                # RUN PRE-TRAINED LOGIC
                pipeline_data = load_trained_pipeline()
                if pipeline_data:
                    with st.spinner('🔄 Running pre-trained models...'):
                        # Apply pre-trained preprocessing
                        cols_to_drop = ['sn', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']
                        df_pretrained = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
                        
                        target_col = 'is_fraud' if 'is_fraud' in df_pretrained.columns else df_pretrained.columns[-1]
                        X = df_pretrained.drop(columns=[target_col])
                        y_raw = df_pretrained[target_col]
                        
                        encoders = pipeline_data['encoders']
                        try:
                            # Feature encoding
                            for col in X.columns:
                                if col in encoders:
                                    le = encoders[col]
                                    X[col] = X[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                                    X[col] = le.transform(X[col])
                            
                            # Target encoding
                            if 'target' in encoders:
                                y = encoders['target'].transform(y_raw.astype(str) if y_raw.dtype == 'object' else y_raw)
                            else:
                                y = y_raw.values
                                
                            # Sample for evaluation if requested
                            if st.session_state.test_size < 100:
                                _, X_eval, _, y_eval = train_test_split(X, y, test_size=st.session_state.test_size/100, random_state=42, stratify=y)
                            else:
                                X_eval, y_eval = X, y
                                
                            results = evaluate_models(X_eval, y_eval, pipeline_data['models'])
                            st.success('✅ Pre-trained models evaluated!')
                        except Exception as e:
                            st.error(f"Error during pre-trained evaluation: {e}")
                            
            elif st.session_state.run_train_test:
                # TRAIN & TEST LOGIC (Original behavior)
                with st.spinner('🔄 Training models from scratch...'):
                    # Drop non-predictive columns if present to stay consistent with pre-trained models
                    cols_to_drop = ['sn', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']
                    df_training = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
                    
                    target_col = 'is_fraud' if 'is_fraud' in df_training.columns else df_training.columns[-1]
                    X = df_training.drop(columns=[target_col])
                    y_raw = df_training[target_col]
                    
                    # Fresh encoding for this dataset
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y_raw.astype(str) if y_raw.dtype == 'object' else y_raw)
                    
                    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=st.session_state.test_size/100, random_state=42)
                    
                    results = train_all_models(X_train, X_eval, y_train, y_eval)
                    st.success('✅ Models trained and tested!')

        # Display Results if available
        if results:
            if st.session_state.model_choice == 'All Models (Comparison)':
                st.markdown('<div class="section-header">📈 Model Comparison</div>', unsafe_allow_html=True)
                
                comparison_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [float(results[m]['accuracy']) for m in results.keys()],
                    'AUC Score': [float(results[m]['auc']) if results[m]['auc'] is not None else 0.0 for m in results.keys()],
                    'Precision': [float(results[m]['precision']) for m in results.keys()],
                    'Recall': [float(results[m]['recall']) for m in results.keys()],
                    'F1 Score': [float(results[m]['f1']) for m in results.keys()],
                    'MCC Score': [float(results[m]['mcc']) for m in results.keys()]
                })
        
                st.dataframe(
                    comparison_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'F1 Score'])
                    .format(precision=4),
                    width="stretch"
                )
                        
                # Best model highlight
                best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
                best_accuracy = comparison_df['Accuracy'].max()
                
                st.markdown(f"""
                <div class="content-box">
                    <h3>🏆 Best Performing Model</h3>
                    <p><b>{best_model}</b> achieved the highest accuracy of <b>{best_accuracy:.4f}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown('<div class="section-header">📊 Performance Visualization</div>', unsafe_allow_html=True)
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.patch.set_facecolor('#f8f9fa')
                
                metrics = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
                # Professional palette: Deep Slate, Midnight Blue, Teal, Slate Blue, Gold, Coral
                colors = ['#2c3e50', '#2980b9', '#16a085', '#7f8c8d', '#f1c40f', '#e67e22']
                
                for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
                    values = comparison_df[metric].values
                    bars = ax.bar(range(len(results)), values, color=color, alpha=0.85, edgecolor='#ecf0f1', linewidth=1.5)
                    ax.set_xticks(range(len(results)))
                    ax.set_xticklabels([m.replace(' ', '\n') for m in results.keys()], fontsize=10, fontweight='500', color='#34495e')
                    ax.set_title(metric, fontsize=16, fontweight='800', color='#2c3e50', pad=20)
                    ax.set_ylim([0, 1.1])
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                    ax.set_facecolor('#ffffff')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#bdc3c7')
                    ax.spines['bottom'].set_color('#bdc3c7')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Confusion Matrices for all models
                st.markdown('<div class="section-header">🎯 Confusion Matrices Comparison</div>', unsafe_allow_html=True)
                
                # Create a grid for confusion matrices (2 columns)
                cm_cols = st.columns(2)
                for idx, (model_name, model_result) in enumerate(results.items()):
                    with cm_cols[idx % 2]:
                        st.markdown(f"#### {model_name}")
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        sns.heatmap(
                            model_result['confusion_matrix'], 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues',
                            cbar=False,
                            linewidths=1,
                            linecolor='white',
                            square=True,
                            annot_kws={"size": 10}
                        )
                        ax_cm.set_xlabel('Predicted', fontsize=10)
                        ax_cm.set_ylabel('True', fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                
            else:
                # Single model analysis
                st.markdown(f'<div class="section-header">📊 {st.session_state.model_choice} - Detailed Analysis</div>', unsafe_allow_html=True)
                
                model_result = results[st.session_state.model_choice]
                
                # Metrics display
                metric_cols = st.columns(6)
                
                metrics_data = [
                    ('Accuracy', model_result['accuracy']),
                    ('AUC', model_result['auc'] if model_result['auc'] is not None else 0),
                    ('Precision', model_result['precision']),
                    ('Recall', model_result['recall']),
                    ('F1 Score', model_result['f1']),
                    ('MCC', model_result['mcc'])
                ]
                
                for col, (label, value) in zip(metric_cols, metrics_data):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{value:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Confusion Matrix and Learning Curve
                st.markdown('<div class="section-header">🎯 Detailed Evaluation</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        model_result['confusion_matrix'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        cbar_kws={'label': 'Count'},
                        linewidths=2,
                        linecolor='white',
                        square=True
                    )
                    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
                    ax.set_title(f'{st.session_state.model_choice} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
                    st.pyplot(fig)
                
                with col2:
                    if st.session_state.run_train_test:
                        st.markdown("### Learning Curve")
                        with st.spinner('Generating learning curve...'):
                            # Use X_train, y_train only available in run_train_test
                            fig_lc = plot_learning_curve(model_result['model'], X_eval, y_eval, st.session_state.model_choice)
                            st.pyplot(fig_lc)
                            plt.close(fig_lc)
                    else:
                        st.markdown("### Classification Report")
                        report_df = pd.DataFrame(model_result['classification_report']).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']).format(precision=3), width="stretch")
                
                # Classification Report and Insights
                st.markdown('<div class="section-header">🔍 Deep Insights</div>', unsafe_allow_html=True)
                col_bits1, col_bits2 = st.columns(2)
                
                with col_bits1:
                    if st.session_state.run_train_test:
                        st.markdown("### Classification Report")
                        report_df = pd.DataFrame(model_result['classification_report']).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']).format(precision=3), width="stretch")
                    else:
                        st.info("Additional insights available when training from scratch.")
                
                with col_bits2:
                    # Additional insights
                    st.markdown("### 💡 Model Insights")
                    st.markdown(f"""
                    - **Total samples evaluated**: {len(y_eval)}
                    - **Correct Predictions**: {(model_result['y_pred'] == y_eval).sum()}
                    - **Incorrect Predictions**: {(model_result['y_pred'] != y_eval).sum()}
                    - **Overall Accuracy**: {model_result['accuracy']:.4f}
                    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-family: "Crimson Pro", serif; padding: 2rem 0; background-color: #f8fafc; border-top: 1px solid #e2e8f0;'>
    <h4 style="margin: 0; color: #0f172a;">Machine Learning Assignment 2</h4>
    <p style="margin: 0.5rem 0; color: #334155;">BITS Pilani - M.Tech (AIML) Program</p>
    <p style="font-size: 0.8rem; margin-top: 1rem; color: #94a3b8;">© 2026 ML Classification Studio</p>
</div>
""", unsafe_allow_html=True)
