"""
ML Classification Models - Training Script
BITS Pilani - M.Tech (AIML/DSE)
Machine Learning Assignment 2

This script demonstrates the implementation of 6 classification models
and their evaluation metrics calculation.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import kagglehub
from kagglehub import KaggleDatasetAdapter
import glob

import warnings
warnings.filterwarnings('ignore')


class MLClassificationPipeline:
    """
    A comprehensive classification pipeline that trains and evaluates
    multiple ML models on a given dataset.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the classification pipeline.
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classification models."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
        }
    
    def preprocess_data(self, df, target_column):
        """
        Preprocess the dataset: handle categorical variables and split data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        target_column : str
            Name of the target column
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Split dataset
        """
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle categorical features
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Handle categorical target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            label_encoders['target'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test, label_encoders
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics for a model.
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_pred_proba : array, optional
            Predicted probabilities for AUC calculation
        
        Returns:
        --------
        dict : Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['f1'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # AUC Score (handle binary and multi-class)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if y_pred_proba.ndim == 2:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(
                        y_true, y_pred_proba, 
                        multi_class='ovr', 
                        average='weighted'
                    )
            except Exception as e:
                print(f"AUC calculation error: {e}")
                metrics['auc'] = None
        else:
            metrics['auc'] = None
        
        return metrics
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate their performance.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Training and testing data
        
        Returns:
        --------
        dict : Dictionary containing results for all models
        """
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Generate confusion matrix and classification report
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': cr,
                'predictions': y_pred
            }
            
            # Print results
            print(f"\nMetrics for {model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  AUC Score: {metrics['auc']:.4f}" if metrics['auc'] else "  AUC Score: N/A")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  MCC Score: {metrics['mcc']:.4f}")
        
        return self.results
    
    def generate_comparison_table(self):
        """
        Generate a comparison table of all models.
        
        Returns:
        --------
        pandas.DataFrame : Comparison table
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'AUC': f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'MCC Score': f"{metrics['mcc']:.4f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison (default: 'accuracy')
        
        Returns:
        --------
        tuple : (model_name, model, metric_value)
        """
        best_score = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            score = result['metrics'][metric]
            if score is not None and score > best_score:
                best_score = score
                best_model_name = model_name
        
        return (
            best_model_name, 
            self.results[best_model_name]['model'],
            best_score
        )

    def save_pipeline(self, filename='trained_models.joblib', encoders=None):
        """Save the trained models, encoders, and evaluation metrics."""
        data = {
            'models': {name: result['model'] for name, result in self.results.items()},
            'encoders': encoders,
            'metrics': {name: result['metrics'] for name, result in self.results.items()}
        }
        joblib.dump(data, filename)
        print(f"Models and encoders saved to {filename}")


# Example usage
if __name__ == "__main__":
    """
    Train and save models for Credit Card Fraud Detection.
    """
    
    print("ML Classification Pipeline - Training & Saving")
    print("="*60)
    
    # Kaggle dataset identifier
    kaggle_id = "chetanmittal033/credit-card-fraud-data"
    file_to_load = "fraudTest.csv"
    
    print(f"Loading dataset {file_to_load} from {kaggle_id}...")
    try:
        # Load the latest version using the adapter pattern suggested by user
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_id,
            file_to_load,
        )
        print(f"Successfully loaded {file_to_load} ({len(df)} rows)")
    except Exception as e:
        print(f"Error loading via adapter: {e}")
        # Fallback to standard download if adapter fails
        try:
            download_path = kagglehub.dataset_download(kaggle_id)
            target_path = os.path.join(download_path, file_to_load)
            df = pd.read_csv(target_path)
            print(f"Successfully loaded {file_to_load} via download fallback ({len(df)} rows)")
        except Exception as e2:
            print(f"Fallback error: {e2}")
            df = None
        
    if df is not None:
        # Limit rows for faster processing if the dataset is very large
        # The user previously asked for 12000
        if len(df) > 12000:
            print("Truncating training dataset to 12,000 rows as requested...")
            df = df.head(12000)
        
        # Drop columns that are likely not useful
        cols_to_drop = ['sn', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time']
        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Initialize pipeline
        pipeline = MLClassificationPipeline(test_size=0.2, random_state=42)
        
        # Preprocess data
        print("Preprocessing data...")
        # Assume the last column is the target if 'is_fraud' is not found, 
        # but in fraudTest.csv it is 'is_fraud'
        target_col = 'is_fraud' if 'is_fraud' in df_clean.columns else df_clean.columns[-1]
        X_train, X_test, y_train, y_test, encoders = pipeline.preprocess_data(
            df_clean, target_column=target_col
        )
        
        # Train and evaluate all models
        print("Training models...")
        results = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save models and encoders
        pipeline.save_pipeline('trained_models.joblib', encoders=encoders)
        
        # Generate comparison table
        comparison_df = pipeline.generate_comparison_table()
        print("\nModel Comparison Table:")
        print(comparison_df)
