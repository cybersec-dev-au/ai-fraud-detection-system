import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import os

def train_models(X_train, y_train):
    """
    Trains multiple models for comparison.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    # Isolation Forest for Anomaly Detection (unsupervised/semi-supervised)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    trained_models['Isolation Forest'] = iso_forest
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates models and returns a comparison dataframe.
    """
    results = []
    
    for name, model in trained_models.items():
        if name == 'Isolation Forest':
            # -1 for anomaly, 1 for normal -> transform to 1 for fraud, 0 for normal
            preds = model.predict(X_test)
            preds = [1 if x == -1 else 0 for x in preds]
            probs = [1 if x == -1 else 0 for x in preds] # Isolation forest doesn't provide easy probability
        else:
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs) if name != 'Isolation Forest' else 0.5 # Simplified
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        })
        
    return pd.DataFrame(results)

def save_model_artifacts(model, scaler, encoders, features, path="models"):
    """
    Saves the best model and preprocessing objects.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    joblib.dump(model, os.path.join(path, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(path, "encoders.pkl"))
    joblib.dump(features, os.path.join(path, "features_list.pkl"))
    print(f"Model artifacts saved to {path}")
