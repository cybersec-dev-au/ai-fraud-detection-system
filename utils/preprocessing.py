import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """
    Standard preprocessing pipeline for fraud detection.
    """
    df = df.copy()
    
    # 1. Feature Engineering: Extract time-based features
    df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'])
    df['Hour'] = df['Transaction_Time'].dt.hour
    df['DayOfWeek'] = df['Transaction_Time'].dt.dayofweek
    
    # 2. Categorical Encoding
    cat_cols = ['Merchant_Category', 'Device_Type', 'Location']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # 3. Drop IDs and Time (not useful for training after extraction)
    df = df.drop(['Transaction_ID', 'User_ID', 'Transaction_Time'], axis=1)
    
    return df, label_encoders

def prepare_train_test(df, target_col='Fraud_Label'):
    """
    Splits data and handles imbalance using SMOTE.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle Imbalance with SMOTE on Training Data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_resampled, X_test_scaled, y_resampled, y_test, scaler, X.columns
