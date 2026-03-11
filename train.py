import pandas as pd
from utils.data_generator import generate_transaction_data
from utils.preprocessing import preprocess_data, prepare_train_test
from utils.modeling import train_models, evaluate_models, save_model_artifacts
import os

def main():
    # 1. Generate/Load Data
    print("Step 1: Generating synthetic transaction data...")
    df = generate_transaction_data(n_samples=25000, fraud_ratio=0.03)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/transactions.csv", index=False)
    
    # 2. Preprocessing
    print("Step 2: Preprocessing and feature engineering...")
    df_processed, label_encoders = preprocess_data(df)
    
    # 3. Train/Test Split & Resampling
    print("Step 3: Preparing train/test sets and handling class imbalance...")
    X_train_res, X_test, y_train_res, y_test, scaler, feature_cols = prepare_train_test(df_processed)
    
    # 4. Training
    print("Step 4: Training machine learning models...")
    trained_models = train_models(X_train_res, y_train_res)
    
    # 5. Evaluation
    print("Step 5: Evaluating models...")
    perf_results = evaluate_models(trained_models, X_test, y_test)
    print("\nModel Performance Comparison:")
    print(perf_results.to_string(index=False))
    
    # Accuracy Warning
    print("\nNote: Accuracy is not reliable here because only 3% of transactions are fraudulent.")
    print("A model that predicts 'Not Fraud' 100% of the time would have 97% accuracy but 0 recall/precision for fraud.")
    print("We prioritize F1-Score and Precision/Recall for this task.")

    # 6. Save best model (selecting XGBoost as default best for this demo)
    print("\nStep 6: Saving best model (XGBoost) for deployment...")
    save_model_artifacts(
        trained_models['XGBoost'], 
        scaler, 
        label_encoders, 
        feature_cols.tolist()
    )

if __name__ == "__main__":
    main()
