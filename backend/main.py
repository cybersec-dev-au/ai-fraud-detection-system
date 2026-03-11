from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI(title="AI Fraud Detection API", description="Real-time transaction risk scoring")

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features_list.pkl")

# We'll use global variables for the model components
model = None
scaler = None
encoders = None
features_list = None

@app.on_event("startup")
def load_model():
    global model, scaler, encoders, features_list
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        features_list = joblib.load(FEATURES_PATH)
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

class TransactionRequest(BaseModel):
    Transaction_ID: str
    User_ID: str
    Transaction_Amount: float
    Transaction_Time: str  # ISO format
    Merchant_Category: str
    Device_Type: str
    Location: str
    Previous_Transaction_Count: int
    Average_User_Spending: float

@app.post("/predict_transaction")
async def predict_transaction(data: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # 1. Convert to DataFrame
        df_input = pd.DataFrame([data.dict()])
        
        # 2. Preprocess Time
        dt = datetime.fromisoformat(data.Transaction_Time)
        df_input['Hour'] = dt.hour
        df_input['DayOfWeek'] = dt.weekday()
        
        # 3. Encode Categorical
        for col, le in encoders.items():
            # Handle unseen categories by defaulting to the first class or a special value
            if data.dict()[col] in le.classes_:
                df_input[col] = le.transform([data.dict()[col]])[0]
            else:
                df_input[col] = 0 # Default fallback
                
        # 4. Feature Selection & Alignment
        X = df_input[features_list]
        
        # 5. Scaling
        X_scaled = scaler.transform(X)
        
        # 6. Predict
        fraud_prob = float(model.predict_proba(X_scaled)[0][1])
        
        # 7. Risk Scoring Logic
        risk_level = "LOW"
        action = "ALLOW"
        
        if fraud_prob > 0.70:
            risk_level = "HIGH"
            action = "BLOCK"
        elif fraud_prob > 0.30:
            risk_level = "MEDIUM"
            action = "CHALLENGE (MFA)"
            
        return {
            "transaction_id": data.Transaction_ID,
            "fraud_probability": round(fraud_prob, 4),
            "risk_score": round(fraud_prob * 100, 2),
            "risk_level": risk_level,
            "recommended_action": action,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "Service is running", "model": "XGBoost Fraud Detector"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
