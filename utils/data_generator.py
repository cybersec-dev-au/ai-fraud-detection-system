import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_transaction_data(n_samples=10000, fraud_ratio=0.05):
    """
    Simulates realistic transaction data for fraud detection.
    """
    np.random.seed(42)
    
    # Generate Transaction ID
    transaction_ids = [f"TXN_{i:06d}" for i in range(n_samples)]
    
    # Generate User IDs (approx 1 user for every 10 transactions)
    n_users = n_samples // 10
    user_ids = [f"USER_{np.random.randint(1, n_users + 1):04d}" for _ in range(n_samples)]
    
    # Generate Transaction Time (within last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    times = [start_date + timedelta(seconds=np.random.randint(0, 30*24*3600)) for _ in range(n_samples)]
    
    # Merchant Categories
    categories = ['Retail', 'Travel', 'Food', 'Electronics', 'Entertainment', 'Subscription', 'Services']
    category_choices = np.random.choice(categories, n_samples)
    
    # Device Types
    devices = ['Mobile', 'Desktop', 'Tablet', 'Unknown']
    device_choices = np.random.choice(devices, n_samples)
    
    # Locations
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    location_choices = np.random.choice(locations, n_samples)
    
    # Amounts
    amounts = np.random.exponential(scale=100, size=n_samples) + 1.0 # Basic distribution
    
    # Simulate Fraud Label (initially balanced for logic, then adjusted)
    fraud_labels = np.zeros(n_samples)
    
    # Logic to make some transactions suspicious
    for i in range(n_samples):
        # High amount rule
        if amounts[i] > 1000 and random.random() < 0.3:
            fraud_labels[i] = 1
        
        # High-risk categories
        if category_choices[i] in ['Electronics', 'Travel'] and amounts[i] > 500 and random.random() < 0.2:
            fraud_labels[i] = 1
            
    # Adjust to desired fraud ratio
    current_ratio = sum(fraud_labels) / n_samples
    if current_ratio > fraud_ratio:
        # Reduce fraud to match ratio
        fraud_indices = np.where(fraud_labels == 1)[0]
        n_to_remove = int(len(fraud_indices) - (n_samples * fraud_ratio))
        if n_to_remove > 0:
            remove_indices = np.random.choice(fraud_indices, n_to_remove, replace=False)
            fraud_labels[remove_indices] = 0
    elif current_ratio < fraud_ratio:
        # Increase fraud to match ratio
        non_fraud_indices = np.where(fraud_labels == 0)[0]
        n_to_add = int((n_samples * fraud_ratio) - len(np.where(fraud_labels == 1)[0]))
        if n_to_add > 0:
            add_indices = np.random.choice(non_fraud_indices, n_to_add, replace=False)
            fraud_labels[add_indices] = 1

    # Feature Engineering (Derived features)
    # Previous_Transaction_Count (Simulated)
    prev_count = np.random.randint(0, 100, n_samples)
    
    # Average_User_Spending (Simulated)
    avg_spending = np.random.normal(loc=150, scale=50, size=n_samples)
    avg_spending[avg_spending < 0] = 50 # Floor at 50

    # Ensure some correlation for fraudsters having higher spending or less history
    fraud_indices = np.where(fraud_labels == 1)[0]
    amounts[fraud_indices] += np.random.normal(loc=200, scale=100, size=len(fraud_indices))
    amounts[amounts < 0] = 1.0
    
    prev_count[fraud_indices] = np.random.randint(0, 5, len(fraud_indices)) # New accounts or low history
    
    data = {
        'Transaction_ID': transaction_ids,
        'User_ID': user_ids,
        'Transaction_Amount': amounts,
        'Transaction_Time': times,
        'Merchant_Category': category_choices,
        'Device_Type': device_choices,
        'Location': location_choices,
        'Previous_Transaction_Count': prev_count,
        'Average_User_Spending': avg_spending,
        'Fraud_Label': fraud_labels.astype(int)
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_transaction_data(n_samples=20000, fraud_ratio=0.05)
    output_path = os.path.join("data", "raw", "transactions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions and saved to {output_path}")
    print(f"Fraud distribution: \n{df['Fraud_Label'].value_counts(normalize=True)}")
