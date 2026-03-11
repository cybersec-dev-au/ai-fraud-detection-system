import requests
import json
from datetime import datetime

def test_fraud_api():
    url = "http://localhost:8000/predict_transaction"
    
    # Test cases
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "Transaction_ID": "TXN_TEST_001",
                "User_ID": "USER_001",
                "Transaction_Amount": 45.50,
                "Transaction_Time": datetime.now().isoformat(),
                "Merchant_Category": "Retail",
                "Device_Type": "Desktop",
                "Location": "New York",
                "Previous_Transaction_Count": 50,
                "Average_User_Spending": 60.00
            }
        },
        {
            "name": "Suspicious Transaction (High Amount, New Account)",
            "data": {
                "Transaction_ID": "TXN_TEST_002",
                "User_ID": "USER_999",
                "Transaction_Amount": 2500.00,
                "Transaction_Time": datetime.now().isoformat(),
                "Merchant_Category": "Electronics",
                "Device_Type": "Mobile",
                "Location": "Unknown",
                "Previous_Transaction_Count": 0,
                "Average_User_Spending": 150.00
            }
        }
    ]
    
    print("Testing Fraud Detection API...\n")
    
    for case in test_cases:
        print(f"--- Case: {case['name']} ---")
        try:
            response = requests.post(url, json=case['data'])
            if response.status_code == 200:
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection error: {e}")
        print("\n")

if __name__ == "__main__":
    test_fraud_api()
