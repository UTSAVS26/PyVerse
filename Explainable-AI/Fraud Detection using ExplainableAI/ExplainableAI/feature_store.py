import redis
import pandas as pd
import json
import numpy as np

# Redis connection
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    # Test the connection
    r.ping()
    print("Connected to Redis successfully.")
except redis.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
    print("Ensure Redis is running (e.g., 'docker run -d -p 6379:6379 redis')")
    exit(1)

def store_features(user_id, features):
    """Store user features in Redis"""
    try:
        r.set(f"user:{user_id}", json.dumps(features))
        print(f"Stored features for user {user_id}")
    except redis.RedisError as e:
        print(f"Error storing features for user {user_id}: {e}")

def get_features(user_id):
    """Get user features"""
    try:
        data = r.get(f"user:{user_id}")
        return json.loads(data) if data else {}
    except redis.RedisError as e:
        print(f"Error retrieving features for user {user_id}: {e}")
        return {}

# Example: Store sample features from data
try:
    df = pd.read_csv('creditcard.csv').head(10)  # Sample
    for idx, row in df.iterrows():
        user_id = f"pseudo_{idx}"
        features = {
            'txns_last_1h': row.get('txns_last_1h', 0),  # From engineered data
            'avg_amount_30d': row.get('avg_amount_30d', 0),
            'LogAmount': np.log1p(row['Amount'])
        }
        store_features(user_id, features)
    print("Features stored in Redis.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' file not found.")
except Exception as e:
    print(f"Unexpected error: {e}")