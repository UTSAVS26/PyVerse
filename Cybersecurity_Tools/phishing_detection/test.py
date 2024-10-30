# Load the trained model
import joblib
from features_extraction import extract_features
import numpy as np


def predict_phishing(url):
    try:
        # Load the model
        model = joblib.load('voting_classifier_model.pkl')
        
        # Extract features
        get_features = extract_features(url)

        features = [i for i in get_features.values()]
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        return prediction[0]
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    while True:
        try:
            # Get URL input
            url = input("\nEnter URL to check (or 'quit' to exit): ")
            
            if url.lower() == 'quit':
                break
            
            # Make prediction
            result = predict_phishing(url)
            
            # Display result
            if result is not None:
                if result == 1:
                    print("⚠️ Warning: This URL is potentially PHISHING")
                else:
                    print("✅ This URL appears to be LEGITIMATE")
            
            # Ask if user wants to check another URL
            choice = input("\nWould you like to check another URL? (y/n): ")
            if choice.lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a valid URL")

