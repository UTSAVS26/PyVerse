# Web Application for Loan Prediction System
# Run this after training the model using the main script

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Set page configuration
st.set_page_config(
    page_title="Bank Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load model and preprocessors
@st.cache_resource
def load_model():
    try:
        with open('loan_prediction_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None

def create_sample_model():
    """Create a sample model if none exists"""
    try:
        # Import the LoanPredictionSystem from the training script
        sys.path.append('.')
        
        # Create a basic model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Create sample data for training
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample features
        data = {
            'Current Loan Amount': np.random.lognormal(9, 0.8, n_samples).astype(int),
            'Term': np.random.choice(['Short Term', 'Long Term'], n_samples, p=[0.3, 0.7]),
            'Credit Score': np.random.normal(650, 100, n_samples).astype(int),
            'Annual Income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
            'Years in current job': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                                                    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], n_samples),
            'Home Ownership': np.random.choice(['Rent', 'Own Home', 'Mortgage', 'HaveMortgage'], n_samples, p=[0.3, 0.3, 0.3, 0.1]),
            'Purpose': np.random.choice(['Debt Consolidation', 'Home Improvements', 'Other', 'Buy House', 
                                       'Business Loan', 'Buy a Car', 'major_purchase', 'Medical Bills',
                                       'Educational Expenses', 'Vacation', 'Wedding', 'Moving', 'Renewable Energy'], n_samples),
            'Monthly Debt': np.random.randint(100, 5000, n_samples),
            'Years of Credit History': np.random.randint(1, 50, n_samples),
            'Months since last delinquent': np.random.randint(0, 200, n_samples),
            'Number of Open Accounts': np.random.randint(1, 30, n_samples),
            'Number of Credit Problems': np.random.randint(0, 10, n_samples),
            'Current Credit Balance': np.random.randint(0, 100000, n_samples),
            'Maximum Open Credit': np.random.randint(1000, 200000, n_samples),
            'Bankruptcies': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03]),
            'Tax Liens': np.random.choice([0, 1, 2], n_samples, p=[0.90, 0.08, 0.02])
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Clip values to reasonable ranges
        df['Current Loan Amount'] = np.clip(df['Current Loan Amount'], 1000, 500000)
        df['Credit Score'] = np.clip(df['Credit Score'], 300, 850)
        df['Annual Income'] = np.clip(df['Annual Income'], 20000, 200000)
        
        # Create target variable
        credit_score_norm = (df['Credit Score'] - 300) / 550
        income_norm = np.log(df['Annual Income']) / np.log(200000)
        debt_ratio = df['Monthly Debt'] * 12 / df['Annual Income']
        
        approval_prob = (
            0.4 * credit_score_norm +
            0.3 * income_norm +
            0.2 * (1 - debt_ratio) +
            0.1 * (1 - df['Number of Credit Problems'] / 10)
        )
        
        approval_prob += np.random.normal(0, 0.1, n_samples)
        approval_prob = np.clip(approval_prob, 0, 1)
        
        y = np.where(approval_prob > 0.5, 'Fully Paid', 'Charged Off')
        
        # Prepare features for training
        X = df.copy()
        
        # Initialize encoders
        label_encoders = {}
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        label_encoders['target'] = le_target
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_encoded)
        
        # Save model data
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': list(df.columns)
        }
        
        with open('loan_prediction_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        return model_data
        
    except Exception as e:
        st.error(f"Error creating sample model: {e}")
        return None

def main():
    st.title("🏦 Bank Loan Eligibility Predictor")
    st.markdown("---")

    # Load or create model
    model_data = load_model()

    if model_data is None:
        st.warning("No trained model found. Creating a sample model...")
        with st.spinner("Creating sample model..."):
            model_data = create_sample_model()
        if model_data is None:
            st.error("❌ Failed to create sample model. Please check the logs.")
            st.stop()  # Stop execution safely

    # Proceed if model is available
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Single Prediction", "Batch Prediction", "Model Info"])

    if page == "Single Prediction":
        single_prediction_page(model, scaler, label_encoders, feature_names)
    elif page == "Batch Prediction":
        batch_prediction_page(model, scaler, label_encoders, feature_names)
    else:
        model_info_page(model, feature_names)



def single_prediction_page(model, scaler, label_encoders, feature_names):
    st.header("Single Loan Application Prediction")
    st.write("Enter the applicant's details to predict loan eligibility.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        current_loan_amount = st.number_input("Current Loan Amount ($)", 
                                            min_value=0, max_value=10000000, 
                                            value=100000, step=1000)
        
        term = st.selectbox("Loan Term", ["Short Term", "Long Term"])
        
        credit_score = st.number_input("Credit Score", 
                                     min_value=300, max_value=850, 
                                     value=650, step=1)
        
        annual_income = st.number_input("Annual Income ($)", 
                                      min_value=0, max_value=10000000, 
                                      value=50000, step=1000)
        
        years_current_job = st.selectbox("Years in Current Job", 
                                       ["< 1 year", "1 year", "2 years", "3 years", 
                                        "4 years", "5 years", "6 years", "7 years", 
                                        "8 years", "9 years", "10+ years"])
        
        home_ownership = st.selectbox("Home Ownership", 
                                    ["Rent", "Own Home", "Mortgage", "HaveMortgage"])
    
    with col2:
        st.subheader("Financial Information")
        purpose = st.selectbox("Loan Purpose", 
                             ["Debt Consolidation", "Home Improvements", "Other", 
                              "Buy House", "Business Loan", "Buy a Car", 
                              "major_purchase", "Medical Bills", "Educational Expenses",
                              "Vacation", "Wedding", "Moving", "Renewable Energy"])
        
        monthly_debt = st.number_input("Monthly Debt ($)", 
                                     min_value=0, max_value=50000, 
                                     value=1000, step=50)
        
        years_credit_history = st.number_input("Years of Credit History", 
                                             min_value=0, max_value=50, 
                                             value=10, step=1)
        
        months_since_delinquent = st.number_input("Months Since Last Delinquent", 
                                                min_value=0, max_value=200, 
                                                value=24, step=1)
        
        num_open_accounts = st.number_input("Number of Open Accounts", 
                                          min_value=0, max_value=50, 
                                          value=5, step=1)
        
        num_credit_problems = st.number_input("Number of Credit Problems", 
                                            min_value=0, max_value=20, 
                                            value=0, step=1)
        
        current_credit_balance = st.number_input("Current Credit Balance ($)", 
                                               min_value=0, max_value=1000000, 
                                               value=10000, step=500)
        
        max_open_credit = st.number_input("Maximum Open Credit ($)", 
                                        min_value=0, max_value=1000000, 
                                        value=50000, step=1000)
        
        bankruptcies = st.number_input("Number of Bankruptcies", 
                                     min_value=0, max_value=10, 
                                     value=0, step=1)
        
        tax_liens = st.number_input("Number of Tax Liens", 
                                  min_value=0, max_value=10, 
                                  value=0, step=1)
    
    # Predict button
    if st.button("Predict Loan Eligibility", type="primary"):
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'Current Loan Amount': [current_loan_amount],
                'Term': [term],
                'Credit Score': [credit_score],
                'Annual Income': [annual_income],
                'Years in current job': [years_current_job],
                'Home Ownership': [home_ownership],
                'Purpose': [purpose],
                'Monthly Debt': [monthly_debt],
                'Years of Credit History': [years_credit_history],
                'Months since last delinquent': [months_since_delinquent],
                'Number of Open Accounts': [num_open_accounts],
                'Number of Credit Problems': [num_credit_problems],
                'Current Credit Balance': [current_credit_balance],
                'Maximum Open Credit': [max_open_credit],
                'Bankruptcies': [bankruptcies],
                'Tax Liens': [tax_liens]
            })
            
            # Preprocess input data
            processed_data = preprocess_single_input(input_data, label_encoders, scaler)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            # Convert prediction back to original labels
            prediction_label = label_encoders['target'].inverse_transform([prediction])[0]
            
            # Handle probability display for binary/multi-class
            n_classes = len(label_encoders['target'].classes_)
            class_names = label_encoders['target'].classes_
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "paid" in prediction_label.lower() or "approve" in prediction_label.lower():
                    st.success(f"✅ Loan Status: {prediction_label}")
                else:
                    st.error(f"❌ Loan Status: {prediction_label}")
            
            with col2:
                max_prob = np.max(probability)
                st.metric("Prediction Confidence", f"{max_prob:.2%}")
            
            with col3:
                predicted_class_idx = np.argmax(probability)
                st.metric("Predicted Class", class_names[predicted_class_idx])
            
            # Probability distribution
            if n_classes == 2:
                # Determine which class represents approval
                approval_idx = 0 if "paid" in class_names[0].lower() else 1
                approval_prob = probability[approval_idx]
                
                # Probability gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = approval_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightcoral"},
                            {'range': [25, 50], 'color': "orange"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}}))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Multi-class probability bar chart
                prob_df = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probability
                }).sort_values('Probability', ascending=True)
                
                fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                            title="Class Probabilities")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check that all input values are valid.")

def batch_prediction_page(model, scaler, label_encoders, feature_names):
    st.header("Batch Loan Prediction")
    st.write("Upload a CSV file with multiple loan applications for batch prediction.")
    
    # Show expected format
    st.subheader("Expected CSV Format")
    expected_columns = [col for col in feature_names if col != 'Loan Status']
    expected_df = pd.DataFrame(columns=expected_columns)
    st.dataframe(expected_df)
    
    st.write("**Required columns:**")
    st.write(", ".join(expected_columns))
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file
            data = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            if st.button("Make Predictions", type="primary"):
                # Preprocess data
                processed_data = preprocess_batch_input(data, label_encoders, scaler, feature_names)
                
                if processed_data is not None:
                    # Make predictions
                    predictions = model.predict(processed_data)
                    probabilities = model.predict_proba(processed_data)
                    
                    # Convert predictions back to original labels
                    prediction_labels = label_encoders['target'].inverse_transform(predictions)
                    
                    # Add predictions to original data
                    results = data.copy()
                    results['Predicted_Status'] = prediction_labels
                    results['Max_Probability'] = np.max(probabilities, axis=1)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    status_counts = pd.Series(prediction_labels).value_counts()
                    
                    with col1:
                        st.metric("Total Applications", len(prediction_labels))
                    
                    with col2:
                        most_common = status_counts.index[0]
                        most_common_count = status_counts.iloc[0]
                        st.metric(f"Most Common: {most_common}", most_common_count)
                    
                    with col3:
                        avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    # Visualization
                    fig = px.pie(values=status_counts.values, 
                                names=status_counts.index,
                                title="Loan Application Results")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="loan_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_info_page(model, feature_names):
    st.header("Model Information")
    
    # Model type
    st.subheader("Model Details")
    st.write(f"**Model Type:** {type(model).__name__}")
    st.write(f"**Number of Features:** {len(feature_names)}")
    st.write(f"**Features:** {', '.join(feature_names)}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(15), 
                    x='Importance', y='Feature', 
                    orientation='h',
                    title="Top 15 Most Important Features")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(importance_df)
    
    # Model parameters
    st.subheader("Model Parameters")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        st.dataframe(params_df)

def preprocess_single_input(data, label_encoders, scaler):
    """Preprocess single input for prediction"""
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna('Unknown', inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Encode categorical variables
    for col in data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            # Handle unknown categories
            le = label_encoders[col]
            data[col] = data[col].apply(lambda x, encoder=le: x if x in encoder.classes_ else encoder.classes_[0])
            data[col] = le.transform(data[col])
        else:
            # If encoder doesn't exist, use simple label encoding
            data[col] = pd.Categorical(data[col]).codes
    
    # Scale features
    data_scaled = scaler.transform(data)
    
    return data_scaled

def preprocess_batch_input(data, label_encoders, scaler, feature_names):
    """Preprocess batch input for prediction"""
    try:
        # Remove ID columns if present
        id_cols = ['Loan ID', 'Customer ID', 'ID', 'id']
        for col in id_cols:
            if col in data.columns:
                data = data.drop([col], axis=1)
        
        # Check if we have the required columns
        missing_cols = [col for col in feature_names if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Select only the required columns in the right order
        data = data[feature_names]
        
        # Handle missing values
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown', inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Encode categorical variables
        for col in data.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                # Handle unknown categories
                le = label_encoders[col]
                data[col] = data[col].apply(lambda x, encoder=le: x if x in encoder.classes_ else encoder.classes_[0])
                data[col] = le.transform(data[col])
            else:
                # If encoder doesn't exist, use simple label encoding
                data[col] = pd.Categorical(data[col]).codes
        
        # Scale features
        data_scaled = scaler.transform(data)
        
        return data_scaled
        
    except Exception as e:
        st.error(f"Error preprocessing batch data: {e}")
        return None

if __name__ == "__main__":
    main()
