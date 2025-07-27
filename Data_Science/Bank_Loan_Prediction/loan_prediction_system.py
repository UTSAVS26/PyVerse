# Bank Loan Eligibility Prediction System
# Complete implementation with data preprocessing, model training, and web interface

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LoanPredictionSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = None
        
    def create_sample_data(self):
        """Create sample data for demonstration purposes"""
        print("Creating sample training and testing data...")
        
        # Set random seed for reproducible sample data
        np.random.seed(42)
        
        # Create sample training data
        n_train = 1000
        n_test = 200
        
        # Generate synthetic loan data
        def generate_loan_data(n_samples, include_target=True):
            data = {}
            
            # Generate data for all required features
            data['Current Loan Amount'] = np.random.lognormal(9, 0.8, n_samples).astype(int)
            data['Current Loan Amount'] = np.clip(data['Current Loan Amount'], 1000, 500000)
            
            # Term
            terms = ['Short Term', 'Long Term']
            data['Term'] = np.random.choice(terms, n_samples, p=[0.3, 0.7])
            
            # Credit scores (300-850)
            data['Credit Score'] = np.random.normal(650, 100, n_samples).astype(int)
            data['Credit Score'] = np.clip(data['Credit Score'], 300, 850)
            
            # Annual income (20k-200k)
            data['Annual Income'] = np.random.lognormal(10.5, 0.5, n_samples).astype(int)
            data['Annual Income'] = np.clip(data['Annual Income'], 20000, 200000)
            
            # Years in current job
            job_years = ['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
            data['Years in current job'] = np.random.choice(job_years, n_samples)
            
            # Home ownership
            home_ownership = ['Rent', 'Own Home', 'Mortgage', 'HaveMortgage']
            data['Home Ownership'] = np.random.choice(home_ownership, n_samples, p=[0.3, 0.3, 0.3, 0.1])
            
            # Purpose
            purposes = ['Debt Consolidation', 'Home Improvements', 'Other', 'Buy House', 
                       'Business Loan', 'Buy a Car', 'major_purchase', 'Medical Bills',
                       'Educational Expenses', 'Vacation', 'Wedding', 'Moving', 'Renewable Energy']
            data['Purpose'] = np.random.choice(purposes, n_samples)
            
            # Monthly debt
            data['Monthly Debt'] = (data['Annual Income'] * np.random.uniform(0.1, 0.4, n_samples) / 12).astype(int)
            
            # Years of credit history (1-50)
            data['Years of Credit History'] = np.random.exponential(10, n_samples).astype(int)
            data['Years of Credit History'] = np.clip(data['Years of Credit History'], 1, 50)
            
            # Months since last delinquent
            data['Months since last delinquent'] = np.random.exponential(24, n_samples).astype(int)
            data['Months since last delinquent'] = np.clip(data['Months since last delinquent'], 0, 200)
            
            # Number of open accounts
            data['Number of Open Accounts'] = np.random.poisson(8, n_samples)
            data['Number of Open Accounts'] = np.clip(data['Number of Open Accounts'], 1, 30)
            
            # Number of credit problems
            data['Number of Credit Problems'] = np.random.poisson(1, n_samples)
            data['Number of Credit Problems'] = np.clip(data['Number of Credit Problems'], 0, 10)
            
            # Current credit balance
            data['Current Credit Balance'] = (data['Annual Income'] * np.random.uniform(0.05, 0.3, n_samples)).astype(int)
            
            # Maximum open credit
            data['Maximum Open Credit'] = (data['Current Credit Balance'] * np.random.uniform(1.2, 3.0, n_samples)).astype(int)
            
            # Bankruptcies
            data['Bankruptcies'] = np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03])
            
            # Tax liens
            data['Tax Liens'] = np.random.choice([0, 1, 2], n_samples, p=[0.90, 0.08, 0.02])
            
            # Create target variable based on features (if training data)
            if include_target:
                # Create loan status based on credit score and other factors
                credit_score_norm = (data['Credit Score'] - 300) / 550
                income_norm = np.log(data['Annual Income']) / np.log(200000)
                debt_ratio = data['Monthly Debt'] * 12 / data['Annual Income']
                
                # Calculate approval probability
                approval_prob = (
                    0.4 * credit_score_norm +
                    0.3 * income_norm +
                    0.2 * (1 - debt_ratio) +
                    0.1 * (1 - np.array(data['Number of Credit Problems']) / 10)
                )
                
                # Add some randomness
                approval_prob += np.random.normal(0, 0.1, n_samples)
                approval_prob = np.clip(approval_prob, 0, 1)
                
                # Convert to loan status
                loan_status = np.where(approval_prob > 0.5, 'Fully Paid', 'Charged Off')
                data['Loan Status'] = loan_status
            
            return pd.DataFrame(data)
        
        # Generate training and test data
        self.train_data = generate_loan_data(n_train, include_target=True)
        self.test_data = generate_loan_data(n_test, include_target=False)
        
        # Save sample data
        self.train_data.to_csv('credit_train.csv', index=False)
        self.test_data.to_csv('credit_test.csv', index=False)
        
        print(f"Sample training data created: {self.train_data.shape}")
        print(f"Sample testing data created: {self.test_data.shape}")
        print("Data saved as 'credit_train.csv' and 'credit_test.csv'")
        
        return self.train_data, self.test_data
    
    def load_data(self, train_path=None, test_path=None):
        """Load training and testing datasets"""
        print("Loading datasets...")
        
        try:
            if train_path and test_path:
                self.train_data = pd.read_csv(train_path)
                self.test_data = pd.read_csv(test_path)
            else:
                # Try to load existing files
                if os.path.exists('credit_train.csv') and os.path.exists('credit_test.csv'):
                    self.train_data = pd.read_csv('credit_train.csv')
                    self.test_data = pd.read_csv('credit_test.csv')
                else:
                    raise FileNotFoundError("Data files not found")
        except FileNotFoundError:
            print("Data files not found. Creating sample data...")
            return self.create_sample_data()
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Testing data shape: {self.test_data.shape}")
        return self.train_data, self.test_data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print("\nTraining Data Info:")
        print(self.train_data.info())
        
        print("\nTraining Data Description:")
        print(self.train_data.describe())
        
        print("\nMissing Values in Training Data:")
        print(self.train_data.isnull().sum())
        
        print("\nLoan Status Distribution:")
        print(self.train_data['Loan Status'].value_counts())
        
        # Save basic statistics
        try:
            # Create a simple analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loan Status distribution
            self.train_data['Loan Status'].value_counts().plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Loan Status Distribution')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Credit Score distribution
            axes[0,1].hist(self.train_data['Credit Score'].dropna(), bins=30, edgecolor='black')
            axes[0,1].set_title('Credit Score Distribution')
            
            # Annual Income distribution
            axes[1,0].hist(self.train_data['Annual Income'].dropna(), bins=30, edgecolor='black')
            axes[1,0].set_title('Annual Income Distribution')
            
            # Current Loan Amount distribution
            axes[1,1].hist(self.train_data['Current Loan Amount'].dropna(), bins=30, edgecolor='black')
            axes[1,1].set_title('Current Loan Amount Distribution')
            
            plt.tight_layout()
            plt.savefig('loan_data_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Data analysis plots saved as 'loan_data_analysis.png'")
            
        except Exception as e:
            print(f"Could not create plots: {e}")
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Work with copies to avoid modifying original data
        train_copy = self.train_data.copy()
        test_copy = self.test_data.copy()
        
        # Separate target variable
        y = train_copy['Loan Status']
        X_train = train_copy.drop(['Loan Status'], axis=1)
        X_test = test_copy
        
        # Combine for preprocessing
        combined_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        train_len = len(X_train)
        
        print(f"Combined data shape: {combined_data.shape}")
        print("\nMissing values before preprocessing:")
        missing_before = combined_data.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Handle missing values
        # Numerical columns
        numerical_cols = combined_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if combined_data[col].isnull().sum() > 0:
                combined_data[col].fillna(combined_data[col].median(), inplace=True)
        
        # Categorical columns
        categorical_cols = combined_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if combined_data[col].isnull().sum() > 0:
                mode_val = combined_data[col].mode()
                if len(mode_val) > 0:
                    combined_data[col].fillna(mode_val[0], inplace=True)
                else:
                    combined_data[col].fillna('Unknown', inplace=True)
        
        print("\nMissing values after preprocessing:")
        missing_after = combined_data.isnull().sum()
        print(missing_after[missing_after > 0])
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        
        # Split back to train and test
        X_train_processed = combined_data[:train_len]
        X_test_processed = combined_data[train_len:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        X_test_scaled = self.scaler.transform(X_test_processed)
        
        # Store feature names
        self.feature_names = X_train_processed.columns.tolist()
        
        print(f"\nProcessed training data shape: {X_train_scaled.shape}")
        print(f"Processed testing data shape: {X_test_scaled.shape}")
        print(f"Feature names: {self.feature_names}")
        print(f"Target classes: {le_target.classes_}")
        
        return X_train_scaled, X_test_scaled, y_encoded
    
    def train_models(self, X_train, y_train):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training split: {X_train_split.shape}")
        print(f"Validation split: {X_val_split.shape}")
        
        # Define models with better parameters
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=6)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                
                # Train model
                model.fit(X_train_split, y_train_split)
                
                # Make predictions
                y_pred = model.predict(X_val_split)
                y_pred_proba = model.predict_proba(X_val_split)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val_split, y_pred)
                precision = precision_score(y_val_split, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val_split, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val_split, y_pred, average='weighted', zero_division=0)
                
                # Handle ROC-AUC for multi-class
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_val_split, y_pred_proba[:, 1])
                else:
                    try:
                        roc_auc = roc_auc_score(y_val_split, y_pred_proba, multi_class='ovr', average='weighted')
                    except (ValueError, TypeError):
                        roc_auc = 0.0
                   
                results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'ROC-AUC': roc_auc
                }
                
                # Store model
                self.models[name] = model
                
                print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        if not results:
            raise Exception("No models were successfully trained!")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).T
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(results_df.round(4))
        
        # Find best model based on F1-Score
        best_model_name = results_df['F1-Score'].idxmax()
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🏆 Best F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")
        
        return results_df, best_model_name
    
    def evaluate_final_model(self, X_train, y_train):
        """Evaluate the final model with detailed metrics"""
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(self.best_model, X_train, y_train, cv=5, scoring='f1_weighted')
            print(f"Cross-validation F1-scores: {cv_scores}")
            print(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train final model on full training data
            self.best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.best_model.predict(X_train)
            
            # Print classification report
            print("\nClassification Report:")
            target_names = self.label_encoders['target'].classes_
            print(classification_report(y_train, y_pred, target_names=target_names))
            
            # Feature importance (if available)
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10))
                
        except Exception as e:
            print(f"Error in model evaluation: {e}")
    
    def make_predictions(self, X_test):
        """Make predictions on test data"""
        print("\n" + "="*50)
        print("MAKING PREDICTIONS")
        print("="*50)
        
        try:
            # Make predictions
            predictions = self.best_model.predict(X_test)
            prediction_probabilities = self.best_model.predict_proba(X_test)
            
            # Convert predictions back to original labels
            predictions_original = self.label_encoders['target'].inverse_transform(predictions)
            
            # Handle probability extraction for binary/multi-class
            n_classes = len(self.label_encoders['target'].classes_)
            if n_classes == 2:
                # For binary classification, get probability of positive class
                prob_positive = prediction_probabilities[:, 1]
            else:
                # For multi-class, get max probability
                prob_positive = np.max(prediction_probabilities, axis=1)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Predicted_Status': predictions_original,
                'Prediction_Probability': prob_positive
            })
            
            print(f"✅ Predictions made for {len(results)} loan applications")
            print(f"Prediction distribution:")
            print(pd.Series(predictions_original).value_counts())
            
            return results
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def save_model(self, filename='loan_prediction_model.pkl'):
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"\n💾 Model saved as {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filename='loan_prediction_model.pkl'):
        """Load a saved model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            
            print(f"✅ Model loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main function to execute the loan prediction system"""
    print("="*60)
    print("🏦 BANK LOAN ELIGIBILITY PREDICTION SYSTEM")
    print("="*60)
    
    try:
        # Initialize the system
        loan_system = LoanPredictionSystem()
        
        # Load data
        print("\n📊 STEP 1: Loading Data...")
        train_data, test_data = loan_system.load_data()
        
        # Explore data
        print("\n🔍 STEP 2: Exploring Data...")
        loan_system.explore_data()
        
        # Preprocess data
        print("\n⚙️ STEP 3: Preprocessing Data...")
        X_train, X_test, y_train = loan_system.preprocess_data()
        
        # Train models
        print("\n🤖 STEP 4: Training Models...")
        results_df, best_model_name = loan_system.train_models(X_train, y_train)
        
        # Evaluate final model
        print("\n📈 STEP 5: Evaluating Final Model...")
        loan_system.evaluate_final_model(X_train, y_train)
        
        # Make predictions
        print("\n🎯 STEP 6: Making Predictions...")
        predictions = loan_system.make_predictions(X_test)
        
        if predictions is not None:
            # Save predictions
            predictions.to_csv('loan_predictions.csv', index=False)
            print("📁 Predictions saved to 'loan_predictions.csv'")
        
        # Save model
        print("\n💾 STEP 7: Saving Model...")
        success = loan_system.save_model()
        
        if success:
            print("\n" + "="*60)
            print("🎉 LOAN PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\n✅ Model file 'loan_prediction_model.pkl' has been created.")
            print("✅ You can now run the web app using:")
            print("   streamlit run loan_web_app.py")
            print("\n📊 Files created:")
            print("   - loan_prediction_model.pkl (trained model)")
            print("   - credit_train.csv (training data)")
            print("   - credit_test.csv (testing data)")
            print("   - loan_predictions.csv (predictions)")
            print("   - loan_data_analysis.png (data analysis plots)")
        else:
            print("\n❌ Error: Model could not be saved!")
        
        return loan_system, predictions
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        print("\nTrying to create and save a basic model...")
        
        # Fallback: create minimal working model
        try:
            loan_system = LoanPredictionSystem()
            train_data, test_data = loan_system.create_sample_data()
            X_train, X_test, y_train = loan_system.preprocess_data()
            
            # Train just Random Forest as fallback
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42, n_estimators=50)
            model.fit(X_train, y_train)
            loan_system.best_model = model
            
            # Save the model
            success = loan_system.save_model()
            if success:
                print("✅ Fallback model created and saved successfully!")
                print("✅ You can now run: streamlit run loan_web_app.py")
            
            return loan_system, None
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return None, None

# Run the main function
if __name__ == "__main__":
    loan_system, predictions = main()