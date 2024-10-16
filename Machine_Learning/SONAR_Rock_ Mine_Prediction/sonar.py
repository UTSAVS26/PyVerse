import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('sonar_data.csv')

# Display the shape of the dataset
print(data.shape())

# Display information about the dataset
data.info()

# Check for missing values in the dataset
print(data.isnull().sum())

# Get statistical summary of the dataset
print(data.describe())

# Display the columns of the dataset
print(data.columns)

# Plot the count of the target variable (assuming it is at index 60)
sns.countplot(data[60])
plt.show()

# Calculate the mean of each feature grouped by the target variable
data.groupby(60).mean()

# Separate features and target variable
x = data.drop(60, axis=1)  # Features
y = data[60]                # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)             # Train the model
y_pred1 = lr.predict(x_test)         # Make predictions
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred1))  # Calculate accuracy

# K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)            # Train the model
y_pred2 = knn.predict(x_test)        # Make predictions
print("KNN Accuracy:", accuracy_score(y_test, y_pred2))  # Calculate accuracy

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)              # Train the model
y_pred3 = rf.predict(x_test)          # Make predictions
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred3))  # Calculate accuracy

# Stochastic Gradient Descent model
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()

# Train using partial_fit
for i in range(len(x_train)):  # Corrected from 'ramge' to 'range'
    sgd.partial_fit(x_train[i:i+1], y_train[i:i+1], classes=['R', 'M'])

# Calculate and print the score on the test set
score = sgd.score(x_test, y_test)
print("SGD Classifier Accuracy:", score)
