from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from svm import SVM

# Generate synthetic data
X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the SVM model with early stopping parameters
clf = SVM(lr=0.01, lamda=0.01, epochs=100, patience=5)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, zero_division=1))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title('Confusion Matrix')
plt.show()

# Visualization function for SVM decision boundary
def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.title('SVM Decision Boundary')

    # Decision boundary
    x_vals = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 100)
    y_vals = get_hyperplane_value(x_vals, clf.w, clf.b, 0)
    ax.plot(x_vals, y_vals, 'k', label='Decision Boundary')

    # Margins
    y_vals_plus = get_hyperplane_value(x_vals, clf.w, clf.b, 1)
    y_vals_minus = get_hyperplane_value(x_vals, clf.w, clf.b, -1)
    ax.plot(x_vals, y_vals_plus, 'k--', label='Margin +1')
    ax.plot(x_vals, y_vals_minus, 'k--', label='Margin -1')

    plt.legend(loc='best')
    plt.show()

# Visualize the SVM decision boundary
visualize_svm()
