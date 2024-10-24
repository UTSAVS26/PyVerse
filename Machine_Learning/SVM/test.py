from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler
from svm import SVM
import matplotlib.pyplot as plt
import numpy as np

# Generating Synthetic data
X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into training part and testing part
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.25,
                                                 random_state=42)

# Object of our SVM class
clf = SVM()

# Training model on the training data
clf.fit(X_train,y_train)

# Collecting Predictions
y_pred = clf.predict(X_test) 

print("Accuracy is : ", accuracy_score(y_pred,y_test))
print("Precision is : ", precision_score(y_pred,y_test))
print("Recall is : ", recall_score(y_pred,y_test,zero_division=1))

def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    # Decision boundary
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    # Margin lines
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 1)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

    plt.show()

visualize_svm()
