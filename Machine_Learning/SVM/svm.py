import numpy as np

class SVM:
    def __init__(self, lr=0.01, lamda=0.01, epochs=100):
        """ 
        params:
        lr : Learning rate
        lamda : For regularization
        epochs: Epochs(number of times loop will during training)
        """
        self.lr = lr
        self.lamda = lamda
        self.epochs = epochs
        #w - Weights
        self.w = None
        #b - Bias
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initializing weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Converting labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Gradient descent optimization
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lamda * self.w)
                else:
                    self.w -= self.lr * (2 * self.lamda * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            if _ % 10 == 0:
                print(f"After epoch:{_} value of weights and bias is:")
                print(self.w)
                print(self.b)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        # Convert -1 back to 0 for compatibility with the original labels
        return np.where(np.sign(linear_output) == -1, 0, 1)