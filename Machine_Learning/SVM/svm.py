import numpy as np

class SVM:
    def __init__(self, lr=0.01, lamda=0.01, epochs=100, patience=5):
        """ 
        Params:
        lr : Learning rate
        lamda : Regularization parameter
        epochs: Maximum number of epochs
        patience: Number of epochs with no improvement before stopping early
        """
        self.lr = lr
        self.lamda = lamda
        self.epochs = epochs
        self.patience = patience
        self.w = None  # Weights
        self.b = None  # Bias

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        best_loss = float('inf')
        patience_counter = 0

        # Gradient descent with early stopping
        for epoch in range(self.epochs):
            loss = 0
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lamda * self.w)
                else:
                    self.w -= self.lr * (2 * self.lamda * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    loss += max(0, 1 - y_[idx] * (np.dot(x_i, self.w) + self.b))

            # Display weights and bias every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
                print(f"Weights: {self.w}, Bias: {self.b}")

            # Early stopping logic
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        # Convert -1 back to 0 for label compatibility
        return np.where(np.sign(linear_output) == -1, 0, 1)

    def save_model(self, filename='svm_model.npz'):
        np.savez(filename, w=self.w, b=self.b)

    def load_model(self, filename='svm_model.npz'):
        data = np.load(filename)
        self.w = data['w']
        self.b = data['b']
