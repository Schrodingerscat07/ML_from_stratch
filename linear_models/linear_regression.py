import numpy as np

class LinearRegressionScratch:
    """
    A from-scratch Linear Regression model designed for visualization.
    It stores the history of weights and biases during training.
    """
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None  # Initialize as None
        self.b = None  # Initialize as None
        
        # History lists for visualization
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.
        X and y should be NumPy arrays.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.epochs):
            # 1. Make a prediction
            y_pred = np.dot(X, self.w) + self.b
            
            # 2. Calculate the loss (cost)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            
            # 3. Store current parameters for visualization
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)

            # 4. Compute gradients (the "nudge")
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # 5. Update parameters (take a step downhill)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Optional: Print progress
            if i % 100 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Predicts target values for new data X.
        """
        return np.dot(X, self.w) + self.b