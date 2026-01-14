import numpy as np
from .decision_tree import DecisionTree

class XGBoostScratch:
    """
    A simplified XGBoost-style Gradient Boosting model from scratch.
    Focuses on regression (MSE loss) for simplicity, but can be adapted.
    """
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        # Initialize predictions with the mean of target values
        self.base_pred = np.mean(y)
        f_m = np.full(y.shape, self.base_pred)
        
        for _ in range(self.n_estimators):
            # Compute negative gradient (residuals for MSE)
            # Loss = 1/2 * (y - f_m)^2
            # Gradient = -(y - f_m)
            # Residual = y - f_m
            residuals = y - f_m
            
            # Fit a decision tree to the residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task='regression'
            )
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            f_m += self.learning_rate * predictions
            
            # Store the tree
            self.trees.append(tree)

    def predict(self, X):
        # Start with base prediction
        y_pred = np.full(X.shape[0], self.base_pred)
        
        # Add contributions from each tree
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
