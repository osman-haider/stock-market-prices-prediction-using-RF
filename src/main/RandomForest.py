from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    """
    RandomForest is an ensemble machine learning algorithm that builds multiple decision trees
    and merges their predictions to improve the accuracy and control overfitting.
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        """
        Initialize the RandomForest with specified hyperparameters.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            min_samples_split (int): Minimum samples required to split a node.
            n_feature (int or None): Number of features to consider for each split. If None, use all.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        """
        Train the random forest by fitting several decision trees on bootstrapped samples of the data.

        Args:
            X (ndarray): Feature matrix for training data.
            y (ndarray): Target labels for training data.
        """
        self.trees = []  # Clear previous trees, if any
        for _ in range(self.n_trees):
            # Create a new decision tree with the specified parameters
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # Sample with replacement to create a bootstrap sample
            X_sample, y_sample = self._bootstrap_samples(X, y)
            # Fit the tree to the bootstrap sample
            tree.fit(X_sample, y_sample)
            # Store the trained tree
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Generate a bootstrap sample from the dataset (sampling with replacement).

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target labels.

        Returns:
            X_sample (ndarray): Bootstrapped features.
            y_sample (ndarray): Bootstrapped labels.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Find the most common label in an array (for majority voting).

        Args:
            y (array-like): Array of labels.

        Returns:
            Most frequent label in y.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Predict class labels for samples in X by aggregating predictions from all decision trees.

        Args:
            X (ndarray): Feature matrix for which to predict labels.

        Returns:
            predictions (ndarray): Predicted class labels for each sample.
        """
        # Get predictions from each tree for all samples
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Rearrange so each row contains predictions for one sample across all trees
        tree_preds = np.swapaxes(predictions, 0, 1)
        # Perform majority voting for each sample
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
