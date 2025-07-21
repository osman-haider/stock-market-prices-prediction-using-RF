import numpy as np
from collections import Counter

class Node:
    """
    Node class represents a single node in the Decision Tree.

    It can be either:
    - An internal node: contains the feature index and threshold for splitting, and links to left/right child nodes.
    - A leaf node: contains the predicted class label (value).
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initialize a Node.

        Parameters:
        - feature (int): Index of the feature this node splits on (None for leaf).
        - threshold (float): Threshold value for the split (None for leaf).
        - left (Node): Left child node (None for leaf).
        - right (Node): Right child node (None for leaf).
        - value (int): Class label for a leaf node (None for internal).
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        - True if this is a leaf node (contains a value), False otherwise.
        """
        return self.value is not None

class DecisionTree:
    """
    DecisionTree class implements a Decision Tree Classifier from scratch.

    Core methods handle training (fit), prediction, and internal recursive tree construction
    based on information gain and entropy.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        Initialize the Decision Tree.

        Parameters:
        - min_samples_split (int): Minimum samples required to split an internal node.
        - max_depth (int): Maximum depth the tree can grow.
        - n_features (int or None): Number of features to consider when looking for the best split (default: all).
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the Decision Tree classifier to the training data.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Target labels of shape (n_samples,).
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree by finding the best splits.

        Parameters:
        - X (numpy.ndarray): Feature matrix at current recursion level.
        - y (numpy.ndarray): Target labels at current recursion level.
        - depth (int): Current depth of the tree.

        Returns:
        - Node: The root node of the (sub)tree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: max depth, pure node, or too few samples.
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select features for split (if n_features < total features).
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split among selected features.
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Partition the data and recursively build left/right branches.
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold to split the data for maximum information gain.

        Parameters:
        - X (numpy.ndarray): Feature matrix.
        - y (numpy.ndarray): Target labels.
        - feat_idxs (array-like): Indices of features to consider.

        Returns:
        - split_idx (int): Index of the best feature to split.
        - split_threshold (float): Best threshold value for the split.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        # Evaluate all candidate splits and pick the one with highest information gain.
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """
        Compute the information gain from splitting on the given feature/threshold.

        Parameters:
        - y (numpy.ndarray): Target labels.
        - X_column (numpy.ndarray): Single feature column to split on.
        - threshold (float): Threshold value to split at.

        Returns:
        - information_gain (float): Information gain resulting from the split.
        """
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If split doesn't divide the data, gain is zero.
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted average entropy of the children nodes.
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        """
        Split the data indices based on a given feature column and threshold.

        Parameters:
        - X_column (numpy.ndarray): Feature column.
        - split_thresh (float): Threshold for the split.

        Returns:
        - left_idxs (numpy.ndarray): Indices where feature <= threshold.
        - right_idxs (numpy.ndarray): Indices where feature > threshold.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy (impurity) of a set of labels.

        Parameters:
        - y (numpy.ndarray): Target labels.

        Returns:
        - entropy (float): Entropy value.
        """
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Find the most common class label in a set of labels.

        Parameters:
        - y (numpy.ndarray): Target labels.

        Returns:
        - value (int): Most frequent label.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predict class labels for input samples X.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        - predictions (numpy.ndarray): Predicted labels for each sample.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to predict the label for a single sample.

        Parameters:
        - x (numpy.ndarray): Feature values of a single sample.
        - node (Node): Current node in the tree.

        Returns:
        - value (int): Predicted label at the leaf node.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
