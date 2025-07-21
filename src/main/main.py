import pandas as pd
from preprocessing import Preprocessor
from Feature_Extraction import TechnicalIndicatorExtractor
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
from Graphs import ResultPlotter

df = pd.read_csv('../../Data/AAPL.csv')
preprocessor = Preprocessor(alpha=0.2, ma_window=5, rsi_window=5, d=3)

# Preprocessing and feature extraction
processed_df = preprocessor.transform(df)
extractor = TechnicalIndicatorExtractor()
final_features_df = extractor.transform(processed_df)

# Prepare features and labels
X = final_features_df.drop(['Date', 'Target'], axis=1).values
y = ((final_features_df['Target'].values) == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# "Probability" using majority vote
def predict_proba_majority_vote(clf, X):
    votes = np.array([tree.predict(X) for tree in clf.trees])
    prob = votes.mean(axis=0)
    return prob

y_pred_prob = predict_proba_majority_vote(clf, X_test)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc:.4f}")

# ---- Visualization & Metrics ----
plotter = ResultPlotter()

# 1. Prediction vs Reality plot
plotter.plot_pred_vs_real(y_test, predictions, filename="prediction_vs_reality.png")

# 2. Prediction probability vs Reality plot
plotter.plot_pred_prob_vs_real(y_test, y_pred_prob, filename="prediction_prob_vs_reality.png")

# 3. Print Precision, Recall, Specificity, etc.
plotter.print_classification_metrics(y_test, predictions)

# 4. ROC Curve plot
plotter.plot_roc_curve(y_test, y_pred_prob, filename="roc_curve.png")

# 5. PCA Convex Hull for linear separability (on ALL DATA, not just train/test split)
ResultPlotter.plot_pca_convex_hull(X, y, folder="graphs", filename="pca_linear_separability.png")
