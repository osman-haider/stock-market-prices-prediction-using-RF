import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ResultPlotter:
    def __init__(self, folder="graphs"):
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def plot_pred_vs_real(self, y_true, y_pred, filename="prediction_vs_reality.png"):
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, 'o-', label='Actual', alpha=0.7)
        plt.plot(y_pred, 'x-', label='Predicted', alpha=0.7)
        plt.title("Prediction vs Reality on Test Set")
        plt.xlabel("Sample Index")
        plt.ylabel("Class (0/1)")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.folder, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_pred_prob_vs_real(self, y_true, y_pred_prob, filename="prediction_prob_vs_reality.png"):
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, 'o-', label='Actual', alpha=0.7)
        plt.plot(y_pred_prob, 'x-', label='Predicted Probability', alpha=0.7)
        plt.title("Prediction Probability vs Reality on Test Set")
        plt.xlabel("Sample Index")
        plt.ylabel("Probability / Class")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.folder, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_oob_error(self, oob_errors, filename="oob_error.png"):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(oob_errors)+1), oob_errors, marker='o')
        plt.title("OOB Error vs. Number of Trees")
        plt.xlabel("Number of Trees")
        plt.ylabel("OOB Error")
        plt.grid()
        save_path = os.path.join(self.folder, filename)
        plt.savefig(save_path)
        plt.close()

    def print_classification_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")

    def plot_roc_curve(self, y_true, y_score, filename="roc_curve.png"):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        save_path = os.path.join(self.folder, filename)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_pca_convex_hull(X, y, folder="graphs", filename="pca_linear_separability.png"):
        from sklearn.decomposition import PCA
        from scipy.spatial import ConvexHull
        if not os.path.exists(folder):
            os.makedirs(folder)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))

        for label, color in zip([0, 1], ['orange', 'blue']):
            points = X_pca[y == label]
            plt.scatter(points[:, 0], points[:, 1], label=f'Class {label}', alpha=0.4)
            if len(points) >= 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color)
        plt.title("PCA Projection & Convex Hulls (Linear Separability)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path)
        plt.close()
