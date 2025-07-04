import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_breast_cancer.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_svm = SVC(kernel="linear", C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca = train_test_split(X_pca, test_size=0.2, random_state=42)

linear_svm_pca = SVC(kernel="linear", C=1)
linear_svm_pca.fit(X_train_pca, y_train)

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1]
}
rbf_svm = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
rbf_svm.fit(X_train, y_train)
best_rbf = rbf_svm.best_estimator_

best_rbf_pca = SVC(kernel="rbf", C=rbf_svm.best_params_["C"], gamma=rbf_svm.best_params_["gamma"])
best_rbf_pca.fit(X_train_pca, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=20)
    ax.set_title(title)

plot_decision_boundary(linear_svm_pca, X_pca, y, axes[0], "Linear SVM Decision Boundary")
plot_decision_boundary(best_rbf_pca, X_pca, y, axes[1], "RBF SVM Decision Boundary")

plt.tight_layout()
plt.show()

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Linear SVM Cross-Val Score:", np.mean(cross_val_score(linear_svm, X, y, cv=5)))
print("RBF SVM Accuracy:", accuracy_score(y_test, best_rbf.predict(X_test)))
print("RBF SVM Cross-Val Score:", np.mean(cross_val_score(best_rbf, X, y, cv=5)))
