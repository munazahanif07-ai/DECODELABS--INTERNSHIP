"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Build a basic classification model using a small dataset (Iris).
Pipeline: Input (data + scaling) -> Process (train/test split + KNN) -> Output (confusion matrix + F1)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# STEP 1: INPUT - Load and understand the dataset
# ---------------------------------------------------------------------------
iris = load_iris()
X = iris.data                      # shape (150, 4) -> sepal_length, sepal_width, petal_length, petal_width
y = iris.target                    # shape (150,)   -> 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"Samples: {df.shape[0]}, Features: {X.shape[1]}, Classes: {len(target_names)}")
print(df.head())
print("\nClass balance:")
print(df["species"].value_counts())
print("\nSummary statistics:")
print(df.describe())

# ---------------------------------------------------------------------------
# STEP 2: PROCESS - Split into training and testing sets
# ---------------------------------------------------------------------------
# stratify=y keeps the 3 classes balanced in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print("\n" + "=" * 60)
print("STEP 2: TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# STEP 3: PROCESS - Feature scaling (the "Gatekeeper Rule" from your slides)
# ---------------------------------------------------------------------------
# Fit the scaler ONLY on training data, then apply it to both sets.
# This avoids data leakage from the test set into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING")
print("=" * 60)
print(f"Post-scaling train mean (~0): {X_train_scaled.mean(axis=0).round(3)}")
print(f"Post-scaling train std  (~1): {X_train_scaled.std(axis=0).round(3)}")

# ---------------------------------------------------------------------------
# STEP 4: Tune K - find the "elbow" (optimal K) using error rate
# ---------------------------------------------------------------------------
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(np.mean(pred_temp != y_test))

best_k = k_range[int(np.argmin(error_rates))]

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), error_rates, marker="o", linestyle="dashed", color="steelblue")
plt.axvline(best_k, color="orange", linestyle="--", label=f"Optimal K = {best_k}")
plt.title("Tuning the Engine: Choosing K")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/k_tuning_curve.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("STEP 4: CHOOSING K")
print("=" * 60)
print(f"Optimal K found: {best_k} (lowest error rate = {min(error_rates):.4f})")

# ---------------------------------------------------------------------------
# STEP 5: PROCESS - Instantiate, fit, predict (scikit-learn workflow)
# ---------------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train_scaled, y_train)                 # FIT (memorize the map)
predictions = model.predict(X_test_scaled)         # PREDICT (apply logic)

# ---------------------------------------------------------------------------
# STEP 6: OUTPUT - Validation (confusion matrix, F1 score, accuracy)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, predictions)
acc = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average="macro")
report = classification_report(y_test, predictions, target_names=target_names)

print("\n" + "=" * 60)
print("STEP 6: OUTPUT VALIDATION")
print("=" * 60)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro avg): {f1_macro:.4f}")
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=[f"true_{t}" for t in target_names],
                    columns=[f"pred_{t}" for t in target_names]))
print("\nFull classification report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Iris KNN Classifier")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# STEP 7: Sanity check - predict on a brand-new, unseen flower
# ---------------------------------------------------------------------------
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])   # looks like a setosa
new_flower_scaled = scaler.transform(new_flower)
new_pred = model.predict(new_flower_scaled)

print("\n" + "=" * 60)
print("STEP 7: PREDICTION ON NEW DATA")
print("=" * 60)
print(f"New sample {new_flower.tolist()} -> Predicted species: {target_names[new_pred[0]]}")

print("\nDone. Plots saved: k_tuning_curve.png, confusion_matrix.png")