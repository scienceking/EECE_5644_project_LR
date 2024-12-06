import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set dataset folder paths
folder_melanamo = "Melanoma"
folder_nevus = "Melanocytic nevus"

# Define image size
IMG_SIZE = 256

# Function to load images from a folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to IMG_SIZE x IMG_SIZE
            images.append(img)
            labels.append(label)
    return images, labels

# Load data from both folders
images_melanamo, labels_melanamo = load_images_from_folder(folder_melanamo, 0)
images_nevus, labels_nevus = load_images_from_folder(folder_nevus, 1)

# Combine data and labels
images = np.array(images_melanamo + images_nevus)  # Original image data
X = np.array([img.flatten() for img in images])  # Flatten images into 1D vectors
y = np.array(labels_melanamo + labels_nevus)  # Label data

# Normalize data
X = X / 255.0  # Scale pixel values to [0, 1]

# Define 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1  # Track the current fold

# Variables to record overall performance
accuracies = []
all_cm = np.zeros((2, 2))  # Initialize confusion matrix

# Start 5-fold cross-validation
for train_index, test_index in kf.split(X, y):
    print(f"\n--- Fold {fold} ---")

    # Get training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create logistic regression model
    model = LogisticRegression(max_iter=6000, solver='lbfgs')
    model.fit(X_train, y_train)  # Train the model

    # Test the model and output results
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold} Accuracy: {acc:.2f}")
    accuracies.append(acc)

    # Compute and accumulate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    all_cm += cm

    # Classification report
    print(classification_report(y_test, y_pred, target_names=["Melanamo", "Nevus"]))

    fold += 1

# Output average performance
print("\n--- 5-Fold Cross-Validation Results ---")
print(f"Average Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")

# Average confusion matrix
print("\nAverage Confusion Matrix:")
print(all_cm / 5)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=all_cm / 5, display_labels=["Melanamo", "Nevus"])
disp.plot(cmap='Blues')
plt.title("Average Confusion Matrix (5-Fold CV)")
plt.show()
