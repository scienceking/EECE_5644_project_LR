import matplotlib.pyplot as plt


folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
accuracies = [0.69, 0.69, 0.69, 0.69, 0.71]


plt.figure(figsize=(8, 6))
plt.bar(folds, accuracies, color='skyblue', edgecolor='black')
plt.ylim(0.65, 0.75)  #
plt.title("Five-Fold Cross Validation Accuracy", fontsize=16)
plt.xlabel("Folds", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()
