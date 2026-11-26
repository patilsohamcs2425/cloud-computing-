# Step 1: Import Libraries

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Libraries imported successfully")


# Step 2: Download and Load Dataset

path = kagglehub.dataset_download("uciml/iris")
df = pd.read_csv(path + "/Iris.csv")

print("\nComplete Dataset:")
display(df)


# Step 3: Dataset Exploration

print("\nFirst 5 Records:")
display(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()


# Step 4: Statistical Summary

print("\nStatistical Summary:")
display(df.describe())


# Step 5: Check Missing and Duplicate Values

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# Step 6: Flower Species Distribution

print("\nSpecies Distribution:")
print(df["Species"].value_counts())

sns.countplot(x="Species", data=df)
plt.title("Flower Species Distribution")
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")
plt.xticks(rotation=15)
plt.show()


# Step 7: Sepal Length and Width Visualization

sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm",
                hue="Species", data=df)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()


# Step 8: Petal Length and Width Visualization

sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm",
                hue="Species", data=df)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


# Step 9: Pair Plot

sns.pairplot(df.drop(columns="Id"), hue="Species")
plt.show()


# Step 10: Correlation Heatmap

corr = df.drop(columns=["Id", "Species"]).corr()

print("\nCorrelation Matrix:")
display(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Step 11: Separate Input and Target Variables

X = df.drop(columns=["Id", "Species"])
y = df["Species"]

print("\nInput Features:")
display(X.head())

print("\nTarget Variable:")
display(y.head())


# Step 12: Split Training and Testing Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nFirst 5 Testing Samples:")
display(X_test.head())


# Step 13: Manual Euclidean Distance for First 5 Test Samples

for i in range(5):

    sample = X_test.iloc[i]

    distance = np.sqrt(((X_train - sample)**2).sum(axis=1))

    result = pd.DataFrame({
        "Distance": distance,
        "Species": y_train
    })

    result = result.sort_values("Distance")

    print("\nTest Sample:", i + 1)
    print("Measurements:", sample.values)
    print("Actual Species:", y_test.iloc[i])

    print("7 Nearest Neighbors:")
    display(result.head(7))

    for k in [3, 5, 7]:
        print("K =", k, "Prediction:",
              result.head(k)["Species"].mode()[0])


# Step 14: Implement KNN with K = 3, 5 and 7

accuracies = []
predictions = {}

for k in [3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred) * 100

    accuracies.append(accuracy)
    predictions[k] = pred

    print("\nK Value:", k)
    print("Training Accuracy:", round(model.score(X_train, y_train)*100, 2), "%")
    print("Testing Accuracy:", round(accuracy, 2), "%")


# Step 15: Actual vs Predicted Species

result = pd.DataFrame({
    "Actual": y_test.values,
    "K=3": predictions[3],
    "K=5": predictions[5],
    "K=7": predictions[7]
})

print("\nActual vs Predicted Species:")
display(result)


# Step 16: Accuracy Comparison Table

result = pd.DataFrame({
    "K Value": [3, 5, 7],
    "Accuracy (%)": accuracies
})

print("\nAccuracy Comparison:")
display(result.round(2))


# Step 17: Accuracy Comparison Graph

plt.bar(["K=3", "K=5", "K=7"], accuracies,
        color=["skyblue", "orange", "green"])

plt.title("KNN Accuracy Comparison")
plt.xlabel("K Values")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)

for i in range(3):
    plt.text(i, accuracies[i] + 1,
             str(round(accuracies[i], 2)) + "%",
             ha="center")

plt.show()


# Step 18: Confusion Matrix

for k in [3, 5, 7]:

    cm = confusion_matrix(y_test, predictions[k])

    print("\nConfusion Matrix for K =", k)
    print(cm)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)

    plt.title("Confusion Matrix for K = " + str(k))
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.show()


# Step 19: Classification Report

for k in [3, 5, 7]:

    print("\nClassification Report for K =", k)
    print(classification_report(y_test, predictions[k]))


# Step 20: Final Result

print("\nFINAL RESULT")

print("Total Dataset Samples:", len(df))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("Distance Metric: Euclidean Distance")
print("Manual Test Samples: 5")
print("Manually Calculated Distances:", 5 * len(X_train))

for k, acc in zip([3, 5, 7], accuracies):
    print("Accuracy for K =", k, ":", round(acc, 2), "%")

best_k = [k for k, acc in zip([3, 5, 7], accuracies)
          if acc == max(accuracies)]

print("Best Observed K Value(s):", best_k)
print("Highest Accuracy:", round(max(accuracies), 2), "%")