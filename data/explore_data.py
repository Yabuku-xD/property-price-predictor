import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def explore_data(data_path):
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    data = pd.read_csv(data_path, header=None, names=columns, delim_whitespace=True)

    print("Dataset Summary:")
    print(data.describe())

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    data.hist(figsize=(12, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(data["RM"], data["MEDV"], alpha=0.6)
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("Median Value of Homes (MEDV)")
    plt.title("RM vs MEDV")
    plt.show()

data_path = "data/housing.csv"
explore_data(data_path)