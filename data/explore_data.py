import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def explore_data(data_path):
    # Define column names
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    
    # Load the dataset
    data = pd.read_csv(data_path, header=None, names=columns, delim_whitespace=True)
    
    # Summary statistics
    print("Dataset Summary:")
    print(data.describe())
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
    # Feature distributions
    data.hist(figsize=(12, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()
    
    # Scatter plot: RM vs. MEDV
    plt.figure(figsize=(6, 4))
    plt.scatter(data["RM"], data["MEDV"], alpha=0.6)
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("Median Value of Homes (MEDV)")
    plt.title("RM vs MEDV")
    plt.show()

# Define the path to the dataset
data_path = "data/housing.csv"

# Perform EDA
explore_data(data_path)