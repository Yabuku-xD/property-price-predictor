import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    importance_df = pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    })

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance Analysis")
    plt.gca().invert_yaxis()
    plt.show()
    
    return importance_df