# AI-Powered Property Price Predictor
This project focuses on algorithm optimization and machine learning model deployment to predict housing prices in Boston using a Random Forest Regression model. The project aims to predict house prices based on various input features like crime rate, property tax, and number of rooms.

I implemented algorithm optimization techniques such as hyperparameter tuning and feature selection to improve model performance. The trained model is deployed as a web application using Flask to predict housing prices in real-time.

### Project Overview
This project uses a Random Forest Regression model to predict housing prices based on the famous Boston Housing Dataset. The data includes various features such as the crime rate, property tax rates, average number of rooms, and more. By predicting the housing prices accurately, the goal is to provide an efficient tool for real estate professionals.

### Key Steps in the Project:
- Algorithm Optimization: Tuning the Random Forest model to achieve the best performance.
- Feature Engineering & Selection: Selecting key features to improve model prediction and reduce overfitting.
- Model Evaluation & Deployment: Evaluating the model performance and deploying it on the cloud.

### Algorithm Optimization
- Hyperparameter Tuning:
1. Random Forest Model was optimized using Grid Search and Bayesian Optimization.
2. Tuning parameters like number of estimators, max depth, and min samples split helped improve accuracy.
- Feature Selection:
1. Performed feature selection to identify the most important features using Random Forest feature importance.
2. Reduced overfitting by using only the most impactful features, resulting in improved model performance.

### Final Model Performance:
- Mean Squared Error (MSE): 8.35
- Mean Absolute Error (MAE): 1.93
- R-squared (R²): 0.89

### Technologies Used
- Flask: Python web framework for deploying the model as a web app.
- Random Forest Regression: Machine learning model used for prediction.
- Scikit-learn: Python library used for machine learning tasks (model building, hyperparameter tuning, etc.).
- Joblib: Used to save and load the trained model.
- HTML/CSS: For frontend development.
- Render: For deploying the app on the cloud. 
