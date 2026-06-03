# Credit Card Fraud Detection using XGBoost and SMOTE

## Overview
Credit card fraud detection is a critical problem in the financial industry, where fraudulent transactions account for only a very small percentage of total transactions. This extreme class imbalance makes fraud detection a challenging machine learning task.
This project presents an end-to-end machine learning pipeline for identifying fraudulent credit card transactions. The solution combines data preprocessing, exploratory data analysis, class imbalance handling using SMOTE, hyperparameter optimization with RandomizedSearchCV, and fraud classification using XGBoost.
The primary goal is to maximize fraud detection performance while minimizing missed fraudulent transactions.

## Dataset
The project uses the Credit Card Fraud Detection dataset containing transactions made by European cardholders over a two-day period.
The dataset consists of 284,807 transactions, of which only 492 are fraudulent. This means that fraudulent transactions represent less than 0.2% of the data, making the problem highly imbalanced.
To protect user privacy, most features have been transformed using Principal Component Analysis (PCA). The dataset contains twenty-eight anonymized features labeled V1 through V28, along with the transaction amount, transaction time, and the target variable indicating whether a transaction is fraudulent.
Dataset Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Workflow

### Data Exploration
The dataset was first explored to understand its structure, feature distributions, and class imbalance. Visualizations such as class distribution plots, transaction amount boxplots, and correlation heatmaps were used to identify patterns within the data.

### Data Preprocessing
The dataset was separated into features and target labels. Numerical features were standardized using StandardScaler to ensure consistent feature scaling before model training.

### Handling Class Imbalance
Since fraudulent transactions are extremely rare, SMOTE (Synthetic Minority Oversampling Technique) was applied to generate synthetic fraud samples within the training set. This allowed the model to learn fraud patterns more effectively without simply duplicating existing samples.

### Hyperparameter Optimization
RandomizedSearchCV was used to search for optimal XGBoost hyperparameters through cross-validation. This approach provides an efficient alternative to exhaustive grid search while still exploring a wide range of parameter combinations.

### Model Training
An XGBoost classifier was trained on the balanced dataset. XGBoost was selected because of its strong performance on tabular datasets, ability to capture complex patterns, and robustness in classification tasks involving imbalanced data.

### Threshold Optimization
Instead of using the default classification threshold of 0.5, a custom threshold of 0.3 was applied to predicted probabilities. This adjustment improves the model's ability to detect fraudulent transactions by increasing recall while maintaining acceptable precision.

### Model Evaluation
The model was evaluated using multiple metrics including Precision, Recall, F1-Score, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Confusion Matrix, and ROC-AUC. These metrics provide a more meaningful assessment than accuracy alone for highly imbalanced datasets.

## Results
The final model achieved an accuracy of approximately 99.94%, a precision of 80%, and a recall of 85.7%. The F1-score reached 82.8%, while the Matthews Correlation Coefficient exceeded 0.82, indicating strong predictive performance despite the severe class imbalance.
The confusion matrix showed that the model successfully identified the majority of fraudulent transactions while maintaining a very low number of false positives.

## Technologies Used
Python was used as the primary programming language along with several popular data science libraries, including Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost, Imbalanced-Learn, and Joblib.

## Key Learnings
Through this project, I gained practical experience in exploratory data analysis, handling imbalanced datasets, feature preprocessing, hyperparameter tuning, model evaluation, threshold optimization, and model persistence. The project also provided insight into the challenges associated with real-world fraud detection systems and the importance of selecting appropriate evaluation metrics for imbalanced classification problems.

## Future Improvements
Future enhancements may include comparing XGBoost with LightGBM and CatBoost, incorporating explainable AI techniques such as SHAP, deploying the model using Flask or FastAPI, and developing a real-time fraud monitoring dashboard.

## Conclusion
This project demonstrates a complete machine learning workflow for credit card fraud detection, from raw data exploration to model optimization and evaluation. By combining SMOTE, RandomizedSearchCV, and XGBoost, the final solution effectively addresses class imbalance and achieves strong fraud detection performance suitable for real-world financial applications. 

## Author 
Atharva Ambilwade

