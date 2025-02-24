# Customer Churn Prediction

## Objective
The objective of this project is to predict customer churn using various machine learning models. Customer churn refers to the loss of clients or customers. It is a critical metric for businesses to understand and manage, as retaining existing customers is often more cost-effective than acquiring new ones.

## Approach
The approach taken in this project involves the following steps:

1. **Data Import and Exploration**:
    - Importing necessary libraries and loading the dataset.
    - Exploring the dataset to understand its structure and checking for missing values.

2. **Data Preprocessing**:
    - One-hot encoding categorical variables like 'Geography'.
    - Label encoding the 'Gender' column.
    - Dropping irrelevant columns such as 'RowNumber', 'CustomerId', and 'Surname'.

3. **Feature Engineering**:
    - Checking the correlation between features and the target variable 'Exited'.
    - Splitting the dataset into training and testing sets.

4. **Model Training and Evaluation**:
    - Training multiple machine learning models including Logistic Regression, Naive Bayes, Random Forest, XGBoost, Gradient Boosting, and CatBoost.
    - Evaluating each model using classification reports, confusion matrices, and accuracy scores.
    - Implementing ensemble methods like Stacking Classifier and Bagging Classifier to improve performance.

5. **Hyperparameter Tuning**:
    - Using GridSearchCV to find the best hyperparameters for the top-performing models.
    - Comparing the performance of models with tuned hyperparameters.

## Challenges Faced
- Handling categorical variables and ensuring proper encoding.
- Selecting the best model and tuning its hyperparameters for optimal performance and attaining an accuracy of over 86%.

## Results Achieved
- The Stacking Classifiers and Random Forest Classifier showed the best performance in terms of accuracy.
- Hyperparameter tuning further improved the model performance, with the best model achieving an accuracy of ~ 87%.

## Conclusion
This project demonstrates the process of building and evaluating machine learning models for customer churn prediction. By using various models and ensemble techniques, we were able to achieve high accuracy in predicting customer churn. The insights gained from this project can help businesses in making data-driven decisions to retain customers and reduce churn.


## Dependencies
- Python 3.11.5
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- catboost


## Acknowledgements
- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).

