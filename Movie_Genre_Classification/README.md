# Movie Genre Classification

## Overview

This project involves implementing a movie genre classification model based on movie descriptions. The primary objective is to classify movies into their respective genres using Natural Language Processing (NLP) techniques and machine learning models.

## Objectives

- To preprocess and clean the movie description data.
- To convert text data into numerical data using TF-IDF vectorization.
- To train multiple machine learning models for genre classification.
- To evaluate the performance of these models and select the best one.

## Approach

1. **Data Import and Exploration**:
    - Imported the datasets containing movie titles, genres, and descriptions.
    - Explored the data to understand its structure and distribution.

2. **Data Preprocessing**:
    - Cleaned the text data by removing punctuations and stopwords.
    - Tokenized the text data and applied TF-IDF vectorization to convert text into numerical features.

3. **Model Training**:
    - Split the data into training and testing sets.
    - Trained multiple models including Logistic Regression, Support Vector Machine (SVM), Multinomial Naive Bayes, Random Forest, and a Deep Learning model.

4. **Model Evaluation**:
    - Evaluated the models using accuracy, precision, recall, and F1-score.
    - Selected the best-performing model based on these metrics.

## Challenges

Working with a large and complex dataset for NLP tasks presented several challenges:
- **Data Cleaning**: Handling various forms of text data and ensuring consistency.
- **Model Training**: Training models on large datasets required significant computational resources and time. Accuracy remained pretty low after long training times.
- **Model Evaluation**: Balancing between different evaluation metrics to select the best model.

## Results

The models were evaluated on the test data, and the following results were achieved:
- **Logistic Regression**: Achieved an accuracy of more than 60% on the test data (best).
- **Support Vector Machine**: Performance improved with data scaling.
- **Multinomial Naive Bayes**: Achieved an accuracy of approximately 51.87% on the test data.
- **Random Forest**: Achieved an accuracy of approximately 47.44% on the test data.
- **Deep Learning Model**: Achieved an accuracy of approximately 55.47% on the test data.

## Conclusion

Despite the challenges, the project successfully implemented and evaluated multiple models for movie genre classification. The Logistic Regression model performed the best among the tested models. Future work could involve further tuning of model parameters and exploring advanced NLP techniques to improve classification accuracy.

## Dependencies

- Python 3.11.5
- Pandas
- Matplotlib
- NLTK
- Scikit-learn
- Joblib
- TensorFlow

## Acknowledgements

- Special thanks to the contributors and the open-source community for providing the tools and libraries used in this project.
- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
