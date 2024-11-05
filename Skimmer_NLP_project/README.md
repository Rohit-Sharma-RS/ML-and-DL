# Skimmer NLP: Making Reading Medical Abstracts Easier

This notebook explores the use of Natural Language Processing (NLP) techniques to automatically classify sentences within medical research abstracts. The goal is to facilitate faster comprehension of key information by identifying the roles of different sentences (e.g., objective, methods, results, conclusions).

## Key Features:

* **Dataset:** Utilizes the PubMed 20k RCT dataset, where numbers have been replaced with "@" symbols.
* **Preprocessing:** Includes functions to extract relevant data from the dataset and structure it for modeling.
* **Baseline Model:** Implements a Naive Bayes classifier using TF-IDF vectorization as a baseline for performance comparison.
* **Deep Learning:** Explores the use of TensorFlow and Keras to build and train custom embedding layers and deep learning models for sentence classification.
* **Data Pipelines:** Demonstrates the creation of efficient TensorFlow datasets for training and evaluation.
* **Evaluation:** Provides functions to calculate accuracy, precision, recall, and F1-score to assess model performance.


## How to Use:

1. **Clone the Repository:** Run the notebook cells
2. **Use the functions skimmer() to get skimmable output as a dictionary** Store this in a variable say ```skimmed```
3. **Use the functions helper_skimmer_text(<skimmed>) to get the final ouput**
4. **Experiment:** Explore different model architectures and hyperparameters to potentially improve performance.


## Dependencies:

* Python 3
* TensorFlow
* Keras
* Scikit-learn
* Pandas
* Matplotlib


## Future Work:
#### This model is currently in development phase been trained on only 3 epochs and hence less accurate.
* Investigating more advanced deep learning models.
* Fine-tuning hyperparameters for optimal performance.
* Making model more robust using 200K data(more computationally expensive)
* Exploring alternative embedding methods.
* Developing a user interface for abstract summarization.