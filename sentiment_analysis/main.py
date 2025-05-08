import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, redirect, url_for

# Download necessary NLTK data (with error handling)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK download failed...")

# Let's assume we have a CSV file with review data or create sample data
def load_sample_data():
    """Create sample review data if no data is provided"""
    reviews = [
        "This product is amazing! I love it so much.",
        "Terrible experience. Will never buy again.",
        "Pretty good but could be better.",
        "Not worth the money. Disappointed.",
        "Absolutely fantastic product, exceeded my expectations!",
        "Average product, nothing special.",
        "The customer service was horrible.",
        "Great value for money, highly recommend!",
        "It broke after two days.",
        "Decent quality but overpriced.",
        "I can't believe how well this worked for me!",
        "Waste of time and money. Complete garbage.",
        "Good product overall with minor issues.",
        "The quality is inconsistent but when it's good, it's great.",
        "Would definitely purchase this product again.",
        "The worst customer service I've ever experienced.",
        "Very satisfied with my purchase, exactly as described.",
        "Mediocre at best, wouldn't recommend to friends.",
        "This exceeded all my expectations, truly outstanding!",
        "Doesn't work as advertised, very misleading."
    ]
    
    # Assign labels: 1 for positive, 0 for negative
    labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Create DataFrame
    df = pd.DataFrame({
        'review_text': reviews,
        'sentiment': labels
    })
    
    return df

# For real data, uncomment this
# def load_data(file_path):
#     """Load data from a CSV file"""
#     df = pd.read_csv(file_path)
#     return df

# Data preprocessing functions
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def preprocess_data(df, text_column):
    """Preprocess all reviews in the dataframe"""
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    return df

# Feature extraction
def vectorize_text_tfidf(train_data, test_data):
    """Vectorize text using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test, vectorizer

# Traditional ML models
def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a random forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes model"""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model and return metrics"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    for label, metrics in class_report.items():
        if label in ['0', '1']:
            print(f"Class {label}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred,
        'model_name': model_name
    }

# Feature importance analysis
def get_feature_importance(model, vectorizer, top_n=20):
    """Extract feature importance from a trained model"""
    if hasattr(model, 'coef_'):
        # For models like Logistic Regression
        importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        # For models like Random Forest
        importance = model.feature_importances_
    else:
        return None, None
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame for importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by absolute importance (to catch both positive and negative influences)
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
    
    # Get top positive and negative features
    positive_features = feature_importance[feature_importance['importance'] > 0].head(top_n)
    negative_features = feature_importance[feature_importance['importance'] < 0].head(top_n)
    
    return positive_features, negative_features

# Additional text analysis functions
def extract_ngrams(text_series, n=2, top_n=10):
    """Extract most common n-grams from text"""
    from nltk import ngrams
    
    # Combine all texts
    all_text = ' '.join(text_series)
    tokens = word_tokenize(all_text.lower())
    
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Count frequencies
    ngram_freq = pd.Series(n_grams).value_counts().head(top_n)
    
    # Format for readability
    readable_ngrams = [' '.join(gram) for gram in ngram_freq.index]
    
    return pd.Series(ngram_freq.values, index=readable_ngrams)

# Function to create matplotlib figures and convert to base64 for embedding in HTML
def create_figure_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Create figures for the dashboard
def create_sentiment_distribution_chart(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts = df['sentiment'].value_counts()
    labels = ['Negative', 'Positive']
    sizes = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]
    colors = ['red', 'green']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Sentiment Distribution')
    
    return create_figure_to_base64(fig)

def create_review_length_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Positive reviews
    positive_lengths = df[df['sentiment'] == 1]['review_text'].str.len()
    ax.hist(positive_lengths, alpha=0.5, label='Positive', color='green', bins=10)
    
    # Negative reviews
    negative_lengths = df[df['sentiment'] == 0]['review_text'].str.len()
    ax.hist(negative_lengths, alpha=0.5, label='Negative', color='red', bins=10)
    
    ax.set_xlabel('Review Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Review Length Distribution by Sentiment')
    ax.legend()
    
    return create_figure_to_base64(fig)

def create_confusion_matrix_chart(confusion_matrix_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='viridis', ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    return create_figure_to_base64(fig)

def create_feature_importance_charts(positive_features, negative_features):
    # Create two charts: one for positive features, one for negative features
    # Positive features
    if positive_features is not None:
        pos_fig, pos_ax = plt.subplots(figsize=(10, 8))
        pos_features = positive_features.head(15).sort_values('importance')
        pos_ax.barh(pos_features['feature'], pos_features['importance'], color='green')
        pos_ax.set_xlabel('Importance')
        pos_ax.set_title('Top Positive Sentiment Words')
        positive_img = create_figure_to_base64(pos_fig)
    else:
        positive_img = None
    
    # Negative features
    if negative_features is not None:
        neg_fig, neg_ax = plt.subplots(figsize=(10, 8))
        neg_features = negative_features.head(15).sort_values('importance')
        neg_ax.barh(neg_features['feature'], neg_features['importance'], color='red')
        neg_ax.set_xlabel('Importance')
        neg_ax.set_title('Top Negative Sentiment Words')
        negative_img = create_figure_to_base64(neg_fig)
    else:
        negative_img = None
    
    return positive_img, negative_img

def create_word_frequency_chart(text_series, title):
    # Generate word frequencies
    all_words = ' '.join(text_series).split()
    word_freq = pd.Series(all_words).value_counts().head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    word_freq.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Word')
    ax.set_title(title)
    
    return create_figure_to_base64(fig)

# Predict sentiment for new text
def predict_sentiment(text, model, vectorizer):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # Predict
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]  # Probability of positive class
    
    return {
        'raw_text': text,
        'preprocessed_text': preprocessed_text,
        'prediction': 'Positive' if prediction == 1 else 'Negative',
        'probability': probability
    }

# Main execution function to prepare data and models
def prepare_analysis():
    print("Starting sentiment analysis pipeline...")
    
    # Load data
    df = load_sample_data()
    print(f"Loaded {len(df)} review samples")
    
    # Preprocess data
    print("Preprocessing text data...")
    df = preprocess_data(df, 'review_text')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")
    
    # Vectorize text using TF-IDF
    print("Vectorizing text data...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text_tfidf(X_train, X_test)
    
    # Train models
    print("Training multiple models...")
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train_tfidf, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest")
    
    # 3. Naive Bayes
    print("Training Naive Bayes...")
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    nb_metrics = evaluate_model(nb_model, X_test_tfidf, y_test, "Naive Bayes")
    
    # Determine best model
    models = [
        (lr_model, lr_metrics),
        (rf_model, rf_metrics),
        (nb_model, nb_metrics)
    ]
    
    best_model, best_metrics = max(models, key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_metrics['model_name']} with accuracy {best_metrics['accuracy']:.4f}")
    
    # Extract feature importance for the best model
    pos_features, neg_features = get_feature_importance(best_model, tfidf_vectorizer)
    
    # Additional text analysis
    print("\nGenerating additional text insights...")
    
    # Extract common bi-grams
    positive_bigrams = extract_ngrams(df[df['sentiment'] == 1]['cleaned_text'])
    negative_bigrams = extract_ngrams(df[df['sentiment'] == 0]['cleaned_text'])
    
    print("\nTop positive bi-grams:")
    print(positive_bigrams)
    
    print("\nTop negative bi-grams:")
    print(negative_bigrams)
    
    return {
        'df': df,
        'best_model': best_model,
        'best_metrics': best_metrics,
        'tfidf_vectorizer': tfidf_vectorizer,
        'positive_features': pos_features,
        'negative_features': neg_features,
        'all_models': {
            'Logistic Regression': (lr_model, lr_metrics),
            'Random Forest': (rf_model, rf_metrics),
            'Naive Bayes': (nb_model, nb_metrics)
        }
    }

# Initialize Flask application
app = Flask(__name__)

# Global variables to store our analysis results
analysis_results = None

@app.route('/')
def index():
    global analysis_results
    
    # If analysis hasn't been run yet, run it
    if analysis_results is None:
        analysis_results = prepare_analysis()
    
    # Create charts
    df = analysis_results['df']
    best_metrics = analysis_results['best_metrics']
    pos_features = analysis_results['positive_features']
    neg_features = analysis_results['negative_features']
    
    # Generate base64 encoded charts
    sentiment_pie_chart = create_sentiment_distribution_chart(df)
    review_length_hist = create_review_length_histogram(df)
    confusion_matrix_chart = create_confusion_matrix_chart(best_metrics['confusion_matrix'])
    pos_features_chart, neg_features_chart = create_feature_importance_charts(pos_features, neg_features)
    
    # Get sample reviews (5 positive, 5 negative)
    positive_reviews = df[df['sentiment'] == 1].head(5).to_dict('records')
    negative_reviews = df[df['sentiment'] == 0].head(5).to_dict('records')
    
    return render_template('index.html', 
                          sentiment_pie_chart=sentiment_pie_chart,
                          review_length_hist=review_length_hist,
                          confusion_matrix_chart=confusion_matrix_chart,
                          pos_features_chart=pos_features_chart,
                          neg_features_chart=neg_features_chart,
                          positive_reviews=positive_reviews,
                          negative_reviews=negative_reviews,
                          best_model_name=best_metrics['model_name'],
                          accuracy=best_metrics['accuracy'],
                          pos_precision=best_metrics['classification_report']['1']['precision'],
                          pos_recall=best_metrics['classification_report']['1']['recall'],
                          pos_f1=best_metrics['classification_report']['1']['f1-score'],
                          neg_precision=best_metrics['classification_report']['0']['precision'],
                          neg_recall=best_metrics['classification_report']['0']['recall'],
                          neg_f1=best_metrics['classification_report']['0']['f1-score'])

@app.route('/explore', methods=['GET', 'POST'])
def explore_reviews():
    global analysis_results
    
    if analysis_results is None:
        analysis_results = prepare_analysis()
    
    df = analysis_results['df']
    
    # Default to showing all reviews
    filter_type = request.args.get('filter', 'all')
    
    if filter_type == 'positive':
        filtered_df = df[df['sentiment'] == 1]
        title = "Word Frequencies in Positive Reviews"
    elif filter_type == 'negative':
        filtered_df = df[df['sentiment'] == 0]
        title = "Word Frequencies in Negative Reviews"
    else:
        filtered_df = df
        title = "Word Frequencies in All Reviews"
    
    # Generate word frequency chart
    word_freq_chart = create_word_frequency_chart(filtered_df['cleaned_text'], title)
    
    # Get the reviews
    reviews = filtered_df.to_dict('records')
    
    return render_template('explore.html',
                          word_freq_chart=word_freq_chart,
                          reviews=reviews,
                          current_filter=filter_type)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global analysis_results
    
    if analysis_results is None:
        analysis_results = prepare_analysis()
    
    result = None
    
    if request.method == 'POST':
        review_text = request.form.get('review_text', '')
        if review_text:
            result = predict_sentiment(
                review_text, 
                analysis_results['best_model'], 
                analysis_results['tfidf_vectorizer']
            )
    
    return render_template('predict.html', result=result)

@app.route('/compare-models')
def compare_models():
    global analysis_results
    
    if analysis_results is None:
        analysis_results = prepare_analysis()
    
    all_models = analysis_results['all_models']
    model_names = list(all_models.keys())
    accuracies = [metrics['accuracy'] for _, metrics in all_models.values()]
    
    # Create bar chart for model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_names, accuracies, color=['blue', 'green', 'orange'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    model_comparison_chart = create_figure_to_base64(fig)
    
    # Collect metrics for all models
    models_data = []
    for name, (model, metrics) in all_models.items():
        models_data.append({
            'name': name,
            'accuracy': metrics['accuracy'],
            'pos_precision': metrics['classification_report']['1']['precision'],
            'pos_recall': metrics['classification_report']['1']['recall'],
            'pos_f1': metrics['classification_report']['1']['f1-score'],
            'neg_precision': metrics['classification_report']['0']['precision'],
            'neg_recall': metrics['classification_report']['0']['recall'],
            'neg_f1': metrics['classification_report']['0']['f1-score']
        })
    
    return render_template('compare_models.html',
                          model_comparison_chart=model_comparison_chart,
                          models_data=models_data)

if __name__ == "__main__":
    app.run(debug=True)