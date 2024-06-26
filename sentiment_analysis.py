# Import Librairie
import pandas as pd
import spacy
import numpy as np
from spacy.lang.en import English
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the CSV file with low_memory=False to avoid DtypeWarning
df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop NaN values specifically from 'reviews.text' and 'reviews.rating'
df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)

# Check data shape after dropping specific NaNs
print("Data shape after dropping NaNs from 'reviews.text' and 'reviews.rating':", df.shape)

# Initialize NLTK Porter Stemmer
stemmer = PorterStemmer()

# Custom tokenization function to handle punctuation and lemmatization
def custom_tokenizer(text):
    tokens = nlp(text)
    tokens = [token for token in tokens if token.text not in string.punctuation]  # Remove punctuation
    tokens = [token.lemma_ for token in tokens]  # Lemmatization
    return tokens

# Preprocess of all reviews including punctuation removal, lemmatization, and stemming
def preprocess_reviews(reviews):
    preprocessed_reviews = []
    for review in reviews:
        tokens = custom_tokenizer(review)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        cleaned_review = ' '.join(stemmed_tokens)
        preprocessed_reviews.append(cleaned_review)
    return preprocessed_reviews

preprocessed_reviews = preprocess_reviews(df['reviews.text'])

# Print first 5 preprocessed reviews
print(preprocessed_reviews[:5])  

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, df['reviews.rating'], test_size=0.2, random_state=42)

# Convert text data into numerical features that can be used by machine learning models
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the model
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
predictions = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions))