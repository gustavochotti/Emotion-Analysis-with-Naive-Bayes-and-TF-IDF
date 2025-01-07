# train_model.py
import logging
import numpy as np
import csv
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data with multiple emotion categories
def load_data(filename):
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            return [(row[0], row[1]) for row in reader]  # Label is an emotion string
    except FileNotFoundError:
        logging.error(f"File '{filename}' not found.")
        raise
    except ValueError as e:
        logging.error("Error processing data from the CSV file. Check the format.")
        raise

# Load the data
data = load_data("emotions.csv")
texts, labels = zip(*data)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Adjust class weights for imbalanced classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_dict = dict(zip(np.unique(labels), class_weights))

# Train the model
model = MultinomialNB()
logging.info("Training the model with cross-validation for multiple emotions...")
cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
logging.info(f"Cross-validation - Average accuracy: {np.mean(cross_val_scores):.2f}")
model.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred, zero_division=1)
logging.info(f"Test set accuracy: {accuracy}")
logging.info(f"Classification report:\n{classification}")

# Persist model and vectorizer
joblib.dump(model, "naive_bayes_emotions_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_emotions.pkl")
logging.info("Model and vectorizer saved successfully.")
