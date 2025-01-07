# analyze_emotions.py
import logging
import joblib
from sklearn.exceptions import NotFittedError

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to predict emotion for a new message
def predict_new_message(message, model, vectorizer):
    try:
        transformed_message = vectorizer.transform([message])
        prediction = model.predict(transformed_message)
        return prediction[0]  # Return the predicted emotion as a string
    except NotFittedError:
        logging.error("Model or vectorizer not trained.")
        raise
    except ValueError as e:
        logging.error("Error transforming or predicting the message.")
        raise

# Load the model and vectorizer
model = joblib.load("naive_bayes_emotions_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer_emotions.pkl")
logging.info("Model and vectorizer loaded successfully.")

# Test with a new message
new_message = "I hated that movie"
predicted_emotion = predict_new_message(new_message, model, vectorizer)
logging.info(f"New message: {new_message}")
logging.info(f"Predicted emotion: {predicted_emotion}")

print(f"Message: {new_message}")
print(f"Predicted Emotion: {predicted_emotion}")
