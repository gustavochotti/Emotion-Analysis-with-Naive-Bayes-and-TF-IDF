
# Emotion Analysis with Naive Bayes and TF-IDF

This project provides a machine learning solution for classifying textual data into emotional categories. It uses a **Naive Bayes Classifier** along with **TF-IDF Vectorization** for feature extraction and model training. Pre-trained models are included for quick deployment and testing.

---

## Features

- **Model Training**: Train a Naive Bayes classifier on labeled text data.
- **Emotion Prediction**: Predict the emotional category of a given text.
- **Pre-Trained Models**: Use pre-trained models for immediate analysis without retraining.

---

## Project Structure

```
├── pre-trained_models/
│   ├── naive_bayes_emotions_model.pkl       # Pre-trained Naive Bayes model
│   ├── tfidf_vectorizer_emotions.pkl        # Pre-trained TF-IDF Vectorizer
├── train_model.py                           # Script for training the model
├── analyze_emotions.py                      # Script for analyzing emotions in text
├── emotions.csv                             # Example dataset (replace with your own)
├── README.md                                # Project documentation
```

---

## How the Code Works

1. **Training the Model**:
   - The `train_model.py` script:
     - Reads labeled text data from a CSV file (e.g., `emotions.csv`).
     - Splits the data into training and testing sets.
     - Transforms the text data into numerical vectors using the **TF-IDF Vectorizer**.
     - Trains a **Multinomial Naive Bayes Classifier** using the processed data.
     - Saves the trained model (`naive_bayes_emotions_model.pkl`) and the TF-IDF vectorizer (`tfidf_vectorizer_emotions.pkl`) into the `pre-trained_models/` directory.

2. **Analyzing Emotions**:
   - The `analyze_emotions.py` script:
     - Loads the pre-trained Naive Bayes model and TF-IDF vectorizer from the `pre-trained_models/` directory.
     - Transforms the input text using the TF-IDF vectorizer.
     - Predicts the emotional category of the input text using the Naive Bayes model.

3. **Pre-Trained Models**:
   - **`naive_bayes_emotions_model.pkl`**:
     - A serialized Naive Bayes classifier trained on labeled emotion data.
     - Used for classifying text into emotional categories without needing to retrain.

   - **`tfidf_vectorizer_emotions.pkl`**:
     - A pre-trained TF-IDF vectorizer that converts raw text into numerical feature vectors.
     - Ensures consistency between training and prediction processes.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/gustavochotti/emotion-analysis.git
   cd emotion-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you want to train a new model, make sure you have a properly formatted dataset (CSV file).

---

## Usage

### 1. Training the Model
To train the model with your own dataset:
```bash
python train_model.py
```
- Replace `emotions.csv` with your own dataset file.
- The trained model and vectorizer will be saved in the `pre-trained_models/` directory.

### 2. Analyzing Emotions
To predict emotions for new text using the pre-trained models:
```bash
python analyze_emotions.py
```
- Edit `analyze_emotions.py` to replace the input text (`new_message`) with your own input.

---

## Example Dataset Format
Your dataset (CSV file) should be formatted as follows:
```
text,label
"I am so happy today!","happy"
"I feel sad and lonely.","sad"
"This is absolutely amazing!","excited"
```

---

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `scikit-learn`
- `joblib`
- `matplotlib` (optional, for visualization)
- `pandas` (optional, for working with CSV files)

Install all dependencies using the provided `requirements.txt`.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments
- Thanks to the contributors and the open-source community for tools like `scikit-learn` and `joblib` that made this project possible.
