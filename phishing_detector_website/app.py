from flask import Flask, render_template, request
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load saved models, tokenizer, and TF-IDF vectorizer
model_dir = r"C:\Users\jagad\ML\NLP\phishing_detector\saved_models"
with open(os.path.join(model_dir, 'naive_bayes_model.pkl'), 'rb') as file:
    nb_classifier = pickle.load(file)
with open(os.path.join(model_dir, 'logistic_regression_model.pkl'), 'rb') as file:
    lr_classifier = pickle.load(file)
with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as file:
    tokenizer = pickle.load(file)
with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load LSTM model
lstm_model = load_model(os.path.join(model_dir, 'lstm_model.h5'))

# Constants for padding and tokenizing
max_len = 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form['email_text']
        model_choice = request.form['model']

        # Preprocess input text
        email_seq = tokenizer.texts_to_sequences([email_text])
        email_pad = pad_sequences(email_seq, maxlen=max_len)

        # Predict based on chosen model
        if model_choice == 'Naive Bayes':
            email_tfidf = tfidf_vectorizer.transform([email_text])
            prediction = nb_classifier.predict(email_tfidf)[0]
            print("Naive Bayes Raw Prediction:", prediction)

        elif model_choice == 'Logistic Regression':
            email_tfidf = tfidf_vectorizer.transform([email_text])
            prediction = lr_classifier.predict(email_tfidf)[0]
            print("Logistic Regression Raw Prediction:", prediction)

        elif model_choice == 'LSTM':
            prediction_prob = lstm_model.predict(email_pad)[0][0]
            print("LSTM Raw Prediction Probability:", prediction_prob)
            prediction = int(prediction_prob > 0.4)  # Adjust threshold as needed

        # Interpret prediction result
        result = "Phishing Email" if prediction == 1 else "Legitimate Email"
        return render_template('index.html', email_text=email_text, model_choice=model_choice, result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
