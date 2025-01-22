from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googletrans import Translator

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')


def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    for word in str(text).split():
        if word not in stop_words:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize(word)
            imp_words.append(word)
    output = " ".join(imp_words)
    return output


# Load the dataset
df = pd.read_csv('./datasest/hate_speech.csv')

# Preprocess the dataset
df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: remove_punctuations(x))
df['tweet'] = df['tweet'].apply(lambda text: remove_stopwords(text))

# Balance the dataset
class_2 = df[df['class'] == 2]
class_1 = df[df['class'] == 1].sample(n=3500)
class_0 = df[df['class'] == 0]
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

# Tokenization and Padding
max_words = 5000
max_len = 100
token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(balanced_df['tweet'])

# Load the trained model
model = load_model('./model/hate_speech_detection_model.h5')

# Initialize Google Translator
translator = Translator()


def preprocess_input(text):
    text = text.lower()
    text = remove_punctuations(text)
    text = remove_stopwords(text)

    # Translate non-English text to English
    if not text.isascii():
        text = translate_text(text)

    return text


def translate_text(text, target_language='en'):
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text


def predict_hate_speech(text):
    text = preprocess_input(text)
    sequence = token.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return prediction


@app.route('/')
def home():
    return render_template('index.html', prediction_text="")


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if text.strip() == '':
            return render_template('index.html', prediction_text="Please enter a sentence.")

        prediction = predict_hate_speech(text)

        # Decide final prediction based on the model
        if np.argmax(prediction) == 1:
            result = "🫤 Hate And Offensive Speech"
        else:
            result = "😊 Ahha!! No Hate Detected"

        return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
