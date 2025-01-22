from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()  # Enable eager execution explicitly
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from googletrans import Translator

app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Initialize Google Translator
translator = Translator()

# Read the dataset
df = pd.read_csv('./datasest/hate_speech.csv')

# Lowercase all words in the 'tweet' column
df['tweet'] = df['tweet'].str.lower()

# Remove punctuations from the 'tweet' column
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['tweet'] = df['tweet'].apply(remove_punctuations)

# Remove stopwords from the 'tweet' column
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

df['tweet'] = df['tweet'].apply(remove_stopwords)

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
X = token.texts_to_sequences(balanced_df['tweet'])
X_pad = pad_sequences(X, maxlen=max_len)

# One-hot encoding for target variable
encoder = OneHotEncoder()
Y = encoder.fit_transform(balanced_df[['class']]).toarray()

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X_pad, Y, test_size=0.2, random_state=22)

# Define CNN model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32)

# Save the trained model
model.save('hate_speech_detection_model.h5')

# Prediction function
def predict_hate_speech(text):
    # Translate text to English
    translated_text = translator.translate(text, dest='en').text
    translated_text = translated_text.lower()
    translated_text = remove_punctuations(translated_text)
    translated_text = remove_stopwords(translated_text)
    sequence = token.texts_to_sequences([translated_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return prediction

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_hate_speech(text)
        if np.argmax(prediction) == 1:
            result = "🫤 Hate And Offensive Speech"
        else:
            result = "😊 Ahha!! No Hate Detected"
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
