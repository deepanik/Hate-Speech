import os
import logging
import warnings

# Suppress TensorFlow warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['ABSL_LOGGING_LEVEL'] = 'ERROR'

# Suppress all warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit

# Suppress absl logging after TensorFlow import
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np
from flask_cors import CORS
import json
import time
from functools import wraps
import uuid
from datetime import datetime, timedelta
import os

app = Flask(__name__)
app.static_folder = 'static'
app.static_url_path = '/static'
CORS(app)

# Configure SocketIO with proper settings for deployment
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',  # Using eventlet for WebSocket support
    ping_timeout=30,
    ping_interval=15,
    transports=['websocket', 'polling'],
    logger=True,
    engineio_logger=True
)

# Load the model and tokenizer with warning suppression
import contextlib
import io

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with contextlib.redirect_stderr(io.StringIO()):
        model = load_model('model/hate_speech_model.h5', compile=False)

with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Compile the model to resolve TensorFlow warnings
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 200

# Initialize translator
translator = GoogleTranslator()

# Simple in-memory storage for API keys and usage
api_keys = {}
api_usage = {}

class APIKey:
    def __init__(self, key, tier='free'):
        self.key = key
        self.tier = tier
        self.created_at = datetime.now()
        
        # Set daily limits based on tier
        self.limits = {
            'free': 100,
            'basic': 1000,
            'premium': 10000,
            'enterprise': float('inf')
        }
        
        self.daily_limit = self.limits[tier]

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        
        if not api_key or not api_key.startswith('Bearer '):
            return jsonify({'error': 'No API key provided'}), 401
        
        key = api_key.split(' ')[1]
        if key not in api_keys:
            return jsonify({'error': 'Invalid API key'}), 401
            
        # Check usage limits
        today = datetime.now().date()
        if today not in api_usage[key]:
            api_usage[key][today] = 0
            
        if api_usage[key][today] >= api_keys[key].daily_limit:
            return jsonify({'error': 'Daily API limit exceeded'}), 429
            
        api_usage[key][today] += 1
        
        return f(*args, **kwargs)
    return decorated_function

def generate_api_key(tier='free'):
    key = str(uuid.uuid4())
    api_keys[key] = APIKey(key, tier)
    api_usage[key] = {}
    return key

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate_to_english(text, source_lang):
    if source_lang == 'en':
        return text
    try:
        translation = translator.translate(text, source=source_lang, target='en')
        return translation
    except:
        return text

def get_alternative_suggestions(text, toxic_words):
    suggestions = {}
    # Add your dictionary of alternative words here
    alternatives = {
        'hate': ['dislike', 'disagree with'],
        'stupid': ['uninformed', 'misguided'],
        'idiot': ['uninformed person', 'confused individual'],
        # Add more alternatives
    }
    
    for word in toxic_words:
        if word.lower() in alternatives:
            suggestions[word] = alternatives[word.lower()]
    return suggestions

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def prepare_sequence(text, tokenizer):
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to sequence
    sequences = tokenizer.texts_to_sequences([' '.join(tokens)])
    # Pad sequence
    padded_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_seq

def get_toxic_words(text, prediction_probs):
    if prediction_probs[0] < 0.5:  # If not toxic
        return []
    
    words = text.lower().split()
    toxic_words = []
    # Simple approach: words that are commonly associated with hate speech
    toxic_indicators = ['hate', 'stupid', 'idiot']  # Expand this list
    
    for word in words:
        if word in toxic_indicators:
            toxic_words.append(word)
    
    return toxic_words

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_docs():
    return render_template('api.html')

@app.route('/api/keys', methods=['POST'])
def create_api_key():
    data = request.json
    tier = data.get('tier', 'free')
    
    if tier not in ['free', 'basic', 'premium', 'enterprise']:
        return jsonify({'error': 'Invalid tier'}), 400
        
    key = generate_api_key(tier)
    return jsonify({
        'api_key': key,
        'tier': tier,
        'daily_limit': api_keys[key].daily_limit
    })

@app.route('/api/analyze', methods=['POST'])
@require_api_key
def analyze():
    try:
        data = request.json
        text = data['text']
        
        # Detect language
        source_lang = detect_language(text)
        
        # Translate if not in English
        english_text = translate_to_english(text, source_lang)
        
        # Preprocess the text
        processed_text = preprocess_text(english_text)
        
        # Prepare sequence
        sequence = prepare_sequence(processed_text, tokenizer)
        
        # Get prediction
        prediction = model.predict(sequence)
        
        # Get toxic words and suggestions
        toxic_words = get_toxic_words(english_text, prediction[0])
        suggestions = get_alternative_suggestions(english_text, toxic_words)
        
        # Prepare response
        response = {
            'original_text': text,
            'source_language': source_lang,
            'english_translation': english_text if source_lang != 'en' else None,
            'prediction': {
                'hate_speech': float(prediction[0][0]),
                'offensive_language': float(prediction[0][1]),
                'neither': float(prediction[0][2])
            },
            'toxic_words': toxic_words,
            'suggestions': suggestions
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('analyze_text')
def handle_realtime_analysis(data):
    try:
        text = data['text']
        if len(text.strip()) == 0:
            return
        
        # Detect language
        source_lang = detect_language(text)
        
        # Translate if not in English
        english_text = translate_to_english(text, source_lang)
        
        # Preprocess the text
        processed_text = preprocess_text(english_text)
        
        # Prepare sequence
        sequence = prepare_sequence(processed_text, tokenizer)
        
        # Get prediction
        prediction = model.predict(sequence)
        
        # Get toxic words and suggestions
        toxic_words = get_toxic_words(english_text, prediction[0])
        suggestions = get_alternative_suggestions(english_text, toxic_words)
        
        # Emit results back to client
        emit('analysis_results', {
            'original_text': text,
            'source_language': source_lang,
            'english_translation': english_text if source_lang != 'en' else None,
            'prediction': {
                'hate_speech': float(prediction[0][0]),
                'offensive_language': float(prediction[0][1]),
                'neither': float(prediction[0][2])
            },
            'toxic_words': toxic_words,
            'suggestions': suggestions
        })
        
    except Exception as e:
        emit('analysis_error', {'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Hate Speech Detection App on port {port}...")
    print(f"Access the application at: http://localhost:{port}")
    
    try:
        # Try SocketIO first for full WebSocket support
        print("Starting with SocketIO support...")
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"SocketIO failed: {e}")
        print("Falling back to regular Flask (limited real-time features)...")
        # Fallback to regular Flask
        app.run(host='0.0.0.0', port=port, debug=False)
