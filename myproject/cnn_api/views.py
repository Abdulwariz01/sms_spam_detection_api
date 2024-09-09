from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import tensorflow as tf
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the SMS spam detection API Home Page")

# Load the trained model
model = tf.keras.models.load_model('C:/Users/USER/Desktop/Pro_sms_spam/myproject/cnn_api/sms_spam_model.h5')

# Load the tokenizer
with open('C:/Users/USER/Desktop/Pro_sms_spam/myproject/cnn_api/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def predict_spam(text):
    processed_text = preprocess_text(text)
    
    # Tokenize and pad the text
    tokenized_text = tokenizer.texts_to_sequences([processed_text])
    max_len = 100  # Ensure this matches your training max_len
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=max_len)
    
    prediction = model.predict(padded_text)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "spam" if predicted_class == 1 else "ham"

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '')

            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            prediction = predict_spam(text)
            return JsonResponse({'prediction': prediction})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
