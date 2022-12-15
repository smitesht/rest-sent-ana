import pickle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json

def get_review(review, tokenizer):
    sentence_list = []
    sentence_list.append(review)
    sequences = tokenizer.texts_to_sequences(sentence_list)
    input = pad_sequences(sequences, maxlen=200, padding='pre')
    #print(input)
    model = load_model('./model/restaurant.h5')
    output = model.predict(input)
    #print(output[0])
    
    if output[0] >= 0.5:    
        #print("The feedback from user is Positive")    
        return 1
    else:
        #print("The feedback from user is Negative")
        return 0 
    

with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


app = Flask(__name__)
CORS(app)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/review', methods = ['POST'])
def review():
    if request.method == 'POST':
        record = request.get_json()    
        user_feedback = record['review']
        prediction_val = get_review(user_feedback, tokenizer)    
        prediction = {}
        prediction['sentiment'] = prediction_val
        prediction['review'] = record['review']
        print(prediction)        
        return json.dumps(prediction)
    
if __name__ == "__main__":
    app.run(use_reloader=False)