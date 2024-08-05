from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

filename = 'model/model_dl.pkl'
model = pickle.load(open(filename, 'rb'))

with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('model/label_tokenizer.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 178

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])


def predict():
    message = request.form['sentence']
    sequences = tokenizer.texts_to_sequences([message])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_label_index = np.argmax(prediction[0])
    
    predicted_label = label_tokenizer.index_word[predicted_label_index]
    
    # label_map = {v : k for k, v in label_tokenizer.word_index.items()}
    # emotion = label_map[predicted_label[0]]
    # print(label_map)

    # return jsonify({'emotion': predicted_label})
    return render_template('result.html', prediction = predicted_label)


if __name__ == '__main__':
    app.run(debug=True)