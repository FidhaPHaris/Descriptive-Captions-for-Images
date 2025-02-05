from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.applications import ResNet50
from keras.layers import Input, Dense, LSTM, TimeDistributed, Embedding, RepeatVector, Concatenate, Activation
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from gtts import gTTS
from googletrans import Translator

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Load ResNet50 model
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

# Load vocabulary
vocab = np.load('vocab.npy', allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}
embedding_size = 128
max_len = 40
vocab_size = len(vocab)

# Define image input
image_input = Input(shape=(2048,))
image_features = Dense(embedding_size, activation='relu')(image_input)
image_features = RepeatVector(max_len)(image_features)

# Define language input
language_input = Input(shape=(max_len,))
language_features = Embedding(input_dim=vocab_size, output_dim=embedding_size)(language_input)
language_features = LSTM(256, return_sequences=True)(language_features)
language_features = TimeDistributed(Dense(embedding_size))(language_features)

# Concatenate image and language features
concatenated = Concatenate(axis=-1)([image_features, language_features])

# LSTM layers
x = LSTM(128, return_sequences=True)(concatenated)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
output = Activation('softmax')(x)

# Define model
model = Model(inputs=[image_input, language_input], outputs=output)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Load trained model weights
model.load_weights('mine_model_weights.h5')

# Initialize translation client
translator = Translator()

# Function to translate caption
def translate_caption(caption, target_language):
    translated = translator.translate(caption, src='en', dest=target_language)
    return translated.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, vocab, inv_vocab, resnet

    # Receive uploaded image
    file = request.files['file1']
    file.save('static/file.jpg')

    # Process uploaded image
    img = cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))

    # Extract image features
    features = resnet.predict(img).reshape(1, 2048)

    # Generate caption
    text_in = ['startofseq']
    final = ''
    count = 0
    while count < 20:
        count += 1
        encoded = [vocab[i] for i in text_in]
        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)
        sampled_index = np.argmax(model.predict([features, padded]))
        sampled_word = inv_vocab[sampled_index]
        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        text_in.append(sampled_word)

    # Generate audio from the caption
    tts = gTTS(final, lang='en')  # 'final' contains the generated caption
    audio_file_path = 'static/audio.mp3'  # Path to save the audio file
    tts.save(audio_file_path)

    return render_template('predict.html', final=final)

@app.route('/translate', methods=['POST'])
def translate():
    language = request.json['language']
    caption = request.json['caption']

    # Translate the caption
    translated_caption = translate_caption(caption, language)

    return jsonify({'translated_caption': translated_caption})

if __name__ == "__main__":
    app.run(debug=True)
