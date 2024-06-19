from flask import Flask,render_template,request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from gtts import gTTS



from keras.applications import ResNet50

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

print("="*50)
print("resnet loaded")

vocab = np.load('vocab.npy' , allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}

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

# Load weights
model.load_weights('mine_model_weights.h5')

print("="*50)
print("model loaded")

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, vocab, inv_vocab,resnet
    file = request.files['file1']

    file.save('static/file.jpg')

    img=cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))
    img = np.reshape(img, (1,224,224,3))

    features = resnet.predict(img).reshape(1,2048)

    text_in = ['startofseq']
    final = ''

    print('='*50)
    print("Getting Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([features, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)




    # Generate audio from the caption
    tts = gTTS(final, lang='en')  # 'final' contains the generated caption
    audio_file_path = 'static/audio.mp3'  # Path to save the audio file
    tts.save(audio_file_path)





    return render_template('predict.html',final=final)

if __name__ == "__main__":
    app.run(debug=True)

