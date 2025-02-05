# Descriptive-Captions-for-Images

This project implements an  image captioning system that automatically generates descriptive captions for images, translates them into multiple languages, and provides audio output. By leveraging deep learning and speech synthesis, this system enhances accessibility for visually impaired users and promotes inclusivity in visual content interpretation.

## Key Features

Automated Caption Generation – Generate captions based on image content.

Multilingual Support – Captions can be translated into multiple languages.

Text-to-Speech (TTS) Integration – Converts captions into audio output for accessibility.

CNN-RNN Hybrid Model – Uses CNNs (ResNet) for image feature extraction and LSTMs for language modeling.

Flask API for Deployment – Provides an API endpoint for uploading images and receiving translated captions with audio output.


## How It Works

Image Processing: A CNN extracts visual features from the uploaded image.

Feature Encoding: The CNN output is passed to an LSTM model to generate captions.

Multilingual Translation: The caption is translated into different languages based on user preference.

Text-to-Speech Conversion: The final caption is converted into audio output.

Response: The system returns text-based captions and an audio file for playback.
