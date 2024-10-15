# CeraVe Skincare Chatbot

In recent years, chatbots have become increasingly popular for providing personalized recommendations in various fields, including skincare. This project is a **retrieval-based skincare chatbot** that guides users through their skincare routines based on their needs. It also offers information about **CeraVe products**, including new releases, best sellers, and ingredients. Plus, it provides skincare tips for achieving that coveted glass skin.

## Dataset

The dataset used is a JSON file from the CeraVe website, with 54 intents, each containing patterns and responses. Here’s an example format:

```json
{
  "tag": "skincare_tips",
  "patterns": [
    "share skincare tips", …
  ],
  "responses": [
    "Stay Hydrated 💆🏻 ", …
  ],
  "context": [
    ""
  ]
}

# Implementation

1\. Importing Libraries

bash

Copy code

pip install tensorflow sklearn pandas numpy matplotlib nltk gradio

python

Copy code

import tensorflow as tf

import gradio as gr

import json

import random

import numpy as np

from keras.models import Sequential

from keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.preprocessing import LabelEncoder

from keras_preprocessing.sequence import pad_sequences

2\. Data Preprocessing

Tokenization and Label Encoding were applied for transforming input text into sequences and categorical labels.

python

Copy code

# Load dataset

with open("skincare_intents.json") as file:

  dataset = json.load(file)

# Preprocess dataset

# Tokenization and Label Encoding...

3\. Model Building

A Sequential model was created using Keras with layers like SimpleRNN, Dropout, and Dense.

python

Copy code

model = Sequential()

model.add(Input(shape=(input_shape)))

model.add(Embedding(input_dim=vocabulary_size, output_dim=100))

model.add(Dropout(0.3))

model.add(SimpleRNN(16))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(Dense(output_length, activation="softmax"))

4\. Training

The model was trained with EarlyStopping and TensorBoard for 2000 epochs.

python

Copy code

earlyStopping = EarlyStopping(monitor='loss', patience=400)

tensorboard_callback = TensorBoard(log_dir="logs/fit/")

history = model.fit(features, labels, epochs=2000, batch_size=64, callbacks=[earlyStopping, tensorboard_callback])

5\. Response Generation

This function generates responses based on user input using the trained model.

python

Copy code

def generate_response(query):

    # Process input and predict response

    # Append to history and return response

    history.append((query, random.choice(responses[response_tag])))

    return history

6\. User Interface

The user interface was built using Gradio for easy interaction with the chatbot.

python

Copy code

with gr.Blocks(title= 'Cerave Skincare Chatbot') as demo:

    gr.Markdown("<h1>Cerave Skincare Chatbot</h1>")

    output = gr.outputs.Chatbot()

    input = gr.inputs.Textbox(placeholder="Ask me anything")

    ask_button = gr.Button("Ask")

    ask_button.click(fn=generate

You said:

i need in md format

ChatGPT said:

ChatGPT

Here's the README.md file for your project in Markdown format:

markdown

Copy code

# CeraVe Skincare Chatbot

In recent years, chatbots have become increasingly popular for providing personalized recommendations in various fields, including skincare. This project is a **retrieval-based skincare chatbot** that guides users through their skincare routines based on their needs. It also offers information about **CeraVe products**, including new releases, best sellers, and ingredients. Plus, it provides skincare tips for achieving that coveted glass skin.

## Dataset

The dataset used is a JSON file from the CeraVe website, with 54 intents, each containing patterns and responses. Here's an example format:

```json

{

  "tag": "skincare_tips",

  "patterns": [

    "share skincare tips", ...

  ],

  "responses": [

    "Stay Hydrated 💆🏻 ", ...

  ],

  "context": [

    ""

  ]

}

Implementation

1\. Importing Libraries

bash

Copy code

pip install tensorflow sklearn pandas numpy matplotlib nltk gradio

python

Copy code

import tensorflow as tf

import gradio as gr

import json

import random

import numpy as np

from keras.models import Sequential

from keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.preprocessing import LabelEncoder

from keras_preprocessing.sequence import pad_sequences

2\. Data Preprocessing

Tokenization and Label Encoding were applied for transforming input text into sequences and categorical labels.

python

Copy code

# Load dataset

with open("skincare_intents.json") as file:

  dataset = json.load(file)

# Preprocess dataset

# Tokenization and Label Encoding...

3\. Model Building

A Sequential model was created using Keras with layers like SimpleRNN, Dropout, and Dense.

python

Copy code

model = Sequential()

model.add(Input(shape=(input_shape)))

model.add(Embedding(input_dim=vocabulary_size, output_dim=100))

model.add(Dropout(0.3))

model.add(SimpleRNN(16))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(Dense(output_length, activation="softmax"))

4\. Training

The model was trained with EarlyStopping and TensorBoard for 2000 epochs.

python

Copy code

earlyStopping = EarlyStopping(monitor='loss', patience=400)

tensorboard_callback = TensorBoard(log_dir="logs/fit/")

history = model.fit(features, labels, epochs=2000, batch_size=64, callbacks=[earlyStopping, tensorboard_callback])

5\. Response Generation

This function generates responses based on user input using the trained model.

python

Copy code

def generate_response(query):

    # Process input and predict response

    # Append to history and return response

    history.append((query, random.choice(responses[response_tag])))

    return history

6\. User Interface

The user interface was built using Gradio for easy interaction with the chatbot.

python

Copy code

with gr.Blocks(title='Cerave Skincare Chatbot') as demo:

    gr.Markdown("<h1>Cerave Skincare Chatbot</h1>")

    output = gr.outputs.Chatbot()

    input = gr.inputs.Textbox(placeholder="Ask me anything")

    ask_button = gr.Button("Ask")

    ask_button.click(fn=generate_response, inputs=input, outputs=output)

demo.launch(share=True)

# Demo Link: https://youtu.be/dFPCwRgH35c?si=Ayxgkjyp1gXicwQ5
