![](https://miro.medium.com/v2/resize:fit:1400/1*a6GbKZOzFyEpszG1JVnddw.png)

What is Chatbot?
================

A chatbot is a computer program designed to simulate conversation with human users, especially over the Internet. Chatbots are often used for customer service, marketing on social media platforms, and instant messaging. There are two main types of chatbot models: retrieval-based and generative-based. Retrieval-based models provide pre-programmed responses to user input, while generative-based models use artificial intelligence to generate responses based on the input received.

![](https://miro.medium.com/v2/resize:fit:1400/1*ig1H8g_92KccAg79B0y-zQ.png)

[source](https://scholarworks.sjsu.edu/etd_projects/630/)

About this project
==================

In recent years, chatbots have become increasingly popular for providing personalized recommendations and advice in various fields, including skincare. Skincare chatbots can help users identify their skin concerns, recommend products, and provide tips and tricks for maintaining healthy skin.

For this project, I've built a retrieval-based skincare chatbot that will guide users about their skincare routines based on their needs. It would also guide users about CeraVe products available, new products, best sellers, and ingredients contained in a product. You can also trust this chatbot to give you skincare tips for that glass skin.

![](https://miro.medium.com/v2/resize:fit:568/1*OtbnlCB2qXSkqgTlJcg9HQ.png)

[source](https://dzone.com/articles/significance-of-chatbot)

Dataset used
============

The dataset is a JSON file that includes data collected from the CeraVe website. The dataset consists of 54 intents, each with its own patterns and responses, and is structured in the following format:

{\
"tag": "skincare_tips",\
"patterns": [\
"share skincare tips", ...\
  ],\
"responses": [\
"Stay Hydrated 💆🏻 ", ...\
  ],\
"context": [\
""\
  ]\
}

Implementation
==============

1\. Import necessary packages
-----------------------------

First of all, I've installed and imported all the packages that will be required for the project. I've also loaded the TensorBoard extension for Jupyter notebooks to visualize the training progress of their TensorFlow models, compare the performance of different models, and analyze the distributions and histograms of model parameters.

pip install tensorflow sklearn pandas numpy matplotlib nltk gradio

import os\
import io\
import nltk\
import string\
import json\
import random\
import pandas as pd\
import numpy as np\
import gradio as gr\
import matplotlib as mpl\
import tensorflow as tf\
%load_ext tensorboard\
import tensorboard\
import matplotlib.pyplot as plt\
from tensorflow import keras\
from keras.models import Sequential\
from sklearn.preprocessing import LabelEncoder\
from keras.preprocessing.text import Tokenizer\
from keras.callbacks import TensorBoard, EarlyStopping\
from keras_preprocessing.sequence import pad_sequences\
from keras.layers import Input, Embedding, SimpleRNN, BatchNormalization, LSTM, Dense, GlobalAveragePooling1D, Flatten, Dropout, GRU, Conv1D, MaxPool1D

2\. Load and preprocess the data file
-------------------------------------

In this step, I loaded the dataset and parsed the JSON file, and then preprocessed the data using the process of tokenization and label encoder. Tokenization is the process of breaking a string of text into smaller pieces called tokens and label encoding is used for encoding categorical variables as numerical values.

#Loading the intent based data in our notebook\
with open("skincare_intents.json") as cerave_skincare_chatbot:\
  dataset = json.load(cerave_skincare_chatbot)

#Extracting tags (intent), patterns (input provided by user) and responses (output provided by the bot) from dataset\
def processing_json_dataset(dataset):\
  tags = []\
  patterns = []\
  responses = {}\
  for intent in dataset['intents']:\
    responses[intent['tag']]=intent['responses']\
    for lines in intent['patterns']:\
      patterns.append(lines)\
      tags.append(intent['tag'])\
  return [tags, patterns, responses]

#Storing our extracted data\
[tags, patterns, responses] = processing_json_dataset(dataset)

#For each letter in pattern converting it to lowercase if it's not a special character\
dataset['patterns'] = dataset['patterns'].apply(lambda sequence:\
                                            [letters.lower() for letters in sequence if letters not in string.punctuation])

#Joining the lowercase letters into words again\
dataset['patterns'] = dataset['patterns'].apply(lambda word: ''.join(word))

#The ids for most frequent 1000 words would be returned.\
tokenizer = Tokenizer(num_words=1000)

#An index based vocabulary would be created based on the word frequency.\
tokenizer.fit_on_texts(dataset['patterns'])

#It basically takes each word in the text and replaces it with its corresponding integer value from the vocabulary made in the previous step.\
train = tokenizer.texts_to_sequences(dataset['patterns'])

#It will pad with 0 the shorter sequences to make sure all sequences have same length\
features = pad_sequences(train)

#It takes a categorical column and converts/maps it to numerical values.\
labelEncoder = LabelEncoder()\
labels = labelEncoder.fit_transform(dataset['tags'])

3\. Build the model
-------------------

In this step, I defined and compiled a linear stack of layers for a deep-learning model using the Keras library. The first layer is an `Input` layer that specifies the shape of the input data that the model will receive. The next layer is an `Embedding` layer that converts each word in the input data into a fixed-length vector of a defined size. The model also includes a `Dropout` layer, which randomly sets input units to 0 with a rate of 0.3 to reduce overfitting. The model then has a `SimpleRNN` layer, which is a fully-connected RNN where the output is fed back to the input. This is followed by a `BatchNormalization` layer, which normalizes the activations of the previous layer. The model also has a `Dense` layer with a `relu` activation function and an `L2` regularizer, which applies penalties on the layer parameters to prevent overfitting. The final layer is a `Dense` layer with a `softmax` activation function, which normalizes the output of the network to a probability distribution over the predicted output classes. The model is then compiled by specifying the loss function as `sparse_categorical_crossentropy` and the optimizer as `adam`, and specifying the metric to be tracked as `accuracy`.

#Creating a linear stack of layers.\
model = Sequential()

#Letting the model know about the shape of input it will be receiving\
model.add(Input(shape=(input_shape)))

#This layer will enable us to convert each word into a fixed length vector of defined size in this case 100\
model.add(Embedding(vocabulary+1, 100))

#Adding dropout to randomly set inputs to 0 out of all scalars inputs with a rate of 0.3.\
model.add(Dropout(0.3))

#SimpleRNN layer which is a fully-connected RNN where the output is to be fed back to input\
model.add(SimpleRNN(16, return_sequences=False))

#Applying batch normalization to maintain the mean output close to 0 and the output standard deviation close to 1\
model.add(BatchNormalization())

#Fully-connected layer with relu activation and l2 regularizer to apply penalties on layer parameters\
model.add(Dense(128,activation="relu", activity_regularizer = tf.keras.regularizers.L2(0.0001)))

#Softmax output layer to normalize the output of a network to a probability distribution over predicted output classes\
model.add(Dense(output_length, activation="softmax", activity_regularizer = tf.keras.regularizers.L2(0.0001)))

#Compiling the model by specifying loss as sparse_categorical_crossentropy and optimizer as adam\
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

4\. Testing the model
---------------------

Next, I set up two callbacks for training a deep learning model using the Keras library: an `EarlyStopping` callback and a `TensorBoard` callback. The `EarlyStopping` callback is used to stop the training when there is no improvement in the loss for a specified number of consecutive epochs. In this case, the callback is set to stop the training when there is no improvement in the loss for 400 epochs. The `TensorBoard` callback is used to log the training progress for visualization in TensorBoard. The model is then trained for 2000 epochs with a batch size of 64, and the two callbacks are passed to the `fit` method as arguments. The `fit` method trains the model on the specified features and labels and returns a history object containing the training loss and metrics for each epoch.

#This callback will stop the training when there is no improvement in the loss for forty consecutive epochs.\
earlyStopping = EarlyStopping(monitor = 'loss', patience = 400, mode = 'min', restore_best_weights = True)

# Define the Keras TensorBoard callback.\
logdir="logs/fit/"\
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#Train for 2000 epochs with batch size of 64.\
history_training = model.fit(features, labels, epochs=2000, batch_size=64, callbacks=[earlyStopping, tensorboard_callback])

Results:
--------

![](https://miro.medium.com/v2/resize:fit:1400/1*UIJl4q0x3EqKYUsWWz4gbw.png)

Accuracy curve:
---------------

![](https://miro.medium.com/v2/resize:fit:1400/1*RaXBgGm6TbuVk-Nh2ScgCw.png)

Loss curve:
-----------

![](https://miro.medium.com/v2/resize:fit:1400/1*yLCj5Ah67tIEM6zJtOYAvw.png)

5\. Generating responses
------------------------

In this step, a function to generate responses is defined that takes a user input query as input and returns a list of tuples containing the user's query and the bot's response. The function first processes the user input by lowercasing the letters and removing punctuation. The processed input is then tokenized using the `tokenizer` object and transformed into numerical data using the `texts_to_sequences` method. The numerical data is then padded to a fixed length using the `pad_sequences` function. The padded numerical data is then used as input to the model, which predicts the output using the `predict` method. The output is transformed back into the original data from numerical values using the `inverse_transform` method of the `labelEncoder` object. The bot's response is chosen randomly from a list of possible responses stored in the `responses` dictionary. The user's query and the bot's response are then appended to the `history` list and the list is returned by the function.

#To store messages for session\
history = []\
def generate_response(query):\
  texts = []

  #Processing user input\
  user_input = query\
  user_input = [letters.lower() for letters in user_input if letters not in string.punctuation]\
  user_input = ''.join(user_input)\
  texts.append(user_input)\
  user_input = tokenizer.texts_to_sequences(texts)\
  user_input = np.array(user_input).reshape(-1)\
  user_input = pad_sequences([user_input],input_shape)

  #Predicting the results\
  output = model.predict(user_input)

  #Returning the indices of maximum value\
  output = output.argmax()

  #Transforming back to original data from numerical values\
  response_tag = labelEncoder.inverse_transform([output])[0]

  #Appending user query with bots reply to render in UI\
  history.append((query, random.choice(responses[response_tag])))\
  return history

6\. Create the user interface
-----------------------------

Finally, for the UI I used the Gradio library for a better user experience and have also used custom CSS to further enhance the UI which can be found in the code repo link given at the end.

#Creating a gradio block to render UI for chatbot\
with gr.Blocks(title= 'Cerave Skincare Chatbot', css='gradio.css') as demo:\
    gr.Markdown("<h1>Cerave Skincare Chatbot</h1>")\
    output = gr.outputs.Chatbot()

    #Gradio row for user input and ask button\
    with gr.Row():\
      input = gr.inputs.Textbox(placeholder="Ask me anything")\
      ask_button = gr.Button("Ask")

    #Calling generate response and clearing user input textbox\
    ask_button.click(fn=generate_response, inputs=input, outputs=output)\
    ask_button.click(lambda x: "", input, input)

demo.launch(share=True)

7\. Run the chatbot
-------------------

Now that we have implemented our chatbot its now time to interact with it and test its performance. I've attached a couple of screenshots of our chatbot interaction with real users' queries.

![](https://miro.medium.com/v2/resize:fit:1560/1*mB7m8W3qoYSiqF7c6t-GYA.png)

![](https://miro.medium.com/v2/resize:fit:1584/1*5vkyLMX3gddA-xWg_7hPbg.png)

![](https://miro.medium.com/v2/resize:fit:1592/1*nkovfdlYwUv4JDjNR9eo6g.png)

![](https://miro.medium.com/v2/resize:fit:1482/1*OFiVJg7GvsZwAFns5-FJPg.png)

![](https://miro.medium.com/v2/resize:fit:1500/1*qMk0lG6qGI1fCSaO1PyoJA.png)

![](https://miro.medium.com/v2/resize:fit:1512/1*p0rRFFsDJ0Ba8AeqFkQF0Q.png)

Demo Link:
----------
