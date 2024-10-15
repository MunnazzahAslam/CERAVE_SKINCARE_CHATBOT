![](https://miro.medium.com/v2/resize:fit:1400/1*a6GbKZOzFyEpszG1JVnddw.png)

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

2\. Load and preprocess the data file
-------------------------------------

In this step, I loaded the dataset and parsed the JSON file, and then preprocessed the data using the process of tokenization and label encoder. Tokenization is the process of breaking a string of text into smaller pieces called tokens and label encoding is used for encoding categorical variables as numerical values.

3\. Build the model
-------------------

In this step, I defined and compiled a linear stack of layers for a deep-learning model using the Keras library. The first layer is an `Input` layer that specifies the shape of the input data that the model will receive. The next layer is an `Embedding` layer that converts each word in the input data into a fixed-length vector of a defined size. The model also includes a `Dropout` layer, which randomly sets input units to 0 with a rate of 0.3 to reduce overfitting. The model then has a `SimpleRNN` layer, which is a fully-connected RNN where the output is fed back to the input. This is followed by a `BatchNormalization` layer, which normalizes the activations of the previous layer. The model also has a `Dense` layer with a `relu` activation function and an `L2` regularizer, which applies penalties on the layer parameters to prevent overfitting. The final layer is a `Dense` layer with a `softmax` activation function, which normalizes the output of the network to a probability distribution over the predicted output classes. The model is then compiled by specifying the loss function as `sparse_categorical_crossentropy` and the optimizer as `adam`, and specifying the metric to be tracked as `accuracy`.

4\. Testing the model
---------------------

Next, I set up two callbacks for training a deep learning model using the Keras library: an `EarlyStopping` callback and a `TensorBoard` callback. The `EarlyStopping` callback is used to stop the training when there is no improvement in the loss for a specified number of consecutive epochs. In this case, the callback is set to stop the training when there is no improvement in the loss for 400 epochs. The `TensorBoard` callback is used to log the training progress for visualization in TensorBoard. The model is then trained for 2000 epochs with a batch size of 64, and the two callbacks are passed to the `fit` method as arguments. The `fit` method trains the model on the specified features and labels and returns a history object containing the training loss and metrics for each epoch.

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

6\. Create the user interface
-----------------------------

Finally, for the UI I used the Gradio library for a better user experience and have also used custom CSS to further enhance the UI which can be found in the code repo.

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
https://youtu.be/dFPCwRgH35c?si=Ayxgkjyp1gXicwQ5
