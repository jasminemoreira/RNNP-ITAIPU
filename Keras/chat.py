# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:44:32 2021

@author: Jasmine Moreira
"""

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

qdf = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP\Keras\ChatBot\questions.csv',sep=";")
adf = pd.read_csv (r'C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP\Keras\ChatBot\answers.csv',sep=";")

max_len = 15
max_words = 1000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(qdf.question)
sequences = tokenizer.texts_to_sequences(qdf.question)

x_train = pad_sequences(sequences, maxlen=max_len)
y_train = to_categorical(qdf.answer_id)

model = Sequential()
model.add(Embedding(1000, 8, input_length= max_len))
model.add(Flatten())
model.add(Dense(adf.answer_id.max()+1, activation = 'softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=1000, batch_size=22, validation_split=0.2)
 
prediction = model(x_train)
print(prediction)
np.argmax(prediction, axis=1) 

while(True):
    sentence = input("Você: ")
    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=max_len)
    prediction = model(sentence)
    category = np.argmax(prediction, axis=1)[0]
    answer = adf.query('answer_id=='+str(category)).to_numpy()
    print("Janaína:"+answer[0][1])

