#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2021

@author: jasmine

1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4)  Executar treinamento
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image

# Preparação dos dados

base_dir = r"C:\Users\jasmi\OneDrive\Área de Trabalho\RNNP\Keras"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir,"catdog_train"),
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
        )

validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir,"catdog_test"),
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
        )

# criar o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3),activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# treinar o modelo
history = model.fit_generator(
        train_generator, 
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 50
        )

# avaliar resultado do treinamento
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# salvar o modelo treinado
model.save(base_dir+'/cats_and_dogs_small_1.h5')


""" Testar o modelo já treinado  """
# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model(base_dir+'/cats_and_dogs_small_1.h5')
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# predicting images
img = image.load_img(base_dir+"/catdog_test/dogs/dog.1550.jpg", target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images)
print(classes)
