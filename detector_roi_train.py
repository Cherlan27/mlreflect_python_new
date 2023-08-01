# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:52:50 2023

@author: petersdorf
"""

from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
layers = keras.layers

def gaussian2d(x_pix, y_pix, amp, x_pos, y_pos, x_sigma, y_sigma):
    return amp * np.exp(-((x_pix-x_pos)**2/(2*x_sigma**2)+(y_pix-y_pos)**2/(2*y_sigma**2)))

def gaussian_creater(amp, x_pos, y_pos, x_sigma, y_sigma):
    detector_img_signal = np.zeros((516,1554))
    for x_pix in range(0,516):
        for y_pix in range(0,1554):
            detector_img_signal[x_pix,y_pix] = gaussian2d(y_pix, x_pix, amp, x_pos, y_pos, x_sigma, y_sigma)
    return detector_img_signal + np.random.rand(516,1554)*5


def generation_data(size):
    train = []
    truth =  []
    for i in range(size):
        y_pos = np.random.uniform(516)
        x_pos = np.random.uniform(1554)
        y_sigma = np.random.uniform(3)
        x_sigma = np.random.uniform(50)
        amp = np.random.uniform(15000)
        truth_result = truth_data(size, y_pos, x_pos, y_sigma, x_sigma, amp)
        train_result = training_data(size, y_pos, x_pos, y_sigma, x_sigma, amp)
        train.append(train_result)
        truth.append(truth_result)
        
    return np.array(train), np.array(truth)

# Trainingsdata
def training_data(size, y_pos, x_pos, y_sigma, x_sigma, amp):
    train = gaussian_creater(amp, x_pos, y_pos, x_sigma, y_sigma)
    return train

def truth_data(size, y_pos, x_pos, y_sigma, x_sigma, amp):
    x_value1 = round(x_pos - 1/2*x_sigma)
    x_value2 = x_sigma
    y_value1 = round(y_pos - 1/2*y_sigma)
    y_value2 = y_sigma
    print("Truth-Roi: " + str(np.array([x_value1, y_value1, x_value2, y_value2],)))
    return np.array([x_value1, y_value1, x_value2, y_value2],)

train, truth = generation_data(100)

# xdata, _ = np.meshgrid(np.arange(detector_img_signal.shape[1]),
#                           np.arange(detector_img_signal.shape[0]))
# _, ydata = np.meshgrid(np.arange(detector_img_signal.shape[1]),
#                           np.arange(detector_img_signal.shape[0]))

# fig = plt.figure(dpi = 600)
# fig.patch.set_color("white")
# ax = fig.gca()
# ax.pcolormesh(xdata, ydata, detector_img_signal)

model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(516, 1554)))
model.add(layers.Reshape((516, 1554 ,1)))
model.add(layers.Convolution2D(16, (3, 3), padding='same', activation='relu'))
model.add(layers.Convolution2D(16, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4))

model.summary()

model.compile(
    loss='MeanSquaredError',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy'])

results = model.fit(train, truth,
                    batch_size=8,
                    epochs=40,
                    verbose=2,
                    #validation_split=0.1,
                    )

model.predict(np.array([gaussian_creater(1000, 40, 100, 30, 2)]))