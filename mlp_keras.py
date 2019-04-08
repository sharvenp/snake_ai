
"""
Snake Model:
9 input:
    - 4 adjacent walls
    - 4 adjacent body parts
    - 1 normalized angle to food

4 Output:
    - Direction
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses
from keras import optimizers
from keras.datasets import mnist

import numpy as np
import json

def main():    
    model = Sequential()
    model.add(Dense(5, input_shape=(9,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    model.summary()
    model.compile(loss=losses.mean_squared_error,
              optimizer=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

    training_data = None
    with open('training_data.json') as json_file:  
        training_data = json.load(json_file)
    
    x_train = np.asarray(training_data[0])
    y_train = np.asarray(training_data[1])
    # print(x_train.shape, y_train.shape)

    model.fit(x=x_train, y=y_train, batch_size=100, epochs=500, verbose=2, shuffle=True)

    model.save('trained models/test.h5')

if __name__ == "__main__":
    main()