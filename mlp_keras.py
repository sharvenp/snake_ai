
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
    model.add(Dense(5, input_dim=9, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(4, activation='sigmoid'))
    model.summary()
    model.compile(loss=losses.mean_squared_error,
              optimizer=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

        
    # model.fit(x=x_train, y=y_train, batch_size=100, epochs=100, shuffle=True)

if __name__ == "__main__":
    main()