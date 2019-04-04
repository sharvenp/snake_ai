
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses
from keras import optimizers
from keras.datasets import mnist
import numpy as np

def main():
    model = Sequential()
    model.add(Dense(50, input_dim=784, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.summary()
    model.compile(loss=losses.mean_squared_error,
              optimizer=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 784) / 255
    x_test = x_test.reshape(x_test.shape[0], 784) / 255

    y_tr = np.zeros((60000, 10))
    y_te = np.zeros((10000, 10))
    for i in range(60000):
        y_tr[i][y_train[i]] = 1.0  

    for i in range(10000):
        y_te[i][y_test[i]] == 1.0

    y_train = y_tr
    y_test = y_te

    model.fit(x=x_train, y=y_train, batch_size=100, epochs=100, shuffle=True)
    print(model.evaluate(x=x_test, y=y_test, batch_size=100))

if __name__ == "__main__":
    main()