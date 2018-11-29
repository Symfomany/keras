from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


## 1. Load Data

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


## 2. Define Model


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 3. Fit the Model

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)


# 4. Evaluate Model: Accurency
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# 5. Make Predictions

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

"""

...
    Epoch 145/150
    768/768 [==============================] - 0s - loss: 0.5105 - acc: 0.7396
    Epoch 146/150
    768/768 [==============================] - 0s - loss: 0.4900 - acc: 0.7591
    Epoch 147/150
    768/768 [==============================] - 0s - loss: 0.4939 - acc: 0.7565
    Epoch 148/150
    768/768 [==============================] - 0s - loss: 0.4766 - acc: 0.7773
    Epoch 149/150
    768/768 [==============================] - 0s - loss: 0.4883 - acc: 0.7591
    Epoch 150/150
    768/768 [==============================] - 0s - loss: 0.4827 - acc: 0.7656
    32/768 [>.............................] - ETA: 0s
    acc: 78.26%

"""