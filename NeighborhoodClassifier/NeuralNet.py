import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
train = pd.read_csv('NeighborhoodTraining.csv')
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
train_X = train_df.drop(columns=['In Need of Help'])
train_Y = to_categorical(train_df['In Need of Help'])
test = pd.read_csv('NeighborhoodTesting.csv')
test_df = pd.DataFrame(scaler.fit_transform(test ), columns=test.columns, index=test.index)
test_X = test_df.drop(columns=['In Need of Help'])
test_Y = to_categorical(test_df['In Need of Help'])
# Creates my model
model = Sequential()
number_columns = train_X.shape[1]
# Play around with the number of nodes. Modulating that changes the model's computational capabilities
# Dense is my layer type
model.add(Dense(200, activation='relu', input_shape=(number_columns,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))
# alpha, learning rate
# for now, 0.1 seems to be the optimal alpha
opt = SGD(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Once we have gone through 500 epochs in a row where the classifier has stopped improving, stop training
early_stopping_monitor = EarlyStopping(patience=500)
model.fit(train_X, train_Y, validation_split=0.2, epochs=5000, callbacks=[early_stopping_monitor])
test_labels = np.argmax(test_Y, axis=1)
test_pred = model.predict_classes(test_X)
print(test_pred)
print(test_labels)
print(accuracy_score(test_pred, test_labels))