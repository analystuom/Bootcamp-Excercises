import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

#Load Data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Transform Data in 1D
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.flatten()

x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.flatten()

#Normalize Data
x_train = x_train/256
x_test = x_test/256

#Develop Model
model = XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

