import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense

np.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=150,batch_size=10)
scores = model.evaluate(x,y)
print("\n%s: %.2f%% is the result"%(model.metrics_names[1],scores[1]*100))

model_json=model.to_json()
with open("model_try.json","w") as json_file:
	json_file.write(model_json)
model.save_weights("model_try.h5")
print("Saved model to disk")



