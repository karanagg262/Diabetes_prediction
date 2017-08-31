import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model_try.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_try.h5")
print("Loaded model from disk")

predictions = loaded_model.predict(x)
rounded = [round(x[0]) for x in predictions]
print(rounded)
np.savetxt(
	'pima_output.csv',
	rounded,
	fmt='%.9f',
	delimiter=",",
	newline='\n',
	header='probability')

