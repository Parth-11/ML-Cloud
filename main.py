import data_processing as data
import pandas as pd

training_data = data.get_traning_data()
validation_data = data.get_validation_data()

X_train,y_train = training_data
X_test, y_test = validation_data

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
