# import data_processing as data
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from model import Classifier
import torch
# import train
import exec_time

TRAIN_MODEL = True
BATCH_SIZE = 100
EPOCHS = 8

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

training_data = exec_time.train_data()
validation_data = exec_time.test_data()

X_train,y_train = training_data
X_test,y_test = validation_data

print(X_train.shape)

# training_data = data.get_traning_data()
# validation_data = data.get_validation_data()

# X_train,y_train = training_data
# X_test, y_test = validation_data

# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# classifier = Classifier().to(device)

# X_train = torch.tensor(X_train).float()
# y_train = torch.tensor(y_train).long()
# X_test = torch.tensor(X_test)
# y_test = torch.tensor(y_test)

# if TRAIN_MODEL:
#     train.model_train(classifier,X_train,y_train,BATCH_SIZE,EPOCHS,device)
#     with open('classifier.pt','wb') as f:
#         torch.save(classifier.state_dict(),f)
# else:
#     with open('classifier.pt','rb') as f:
#         classifier.load_state_dict(torch.load(f))
