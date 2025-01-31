from tqdm import tqdm
from model import Classifier
from model import Regression
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

def model_train(model:Classifier,X :torch.Tensor,y :torch.Tensor,BATCH_SIZE :int,EPOCHS :int,device):
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(EPOCHS):
        for i in tqdm(range(0,len(X),BATCH_SIZE)):
            batch_x = X[i:i+BATCH_SIZE].to(device)
            batch_y = y[i:i+BATCH_SIZE].to(device)

            output = model(batch_x)

            model.zero_grad()
            loss = loss_func(output,batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch} loss:{loss}')

def train(model:Regression,X:torch.Tensor,y:torch.Tensor,BATCH_SIZE,EPOCHS,device):
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(EPOCHS):
        for i in tqdm(range(0,len(X),BATCH_SIZE)):
            batch_x = X[i:i+BATCH_SIZE].to(device)
            batch_y = y[i:i+BATCH_SIZE].to(device)

            output = model(batch_x)

            model.zero_grad()
            loss = loss_func(output,y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch:{epoch} loss{loss}")