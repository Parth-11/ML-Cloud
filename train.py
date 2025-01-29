from tqdm import tqdm
from model import Classifier
import torch.optim as optim
import torch.nn as nn

def model_train(model:Classifier,X,y,BATCH_SIZE,EPOCHS):
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    
    for epoch in range(EPOCHS):
        for i in tqdm(range(0,len(X),BATCH_SIZE)):
            batch_x = X[i:i+BATCH_SIZE]
            batch_y = y[i:i+BATCH_SIZE]

            output = model(batch_x)
            
            model.zero_grad()
            loss = loss_func(output,batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch} loss:{loss}')

