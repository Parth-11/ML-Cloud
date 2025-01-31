import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(15,64)
        self.fc2 = nn.Linear(64,100)
        self.fc3 = nn.Linear(100,150)
        self.output = nn.Linear(150,2)
    
    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)

        return F.log_softmax(x,dim=1)
    
class Regression(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(21,100)
        self.fc2 = nn.Linear(100,220)
        self.fc3 = nn.Linear(220,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x,dim=1)