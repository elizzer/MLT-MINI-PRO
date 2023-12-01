import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import lightning as L
from torch.utils.data import TensorDataset,DataLoader

class LSTM(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.lstm=nn.LSTM(input_size=1, hidden_size=1)
    
    def forward(self, input):

        input_trans=input.view(len(input),1)

        lstm_out,temp=self.lstm(input_trans)
        
        pred=lstm_out[-1]

        return lstm_out
    

model=LSTM()
input=torch.Tensor([3,2,4,5,2],dtype=torch.long)
input = input.unsqueeze(1)  # Add batch dimension

print(model(input))