from google.colab import drive

drive.mount('/content/drive')


import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer,AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

dataPath='/content/drive/MyDrive/MiniProject/singleton_removed.csv'
df=pd.read_csv(dataPath)
X_train=df["comment_text"][:500]
Y_train=df["label"][:500]
X_test = df["comment_text"].iloc[-100:].reset_index(drop=True)
Y_test = df["label"].iloc[-100:].reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN=512

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x,y ,tokenizer, max_len):
        # print(df.head())
        self.df = df
        self.tokenizer = tokenizer
        self.max_len= max_len
        self.title=x
        self.targets=y
        # print(self.title)
    def __len__(self):
        return len(self.title)

    def __getitem__(self,index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer(
            title,
            # add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        target=self.targets[index]
        return inputs,target
    
train_dataset=CustomDataset(X_train,Y_train,tokenizer,MAX_LEN)
val_dataset=CustomDataset(X_test,Y_test,tokenizer,MAX_LEN)

train_data_loader=torch.utils.data.DataLoader(
    train_dataset,
    # shuffle=True,
    batch_size=64,
    num_workers=0

)
val_data_loader=torch.utils.data.DataLoader(
    val_dataset,
    # shuffle=True,
    batch_size=64,
    num_workers=0

)

device= torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class ToxicClassiferModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.BERTLayer=AutoModel.from_pretrained("bert-base-uncased",return_dict=True)
        for param in self.BERTLayer.parameters():
            param.requires_grad = False
        self.dropoutLayer=nn.Dropout(0.3)
        self.fc=nn.Linear(768,1)

    def forward(self,input_ids,attention_mask,token_type_id):
        # print("[+]Forward")
        output=self.BERTLayer(input_ids,attention_mask,token_type_id,return_dict=True)
        output=self.dropoutLayer(output.pooler_output)
        output=self.fc(output)
        output=F.softmax(output,1)
        return output

model=ToxicClassiferModel()
model.to(device)

def loss_fn(outputs,target):
    return nn.BCEWithLogitsLoss()(outputs,target)

optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)


epochs=10

for epoch in range(epochs):
    val_loss=0
    train_loss=0
    model.train()

    for index,(inputs,target) in enumerate(train_data_loader):
        print(f'Batch {index}')
        dim1=inputs['input_ids'].shape[0]
        # print("[+]Dim ",dim1)
        # print("[+]Input_ids shape ",input_ids.shape)
        input_ids=inputs['input_ids'].view(dim1,512).to(device)
        attention_mask=inputs['attention_mask'].view(dim1,512).to(device)
        token_type_ids=inputs['token_type_ids'].view(dim1,512).to(device)
        target=target.view(dim1,1).float().to(device)
        optimizer.zero_grad()
        output=model(input_ids,attention_mask,token_type_ids)
        # output.float()
        loss=loss_fn(output,target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
      for index,(inputs,target) in enumerate(val_data_loader):
          print(f'Batch {index}')
          dim1=inputs['input_ids'].shape[0]
          # print("[+]Dim ",dim1)
          # print("[+]Input_ids shape ",inputs['input_ids'].shape)
          input_ids=inputs['input_ids'].view(dim1,512).to(device)
          attention_mask=inputs['attention_mask'].view(dim1,512).to(device)
          token_type_ids=inputs['token_type_ids'].view(dim1,512).to(device)
          # print("[+]Target shape ",target.shape)
          target=target.view(dim1,1).float().to(device)
          output=model(input_ids,attention_mask,token_type_ids)
          # output.float()
          loss=loss_fn(output,target)
          val_loss+=loss

    print(f'Epoch {epoch} Training_loss:{train_loss} and Val_loss:{val_loss}')