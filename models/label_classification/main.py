#%%
import torch
import torch.nn as nn
import transformers
import tez
from transformers import AdamW,get_linear_schedule_with_warmup
from sklearn import metrics
import pandas as pd
#%%
tok=transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=False
        )
mod=transformers.BertModel.from_pretrained(
            "bert-base-uncased"
)
#%%
class BertDataset:

    def __init__(self,text,target,max_len=512):
        self.text=text
        self.targer=target
        self.max_len=max_len

        self.tokenizer=transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=False
        )

        # self.tokenizer=tok

    def __len__(self):
        return len(self.targer)
    
    def __getitem__(self,index):
        text=str(self.text[index])
        target=self.targer[index]

        tokens=self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        return {
            "ids":torch.tensor(tokens["input_ids"],dtype=torch.long),
            "attn_mask":torch.tensor(tokens["attention_mask"],dtype=torch.long),
            "token_id":torch.tensor(tokens["token_type_ids"],dtype=torch.long),
            "target":torch.tensor(target,dtype=torch.float),   
        }
    
class TextModel(tez.Model):
    def __init__(self,num_classes,num_train_steps):
        super().__init__()
        self.bert=transformers.BertModel.from_pretrained(
            "bert-base-uncased",
            return_dict=False
        )

        # self.bert=mod

        self.bert_drop=nn.Dropout(0.1)
        self.fc=nn.Linear(768,num_classes)
        self.num_train_steps=num_train_steps

    def fetch_optimizer(self):
        return AdamW(self.parameters(),lr=5e-05 )

    def fetch_scheduler(self):
        sch=get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps
        )

        return sch
    
    def loss(self,output,target):
        return nn.BCEWithLogitsLoss()(output,target.view(-1,1))
    
    def monitor_metrics(self,output,target):
        output=torch.sigmoid(output).cpu().detach().numpy()>=0.5
        target=target.cpu().detach().numpy()
        return {
            "accuracy":metrics.accuracy_score(target,output)
        }

    def forward(self,ids,attn_mask,token_type_ids,target=None):
        _,x=self.bert(ids,attention_mask=attn_mask,token_type_ids=token_type_ids)
        x=self.bert_drop(x)
        x=self.fc(x)

        if target is not None:
            loss=self.loss(x,target)
            metrics=self.monitor_metrics(x,target)
            return x,loss,metrics
        return x, 0,{}

def train():
    df=pd.read_csv("singleton_removed.csv")
    df=df[["comment_text","label"]]
    df_train=df[:1000].reset_index(drop=True)
    df_valid=df[1000:1200].reset_index(drop=True)
    print(df_train.head())
    print(df_valid.head())

    train_dataset=BertDataset(df_train["comment_text"],df_train['label'])
    valid_dataset=BertDataset(df_valid["comment_text"],df_valid['label'])
    
    n_train_steps=int(len(df_train)/(32*5))
    model=TextModel(num_classes=1,num_train_steps=n_train_steps)

    es=tez.callbacks.EarlyStopping(monitor='valid_loss',patience=3, model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        device='cuda',
        epochs=5,
        train_bs=32,
        callbacks=[es]
        )
#%%
if __name__=="__main__":
    train()
# %%
