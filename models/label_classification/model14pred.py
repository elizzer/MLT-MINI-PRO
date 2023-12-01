import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class args:
    model = "bert-base-uncased"
    epochs = 20
    batch_size = 24
    learning_rate = 5e-5
    train_batch_size = 24
    valid_batch_size = 24
    max_len = 256
    accumulation_steps = 1
    checkpoint_path="G:\\acadamics\miniProject\models\label_classification\model_recent_checkpoint.pth"

class TextModel_pred(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 5e-5

        config = AutoConfig.from_pretrained(args.model)

        config.update(
            {
                "output_hidden_states": True,
                # "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": 1,
            }
        )
        self.bert=AutoModel.from_pretrained(args.model, config=config)
        self.bert_drop=nn.Dropout(0.1)
        self.fc=nn.Linear(768,num_classes)

    def forward(self,ids,attn_mask,token_id,target=None):
        x=self.bert(ids,attention_mask=attn_mask,token_type_ids=token_id)
        x=x.pooler_output
        #x=self.bert_drop(x)
        x=self.fc(x)
        x=torch.sigmoid(x)
        # x=F.softmax(x, dim=1)
        return x
    
model_pred = TextModel_pred(
    # model_name=args.model,
    num_classes=1,
)

print("[+]Model init success...")
model_pred.load_state_dict(torch.load(args.checkpoint_path))
print("[+]Model load success...")

model_pred.to("cuda")
print("[+]Model to cuda success...")


def pred(text):
  def tokenn(text,max_len=args.max_len):
          tokenizer=AutoTokenizer.from_pretrained(
              "bert-base-uncased",
              do_lower_case=False
          )
          tokens=tokenizer(
              text,
              None,
              add_special_tokens=True,
              max_length=max_len,
              padding='max_length',
              truncation=True
          )

          return {
              "ids":torch.tensor(tokens["input_ids"],dtype=torch.long),
              "attn_mask":torch.tensor(tokens["attention_mask"],dtype=torch.long),
              "token_id":torch.tensor(tokens["token_type_ids"],dtype=torch.long),
          }

  params=tokenn(text)
  ids=params['ids'].view(1,256)
  attn_mask=params['attn_mask'].view(1,256)
  token_id=params['token_id'].view(1,256)
  ids,attn_mask,token_id=ids.to("cuda"),attn_mask.to("cuda"),token_id.to("cuda")
  output=model_pred.forward(ids,attn_mask,token_id)
  if output<0.5:
    return "Non toxic"
  else:
    return "Toxic"

if __name__ =="__main__":
    while(True):
        text=input("Enter the string to be classiffied:")
        result=pred(text)
        print(result)
        print("Wanna try again")

# print('[+]Start predicting...')
# pred_length=10000
# toxic_df=pd.read_csv("singleton_removed.csv")
# true=toxic_df["label"]
# toxic_df["pred"]=0

# pred_values=[0]*pred_length

# for i in range(pred_length):
#     pred_values[i]=pred(toxic_df["comment_text"][i])
#     print(f"Completed {i}/{pred_length}: predicted Value {pred_values[i]}>>{true[i]} ")

# print("Finished predecting...")

# cm=confusion_matrix(true[:pred_length],pred_values)
# print("[+]Calculating confusion matrix ")
# print(cm)
