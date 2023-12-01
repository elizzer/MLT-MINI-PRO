#%%

from transformers import BertModel, BertTokenizer,AutoModel
import torch

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#%%
tokenized=tokenizer("swiming")
input_Ids=tokenized

print(input_Ids)
bert_embedding= bert.embeddings( torch.tensor(input_Ids))
print(bert_embedding)

# %%
print(torch.tensor(input_Ids))
# %%
text = ["this is a bert model tutorial adithya hello are you ther","He is swiming"]

# encode text
sent_id = tokenizer(text,padding=True,return_token_type_ids=False)
mask=torch.tensor(sent_id['attention_mask'])
sent_id=torch.tensor(sent_id['input_ids'])
# %%
print(sent_id)
print(sent_id.shape)
# %%
for param in bert.parameters():
    param.requires_grad = False
# %%
output = bert(sent_id,mask)

print(output[0])
# print(_,output)
# %%
import torch
import torch.nn as nn

embedded=nn.Embedding(20,30)

out=embedded(sent_id)
# %%
