#%%
from transformers import BertTokenizer,BertForTokenClassification
import torch
# %%
text = "Mark Hume has always been the stereotypical Vancouverite. Scream how logging must be stopped, while living in a house made of wood. It's so far past hypocritical to be ridiculous."
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
inputs = tokenizer(text, return_tensors="pt")
# %%
tokens
# %%
current_index=0
token_indices=[]
text=text.lower()
print(text)
for word in tokens:
  word=word.split("#")[-1]
  s_i=text.find(word,current_index)
  e_i=s_i+len(word)
  token_indices.append((s_i,e_i,word))
  current_index=e_i
token_indices
# %%
span=[[(149, 'b'), (150, 'i'), (151, 'i'), (152, 'i'), (153, 'i'), (154, 'i'), (155, 'i'), (156, 'i'), (157, 'i'), (158, 'i'), (159, 'i'), (160, 'i')], [(168, 'b'), (169, 'i'), (170, 'i'), (171, 'i'), (172, 'i'), (173, 'i'), (174, 'i'), (175, 'i'), (176, 'i'), (177, 'i')]]
# %%
