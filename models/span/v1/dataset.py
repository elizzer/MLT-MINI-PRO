import torch
# from torch.Datasets import Datasets
from transformers import BertTokenizer
from ast import literal_eval
import config
from torch.utils.data import Dataset,random_split,DataLoader
import pandas as pd


def annotate_sentence(sentence, indices):
    # Sort the indices to handle unordered cases
    indices.sort()

    # Initialize a list to store the annotated tuples
    annotated_tuples = []

    current_index = 0

    for start, end in indices:
        # Add 'o' annotations for words between the current_index and start
        words_between = sentence[current_index:start].split()
        annotated_tuples.extend((word, 'o') for word in words_between if word)

        # Add 's' annotations for words between start and end
        words_within = sentence[start:end + 1].split()
        annotated_tuples.extend((word, 's') for word in words_within if word)

        current_index = end + 1

    # Add 'o' annotations for words after the last end index
    remaining_words = sentence[current_index:].split()
    annotated_tuples.extend((word, 'o') for word in remaining_words if word)

    return annotated_tuples


class SpanTaggingDataset(Dataset):
    def __init__(self,df,max_len=config.MAX_LEN):
        self.text=df["comment_text"]
        self.span_index=df['span_split_index'].apply(literal_eval)
        self.tokenizer=config.TOKENIZER
        self.maxlen=max_len
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,index):
        # Get a local instalce of the text and its span
        text=self.text[index]
        span=self.span_index[index]
        
        #get the annotated span
        tagged_sentence=annotate_sentence(text,span)

        # Define the empty list to be filled with tokens and details
        target_tags=[]
        input_ids =[]
        attn_mask=[]
        token_types_ids=[]
        prev=0

        # loop through each word in the tagged_sentence which did tokenize with space
        for word in tagged_sentence:

            # Tokinize each word with wordpice tokenizer
            temp_tokens=self.tokenizer.encode(word[0], add_special_tokens=False)

            # Temp tag store the tag for the current word
            temp_tags=[]

            # If the prev token is "O" and current word is in span
            if prev==0 and word[1]=='s':
                # Make the first token "B-T"
                temp_tags.append(1)
                
                # Make the other token "I-T"
                temp_tags.extend([2]*(len(temp_tokens)-1))

                prev=2 # set the prev is I-T 

            # If prev token is "B-T" or "I-T" and current word in toxic span
            elif (prev==1 or prev==2) and word[1]=='s':

                # Make the tags of the token of the word "I-T"
                temp_tags.extend([2]*(len(temp_tokens)))

                prev=2 # set the prev is I-T 
            
            # If the word is not in toxic span make the tag "O"
            else:
                temp_tags.extend([0]*len(temp_tokens))   

                prev=0 # set the prev is O

            input_ids.extend(temp_tokens) # In the end of the loop extend the temp_tokens to input_ids

            # Bring back the temp_tokens to empty
            temp_tokens=[] 
            
            # Add the temp_tags to the target_tags
            target_tags.extend(temp_tags)
            temp_tags=[]
        if len(input_ids)>=config.MAX_LEN:
            input_ids=input_ids[:(config.MAX_LEN-2)]
        
        input_ids=[101]+input_ids+[102] #Once all the word of a sentence are done, added the start and end characters
        target_tags=[0]+target_tags+[0] # make the tag for start and end charcter "0"
   
        pad_len=self.maxlen-len(input_ids) # Find the remaining length to max_len which is to be padded

        attn_mask=[1]*len(input_ids) # Add 1 to the index having actial words
        token_types_ids=[0]*len(input_ids) # Add 0 token_types_ids for the index having actual words/tokens
        
        input_ids=input_ids+([0]*pad_len) # padd the input token ids with 0
        attn_mask=attn_mask+([0]*pad_len) # padd the attnMask token ids with 0

        token_types_ids=token_types_ids+([0]*pad_len)
        target_tags=target_tags+([0]*pad_len)
        


        return {
            "input_ids":torch.tensor(input_ids[:config.MAX_LEN],dtype=torch.long).to(config.DEVICE),
            "attn_mask":torch.tensor(attn_mask[:config.MAX_LEN],dtype=torch.long).to(config.DEVICE),
            "token_type_ids":torch.tensor(token_types_ids[:config.MAX_LEN],dtype=torch.long).to(config.DEVICE),
            "target_tags":torch.tensor(target_tags[:config.MAX_LEN],dtype=torch.long).to(config.DEVICE)
        }
    
datasetPath="G:\\acadamics\miniProject\dataset\SpanSplit.csv"

df=pd.read_csv(datasetPath)
df=df[["comment_text","span_split_index"]]

print("[+]Dataframe done")

ds=SpanTaggingDataset(df)

print("[+]Dataset done")

# # print(ds[0])
# for i in range(len(ds)):
#     a=ds[i]["input_ids"].shape
#     b=ds[i]["attn_mask"].shape
#     c=ds[i]["token_type_ids"].shape
#     d=ds[i]["target_tags"].shape
#     print(f"[+]Shape of data {i} ==> Input_ids {a} attn_mask {b} token_type_ids {c} target_tags {d}")

data_loader=DataLoader(ds,config.BATCH_SIZE,shuffle=True)

print("[+]Dataloader done")

