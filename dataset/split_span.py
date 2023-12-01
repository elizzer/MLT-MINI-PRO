#%%
import pandas as pd

# %%
df=pd.read_csv("singleton_removed.csv")
df.head()
# %%
length=len(df)
df.head()
#%%
def break_list_on_jump(input_list):
    result = []

    if not input_list:
        return result

    current_start = input_list[0]

    for i in range(1, len(input_list)):
        if input_list[i] - input_list[i - 1] > 1:
            result.append((current_start, input_list[i - 1]))
            current_start = input_list[i]

    result.append((current_start, input_list[-1]))
    return result
# %%
from ast import literal_eval
df['span'] = df['span'].apply(literal_eval)

type(df['span'][0])
# %%
sl=[0]*length
for i in range(length):
    if len(df["span"][i])>0:
        sl[i]=break_list_on_jump(df["span"][i])
        print(f'{i}/{length}')
    else:
        sl[i]=[]
    # print(len(df["span"][i]))
sl
# %%
span=[86, 87, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
break_list_on_jump(span)
# %%
df['span_split_index']=sl
df.head()
# %%
df.to_csv("SpanSplit.csv",index=False,header=True)
# %%

df=pd.read_csv("SpanSplit.csv")
df['span'] = df['span'].apply(literal_eval)
df['span_split']= df['span_split'].apply(literal_eval)
df['span_split_index']= df['span_split_index'].apply(literal_eval)
length=len(df)
df.head()
# %%
for i in range(length):
    for j in range(len(df["span_split"][i])):
        df["span_split"][i][j][0]=(df["span_split"][i][j][0],'b')
        for k in range(1,len(df["span_split"][i][j][1:])+1):
            df["span_split"][i][j][k]=(df["span_split"][i][j][k],'i')
        print(df["span_split"][i][j])
df.to_csv("AnotatedSpanSplit.csv",index=False,header=True)

# %%
for i in range(length):
    for j in range(len(df["span_split"][i])):
        print(i,df["span_split"][i][j][0])

# %%
from transformers import BertTokenizer
# %%
toknizer=BertTokenizer.from_pretrained("bert-base-uncased")

# %%
ids=toknizer.encode("Adithya",add_special_tokens=False)
ids
# %%
