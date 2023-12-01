
import re
import pandas as pd


#removing urls 

df = pd.read_csv('dataset.csv')

def remove_url(text,span):
    matches = [match for match in re.finditer(r'\bhttps?://\S+\b', text)]

    offset=0
    if(matches):
        for match in matches:
            start_index=match.start()-offset
            end_index=match.end()-offset
            length=end_index-start_index
            offset+=length
            new_span=[]
            for s in span:
                if s > start_index:
                    new_span.append(s-length)
                else:
                    new_span.append(s)
            span=new_span
        text=re.sub(r'http\S+', '', text)
        return text,span
    return text,span
for i in range(len(df["span"])):
    df["comment_text"][i],df["span"][i]=remove_url(df["comment_text"][i],eval(df["span"][i]))

df.to_csv("dataset1.csv",mode='w',index=False)


#remove repeated special characters

DATASET_LEN=0

df = pd.read_csv('dataset1.csv')
DATASET_LEN=len(df["span"])
def remove_repeated_characters(text,span):
    pattern = r'[!@#$%^&*]+'
    matches = [match for match in re.finditer(pattern, text) if match.end() - match.start() != 1]
    offset=0
    if (matches != []):
        try:
            print(matches)
        except Exception as e:
            print("Error with",e)
        
    if matches:
        for match in matches:
            start_index=match.start()-offset
            end_index=match.end()-offset
            length=end_index-start_index
            offset+=length
            new_span=[]
            for s in span:
                try:
                    if s > start_index:
                        new_span.append(s-length+1)
                    else:
                        new_span.append(s)
                except Exception as e:
                    print(text)
            span=new_span
        text=re.sub(r'([!@#$%^&*()_+=\-{}[\]:;"\'|<>,./?\\])\1+', r'\1', text)
    return text,span

def remove_empty_comments(d):
    result=[]
    for i in range(DATASET_LEN):
        if d["comment_text"][i]!="":
            result.append(d["comment_text"][i])
    return pd.DataFrame(result)
    
print("[+]Removing empty spaces...")
remove_empty_comments(df)
print("[+]Removed empty space...")
for i in range(DATASET_LEN):
    print(f'[+]Index {i}/{DATASET_LEN}')
    try:
        df["comment_text"][i],df["span"][i]=remove_repeated_characters(df["comment_text"][i],eval(df["span"][i]))
    except Exception as e:
        print("[+] Error in index ",i)

df.to_csv("dataset/dataset_removed_repeated_characters.csv",mode='w',index=False)

#remove span indicating empty space

df=pd.read_csv('dataset/dataset_removed_long_charcter.csv')
DATASET_LEN=len(df["span"])

def remove_empty_span(text,spans):
    for x in spans:
        if text[x] == ' ':
            spans.remove(x)

    return (text,spans)

for i in range(DATASET_LEN):
    try:
        df["comment_text"][i],df["span"][i]=remove_empty_span(df["comment_text"][i],eval(df["span"][i]))
    except Exception as e:
        print("[+] Error in index ",i)

df.to_csv("dataset/space_removed.csv",mode='w',index=False)

#remove singleton


# Define the function a()
def a(text, span):
    for s in span:
        if s < len(text) and text[s] != " " :
            w=[" ",",","!","?",".","'","\"",";",":","\/"]
            while s < len(text) and text[s] not in w:
                if s in span:
                    pass
                else:
                    print(span)
                    print(text[s-1])
                    return False
                s += 1
        else:
            return False
    return True


df = pd.read_csv('dataset/space_removed.csv')  


df['result'] = df.apply(lambda row: a(row['comment_text'], eval(row['span'])), axis=1)


filtered_df = df[df['result']]

filtered_df.to_csv('dataset/singleton_removed.csv',mode='w',index=False)


#Count of rows where 'label' is 1: 9816
#Count of rows where 'label' is 0: 9971
