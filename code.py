
import csv
import pandas as pd

#reducing only 10000 datarows
ds_pointer = pd.read_csv("ntoxic_jigsaw.csv",encoding=' utf-8')
less_data=ds_pointer.head(10000)
less_data["label"]=0
less_data.to_csv("ntoxic.csv",mode='w',index=False)


#Non toxic words removal from span dataset

import pandas as pd
df = pd.read_csv('toxic_spans.csv') 
exclude_conditions = (
    (df['probability'] != '{}') 
)
filtered_df = df[exclude_conditions]
print(len(filtered_df))
filtered_df.to_csv("ntoxic-tsd1.csv",mode='w',index=False)


#cleaning span dataset

file_path='ntoxic-tsd1.csv'
out_file_path='ntoxic-tsd.csv'
header_dict={}
needed_cols=[5,1]

try:
    with open(file_path, 'r', newline='',encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        header = next(csv_reader)
        
        print("CSV Header:")
        print(header)
        for head,i in zip(header,range(0,header.__len__())):
            header_dict[head]=i
        with open(out_file_path,'w', newline='') as outFile:
            csv_writer=csv.writer(outFile)
            csv_writer.writerow([ header[cols] for cols in needed_cols])
            i=0
            for row in csv_reader:
                try:
                    csv_writer.writerow([ row[cols] for cols in needed_cols])
                    print(f"[+]Copied {i}th line..")
                    i+=1

                except Exception as e:
                    print(f"[*]Exception on line {i} is {e}")

except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


#adding label column and renaming column

df = pd.read_csv("ntoxic-tsd.csv",encoding=' Windows-1252')
df['label']=1
df.columns = ['comment_text','span','label']
df.to_csv("ntoxic1.csv",mode='w',index=False)

#curating two datasets

df1 = pd.read_csv('ntoxic.csv')
df2 = pd.read_csv('ntoxic1.csv')
combined_df = pd.concat([df1, df2], ignore_index=True)
combined_df = combined_df.drop_duplicates()
shuffled_df = combined_df.sample(frac=1.0)
shuffled_df.to_csv('dataset.csv', index=False)



#print(ds_pointer.head())
