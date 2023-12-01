import csv
import pandas as pd

# file_path='dataset/all_data.csv/all_data.csv'
# out_file_path='dataset/curated.csv'
# header_dict={}
# needed_cols=[1,13,14,15,16,17,18,19]
# try:
#     with open(file_path, 'r', newline='',encoding='utf-8') as csv_file:
#         csv_reader = csv.reader(csv_file)
        
#         # Read the header row
#         header = next(csv_reader)
        
#         # Print the header
#         print("CSV Header:")
#         print(header)
#         # for head,i in zip(header,range(0,header.__len__())):
#         #     header_dict[head]=i
#         # with open(out_file_path,'w', newline='') as outFile:
#         #     csv_writer=csv.writer(outFile)
#         #     csv_writer.writerow([ header[cols] for cols in needed_cols])
#         #     i=0
#         #     for row in csv_reader:
#         #         try:
#         #             csv_writer.writerow([ row[cols] for cols in needed_cols])
#         #             print(f"[+]Copied {i}th line..")
#         #             i+=1
#         #             if i>30000:
#         #                 break
#         #         except Exception as e:
#         #             print(f"[*]Exception on line {i} is {e}")


                
            
# except FileNotFoundError:
#     print(f"The file '{file_path}' was not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

#simmple summation



ds_pointer = pd.read_csv("dataset/curated.csv",encoding=' Windows-1252')
columns_to_sum = ['toxicity', 'severe_toxicity', 'obscene','sexual_explicit','identity_attack','insult','threat']
ds_pointer['toxicity_score_un'] = ds_pointer[columns_to_sum].sum(axis=1)

min_value = ds_pointer['toxicity_score_un'].min()
max_value = ds_pointer['toxicity_score_un'].max()
print(f"[=]The min value is {min_value}")
print(f"[=]The max value is {max_value}")
ds_pointer['toxicity_score_normalized'] = ds_pointer['toxicity_score_un'] / max_value
print('[+]Normalization done...')
ds_pointer["class"]=ds_pointer["toxicity_score_normalized"]<=0.2
print("[=]Class defined...")
print("[=]CSV writting started")
ds_pointer.to_csv("dataset/curated2.csv",mode='w', index=False)
print(f"[=]CSV written")

ntoxic=ds_pointer[ds_pointer['class']==True]
ntoxic=ntoxic[["comment_text"]]
ntoxic["span"]="[]"
print(f'[+]Non toxic data filtred...{ntoxic.__len__()}')
ntoxic.to_csv("dataset/ntoxic_jigsaw.csv",mode='w',index=False)
print('[=]CSV written...')



# ds_pointer=[row for row in ds_pointer if row["toxicity_score_normalized"]<=0.2]
# ds_pointer['class']=[int(row["toxicity_score_normalized"]<=0.2) for row in ds_pointer]
# print(ds_pointer.dtype())
# Min-max scaling to normalize the 'overall_toxicity' column
# min_value = ds_pointer['toxicity_score_un'].min()
# max_value = ds_pointer['toxicity_score_un'].max()
# ds_pointer['toxicity_score'] = (ds_pointer['toxicity_score_un'] - min_value) / (max_value - min_value)
# print(ds_pointer.head())

#  # Filter rows where 'toxicity_score' is less than 0.2 and then count them
# count_less_than_0_2 = (ds_pointer['toxicity_score'] <= 0.2).sum()
# print(f"Count of rows where 'toxicity_score' is less than 0.2: {count_less_than_0_2}")


# #filtering only non toxic data
# ntoxic_ds = ds_pointer.loc[ds_pointer['toxicity_score'] <= 0.2].iloc[:16000]
# row_count = ntoxic_ds.shape[0]
# print(row_count)
# print((ntoxic_ds['toxicity_score'] <= 0.2).sum())

# #wriring in new csv file
# output_path = 'dataset/ntoxic_jigsaw.csv'

# # Use mode='a' to append to an existing file or mode='w' to overwrite it (default is 'w')
# ntoxic_ds.to_csv(output_path, mode='a', index=False, header=True) 
# ds = pd.read_csv(output_path,encoding=' Windows-1252')
# row_count = ds.shape[0]
# print(row_count)



# print(header_dict)




#reducing only 10000 datarows

# import pandas as pd

# df=pd.read_csv("toxic_spans.csv")
# i=0
# count=0
# for i in range(len(df['position'])):
#     if len(df['position'][i])==0:
#         print("[=]Ntoxic...")
#         count+=1
#     i+=1
# print(count)