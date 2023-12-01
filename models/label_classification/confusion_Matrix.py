#%%
from model14pred import pred
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

# %%
pred("Hello")
# %%
toxic_df=pd.read_csv("singleton_removed.csv")
# %%
true=toxic_df["label"]
toxic_df["pred"]=0
true
#%%
# %%
temp=0
pred_values=[]

# %%
import time

def progress_bar(iterable, prefix='', suffix='', length=30, fill='█', print_end='\r'):
    total = len(iterable)
    
    # Initial print
    print(f'{prefix} |{0 * length}| {suffix}', end='', flush=True)
    
    for i, item in enumerate(iterable):
        yield item
        pred_values.append(pred(item))
        # Update progress bar
        progress = int(length * (i + 1) / total)
        bar = fill * progress + '-' * (length - progress)
        print(f'\r{prefix} |{bar}| {suffix}   {i}/{total}', end='', flush=True)
        time.sleep(0.1)  # Simulate some work being done

    print(print_end)  # Move to the next line after the loop

# Example usage
my_list = range(50)
for _ in progress_bar(toxic_df["comment_text"], prefix='Progress:', suffix='Complete', length=50):
    # Do some work here
    time.sleep(0.01)
# %%
pred_values=pred_values[:6500]
len(pred_values)
#%%
for i in range(6500):
    if(pred_values[i]=='Toxic'):
        pred_values[i]=1
    else:
        pred_values[i]=0
#%%
pred_values
# %%
cm=confusion_matrix(true[:6500],pred_values)
cm
# %%
corret=0
wrong=[]
for i in range(6500):
    if true[i]==pred_values[i]:
        corret+=1
    else:
        wrong.append(i)

print(corret)
# %%
print(len(wrong))

# %%
pred_values=[0]*100
pred_values
# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Replace this with your actual confusion matrix
conf_matrix = [[4706 , 316],[ 248, 4730]]


# Define labels
labels = ['Not Toxic', 'Toxic']

# Create a DataFrame from the confusion matrix
conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Create a heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# %%
