import pandas as pd

# Load your dataset into a pandas DataFrame (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('dataset/singleton_removed.csv')

# Assuming 'label' is the column you're interested in
# Count of rows where the 'label' column value is 1
count_label_1 = (df['label'] == 1).sum()

print("Count of rows where 'label' is 1:", count_label_1)
print("Count of rows where 'label' is 0:", (df['label']).sum())

