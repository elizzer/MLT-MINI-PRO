import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train_loss.csv',header=None) 

plt.plot(df.iloc[:, 0], df.iloc[:, 1])  # Assuming the first column is X and the second is Y
plt.title('Train Loss Graph')  # Replace with your desired title
plt.xlabel('Batch')  # Replace with your desired X-axis label
plt.ylabel('Loss')  # Replace with your desired Y-axis label
plt.grid(True)
plt.show()