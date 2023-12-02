
import dataset
import pandas as pd
from tqdm import tqdm
import model
import config
import engine
import torch
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import csv
# datasetPath="G:\\acadamics\miniProject\dataset\SpanSplit.csv"

# df=pd.read_csv(datasetPath)
# df=df[["comment_text","span_split_index"]]
# df.head()

data_loader=dataset.data_loader

print('[+]Dataset sequence success')

# for i in tqdm(range(len(df))):
#     data.__getitem__(i)

# data=[data]
# print('[+]Structured data acruired')
Model=model.NERModel()
Model.to(config.DEVICE)
print("[+]Model defined")

total_steps = len(data_loader) * config.EPOCH


optimizer = AdamW(Model.parameters(), lr=2e-5, weight_decay=0.01)
   

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * total_steps,  # Adjust warm-up steps
                                            num_training_steps=total_steps)

# engine.train_fn(data_loader,Model,optimizer=optimizer,device=config.DEVICE,scheduler=scheduler)
final_loss=0
Model.train()
train_loss=[]
Model.load_state_dict(torch.load('Model-recent.pth'))
for epoch in range(config.EPOCH):
    final_loss=0

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        loss=Model(data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss+=loss.item()
        train_loss.append(loss.item())
        # print(f"[***]Model output of batch {i} with loss {loss}")
        with open("Train_loss.csv", 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i,loss.item()])
        if i%50==0:
            torch.save(Model.state_dict(), 'Model-recent.pth')

    

# print("[++++]Predecting...")
# for batch in data_loader:
#     output=Model.predict(batch)
#     print("[***]Model output",output.shape)
#     print(type(output))
#     print(output)
#     break