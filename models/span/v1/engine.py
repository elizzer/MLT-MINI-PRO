import torch
from tqdm import tqdm

def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    final_loss=0

    for data in tqdm(data_loader, total=len(data_loader)):
        # for K,V in data.items():
        #     data[K] = V.to(device)

        optimizer.zero_grad()
        loss=model(data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss+=loss.item()
    return final_loss/len(data_loader)


def train_fn(data_loader,model,optimizer,device,scheduler):
    model.eval()
    final_loss=0
    for data in tqdm(data_loader, total=len(data_loader)):
        for K,V in data.items():
            data[K] = V.to(device)
        loss=model(**data)
        final_loss+=loss.item()
    return final_loss/len(data_loader)

