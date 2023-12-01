#%%
import torch

# Creating a 1D LongTensor
tensor = torch.LongTensor([1, 2, 3, 4, 5])

# Accessing and printing the tensor
print(tensor)
# %%
import torch
import torch.nn as nn

# Define the size of the vocabulary and the dimension of the embedding
vocab_size = 10000
embedding_dim = 10

# Create an embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Input data (batch of sequences of word indices)
input_data = torch.LongTensor([[1, 4, 6, 2,5], [5, 3, 7, 0,5], [5,5,3, 7, 0], [5,6, 3, 7, 0]])
# input_data=torch.randint([4,4])
print(input_data)
# Pass input data through the embedding layer to get continuous vectors
embedded_data = embedding_layer(input_data)

# embedded_data will be a tensor of shape (batch_size, sequence_length, embedding_dim)
print(embedded_data[0])
# %%
from transformers import RobertaTokenizer, RobertaModel

# Load the pre-trained RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the pre-trained RoBERTa model
model = RobertaModel.from_pretrained('roberta-base')

# Input text
text = "RoBERTa is a robust variant of BERT."

# Tokenize the text
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Convert input to a PyTorch tensor
input_ids = torch.tensor(input_ids)

# Pass the input through the RoBERTa model
outputs = model(input_ids.unsqueeze(0))
print(outputs)
# The outputs contain the contextualized embeddings
# outputs[0] contains the hidden states, and outputs[1] contains the pooled representation

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import torchcrf

# Define the model
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, num_labels, hidden_dim):
        super(MultiTaskModel, self).__init()

        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define the BiLSTM layer
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        
        # Define linear layers for classification
        self.classification_head = nn.Linear(hidden_dim * 2, num_classes)
        
        # Define the CRF layer for sequence tagging
        self.crf = torchcrf.CRF(num_labels)
        
    def forward(self, input_text):
        # Tokenize the input text and get embeddings from BERT
        input_ids = self.tokenizer(input_text)['input_ids']
        _, output = self.bert(torch.tensor(input_ids))
        # output['last_hidden_state'] contains the contextual embeddings
        
        # Pass the embeddings through the BiLSTM layer
        bilstm_output, _ = self.bilstm(output['last_hidden_state'])
        
        # Apply the classification head to get class labels
        class_labels = self.classification_head(bilstm_output)
        
        return class_labels

# Create an instance of the model
model = MultiTaskModel(num_classes=Ct, num_labels=Yt, hidden_dim=256)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for input_batch, labels in dataloader:
        # Forward pass
        class_labels = model(input_batch)
        
        # Calculate loss and perform backpropagation
        loss = loss_function(class_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# For CRF, you would need to integrate it into your training and decoding process.
# Please refer to the torchcrf documentation for CRF-specific details.

#%%
import re

sentence = "Hello!!!!!! This is a $t3st @of special characters. #GoodLuck!!"
pattern = r'[!@#$%^&*]+'
matches = re.finditer(pattern, sentence)
print(matches)
for match in matches:
    print(match)

# %%
