#%%
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from tez import Tez, TezConfig
import tez
from tez.callbacks import EarlyStopping
# %%
class args:
    model = "bert-base-uncased"
    epochs = 20
    batch_size = 24
    learning_rate = 5e-5
    train_batch_size = 24
    valid_batch_size = 24
    max_len = 256
    accumulation_steps = 1
    checkpoint_path="/content/drive/MyDrive/MiniProject/model/model_recent_checkpoint.pth"
# %%

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

input = Input(shape=(MAX_LEN,))

model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN, mask_zero=True)(input) #default: 20-dim embedding

model = Bidirectional(LSTM(units=50, return_sequences=True,recurrent_dropout=0.1))(model)  # variational biLSTM

model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer

crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
out = crf(model)  # output
# %%
