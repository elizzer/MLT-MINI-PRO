import config
import torch
import transformers
import torch.nn as nn
from torchcrf import CRF
# from bi_lstm_crf import CRF

import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

def loss_fn(output,target,mask):
    lfn=nn.CrossEntropyLoss()





class NERModel(nn.Module):
    def __init__(self):
        super(NERModel,self).__init__()

        self.bert=transformers.BertModel.from_pretrained(config.MODEL_NAME)
        self.bert_drop_1=nn.Dropout(0.3)
        self.lstm= nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=config.LSTM_HIDDEN_STATE,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.LSTM_fc=nn.Linear(config.LSTM_HIDDEN_STATE * 2, config.NUM_TAGS)

        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')


        self.crf = CRF(config.NUM_TAGS)

        # self.bert_drop_2=nn.Dropout(0.3)

    def compute_outputs(self,data):
        out=self.bert(data["input_ids"],attention_mask=data["attn_mask"],token_type_ids=data["token_type_ids"])
        out=out.last_hidden_state
        out,_=self.lstm(out)
        out=self.LSTM_fc(out)

        return out

    def forward(self,data):
        out=self.compute_outputs(data)
        return -self.crf(out, data["target_tags"])
    
    def predict(self, sentences):
        # Compute the emission scores, as above.
        scores = self.compute_outputs(sentences)

        return torch.tensor(self.crf.decode(scores)).view(config.BATCH_SIZE,config.MAX_LEN)
    
    