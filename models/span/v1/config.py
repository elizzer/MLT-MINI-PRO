import transformers

MODEL_NAME="bert-base-uncased"
MAX_LEN=256
MODEL_PATH='model.bin'
EPOCH=1
TOKENIZER=transformers.BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)
BATCH_SIZE=12
LSTM_HIDDEN_STATE=128
NUM_TAGS=3
DEVICE="cuda"
TAG_TO_IX={"B": 1, "I": 2, "O": 0}