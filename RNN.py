import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

%load_ext autoreload
%autoreload 2
from utils import set_seed, create_metric_plots
from main import train_model
from loader import load_data_set, collact_batch, preprocess_review
set_seed(42)
BATCH_SIZE = 64
MAX_LEN = 100
EMBEDDING_DIM = 100
INPUT_DIM = EMBEDDING_DIM
REVIEW_LEN = 100
OUTPUT_DIM = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## DATA ##
data = pd.read_csv("IMDB Dataset.csv")
data.head() 

df = data.copy()
# Add a column with the length of each review
df['review_length'] = df['review'].apply(lambda x: len(x)/5)

# Sort the DataFrame by review length
sorted_df = df.sort_values(by='review_length', ascending=True)
sorted_df
sorted_df['review_length']
%load_ext autoreload
%autoreload 2
from torch.utils.data import DataLoader
import loader
import importlib

importlib.reload(loader)

train_data, test_data = load_data_set(load_my_reviews=False)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, collate_fn=collact_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False, collate_fn=collact_batch)
batch = next(iter(train_dataloader))
labels, embeddings, reviews = batch
print(labels.shape, embeddings.shape, len(reviews))
print(labels.dtype, embeddings.dtype)
review = ' '.join(reviews[0])
embeddings[0, :, :]
preprocess_review(reviews[0][0])
batch_size = 10
inputs = torch.randn(batch_size, REVIEW_LEN, EMBEDDING_DIM)
linear = nn.Linear(EMBEDDING_DIM, OUTPUT_DIM)
outputs = linear(inputs).mean(dim=1)
targets = torch.randint(0, OUTPUT_DIM, (batch_size,))
targets = F.one_hot(targets, num_classes=2).float()
print(inputs, outputs, targets, sep="\n")
loss = nn.CrossEntropyLoss()
output = loss(outputs, targets)



### Model ###
class RNN(nn.Module):
    def __init__(self, input_size=EMBEDDING_DIM, hidden_size=64, output_size=OUTPUT_DIM):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)

        for t in range(seq_length):
            combined = torch.cat((inputs[:, t, :], hidden), 1)
            hidden = self.layer_norm(F.tanh(self.i2h(combined)))

        logits = self.h2o(hidden)
        return logits, hidden

    def name(self):
        return "RNN"

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def train_step(self, inputs, targets, optimizer, criterion):
        optimizer.zero_grad()
        hidden = self.init_hidden(inputs.size(0)).to(inputs.device)
        logits, hidden = self(inputs, hidden)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Add gradient clipping
        optimizer.step()
        return loss.item(), logits

    def validation_step(self, inputs, targets, criterion):
        hidden = self.init_hidden(inputs.size(0)).to(inputs.device)
        logits, hidden = self(inputs, hidden)
        loss = criterion(logits, targets)
        return loss.item(), logits
#%%

rnn64 = RNN(hidden_size=64).to(device)
rnn64_metrics = train_model(rnn64, train_dataloader, test_dataloader, 
                          optimizer=torch.optim.Adam(rnn64.parameters(), lr=0.005), 
                          criterion=nn.CrossEntropyLoss(), 
                          num_epochs=5) 
#%%
rnn128 = RNN(hidden_size=128).to(device)
rnn128_metrics = train_model(rnn128, train_dataloader, test_dataloader, 
                          optimizer=torch.optim.Adam(rnn128.parameters(), lr=0.005), 
                          criterion=nn.CrossEntropyLoss(), 
                          num_epochs=5)

#%%
