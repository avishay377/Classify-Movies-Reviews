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
class MLP(nn.Module):
    def __init__(self, input_size=100, hidden_size=64):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        sub_scores = self.mlp(x)  # (batch_size, seq_len, 2)
        scores = torch.mean(sub_scores, dim=1)  # (batch_size, 2)
        # return scores
        return scores, sub_scores

    def train_step(self, inputs, targets, optimizer, criterion):
        optimizer.zero_grad()
        logits, _ = self(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item(), logits

    def validation_step(self, inputs, targets, criterion):
        logits, _ = self(inputs)
        loss = criterion(logits, targets)
        return loss.item(), logits
    
    def name(self):
        return "MLP"

#%%
mlp = MLP(hidden_size=128).to(device)
mlp_metrics = train_model(mlp, train_dataloader, test_dataloader, 
                          optimizer=torch.optim.Adam(mlp.parameters(), lr=0.02), 
                          criterion=nn.CrossEntropyLoss(), 
                          num_epochs=2)


#%%
mlp64 = MLP(hidden_size=64).to(device)
mlp64_metrics = train_model(mlp64, train_dataloader, test_dataloader, 
                          optimizer=torch.optim.Adam(mlp64.parameters(), lr=0.005), 
                          criterion=nn.CrossEntropyLoss(), 
                          num_epochs=5)
torch.save(mlp64, mlp64.name() + '_64'+ ".pth")

#%%

mlp128 = MLP(hidden_size=128).to(device)
mlp128_metrics = train_model(mlp128, train_dataloader, test_dataloader, 
                          optimizer=torch.optim.Adam(mlp128.parameters(), lr=0.05), 
                          criterion=nn.CrossEntropyLoss(), 
                          num_epochs=5)
torch.save(mlp128, mlp128.name() + '_128'+ ".pth")
#%%
create_metric_plots(mlp_metrics, figsize=(10, 15))

