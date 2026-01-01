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


class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)
        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x

class RestrictedSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_size, window_size=5):
        super(RestrictedSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        
        self.W_q = MatMul(input_dim, hidden_size, use_bias=False)
        self.W_k = MatMul(input_dim, hidden_size, use_bias=False)
        self.W_v = MatMul(input_dim, hidden_size, use_bias=False)
        self.W_o = MatMul(hidden_size, input_dim, use_bias=False)  # Add this line

        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.layer_norm(x)
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Prepare padded sequence for restricted attention
        padded_K = F.pad(K, (0, 0, self.window_size, self.window_size), mode='constant', value=0)
        padded_V = F.pad(V, (0, 0, self.window_size, self.window_size), mode='constant', value=0)
        
        # Initialize output tensor
        # output = torch.zeros_like(x)
        output = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device)
        atten_weights = torch.zeros(batch_size, seq_len, 2 * self.window_size + 1, device=x.device)

        
        for i in range(seq_len):
            # Extract window
            window_K = padded_K[:, i:i+2*self.window_size+1, :]
            window_V = padded_V[:, i:i+2*self.window_size+1, :]
            
            # Compute attention scores
            attention = torch.matmul(Q[:, i:i+1, :], window_K.transpose(-2, -1)) / self.sqrt_hidden_size
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention, dim=-1)
            atten_weights[:, i, :] = attention_weights.squeeze(1)
            
            # Compute weighted sum
            weighted_sum = torch.matmul(attention_weights, window_V)
            output[:, i:i+1, :] = weighted_sum
        
        return output, atten_weights

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # (batch_size, sequence_length, d_model)

class SelfAttentionMLP(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, window_size=5):
        super(SelfAttentionMLP, self).__init__()
        self.positional_encoding = PositionalEncoding(input_size)
        self.self_attention = RestrictedSelfAttention(input_size, hidden_size, window_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def name(self):
        return "SelfAttentionMLP"

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

    def forward(self, x):
        x = self.positional_encoding(x)
        x, atten_weights = self.self_attention(x)
        sub_scores = self.mlp(x)  # (batch_size, seq_len, 2)
        scores = torch.mean(sub_scores, dim=1)  # (batch_size, 2)
        return scores, sub_scores
#%%

# Create and train the model
self_attention_mlp_ = SelfAttentionMLP(hidden_size=128).to(device)
self_attention_mlp_metrics_ = train_model(self_attention_mlp, train_dataloader, test_dataloader, 
                                         optimizer=torch.optim.Adam(self_attention_mlp.parameters(), lr=0.001), 
                                         criterion=nn.CrossEntropyLoss(), 
                                         num_epochs=10)


#%%
# Create and train the model
self_attention_mlp_pos = SelfAttentionMLP(hidden_size=128).to(device)
self_attention_mlp_metrics_pos = train_model(self_attention_mlp_pos, train_dataloader, test_dataloader, 
                                         optimizer=torch.optim.Adam(self_attention_mlp_pos.parameters(), lr=0.001), 
                                         criterion=nn.CrossEntropyLoss(), 
                                         num_epochs=10)



#%%
torch.save(self_attention_mlp_pos, self_attention_mlp_pos.name() + '_128'+ ".pth")
self_attention_mlp_metrics_pos_ext = train_model(self_attention_mlp_pos, train_dataloader, test_dataloader, 
                                         optimizer=torch.optim.Adam(self_attention_mlp_pos.parameters(), lr=0.0005), 
                                         criterion=nn.CrossEntropyLoss(), 
                                         num_epochs=10)

#%%
self_attention_mlp_metrics_pos_ext1 = train_model(self_attention_mlp_pos, train_dataloader, test_dataloader, 
                                         optimizer=torch.optim.Adam(self_attention_mlp_pos.parameters(), lr=0.0001), 
                                         criterion=nn.CrossEntropyLoss(), 
                                         num_epochs=10)

#%%
self_attention_mlp_metrics_pos_ext2 = train_model(self_attention_mlp_pos, train_dataloader, test_dataloader, 
                                         optimizer=torch.optim.Adam(self_attention_mlp_pos.parameters(), lr=0.00001), 
                                         criterion=nn.CrossEntropyLoss(), 
                                         num_epochs=10)
#%%
