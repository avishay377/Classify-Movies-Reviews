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

#### Based on loading models that already we trained and saved. ####
### Use in jupyter notebook or add prints. ####

import torch
from collections import defaultdict
from torch.utils.data import DataLoader

class IndexedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        for batch_idx, (labels, embeddings, reviews) in enumerate(self.dataloader):
            start_index = batch_idx * self.batch_size
            indices = torch.arange(start_index, start_index + len(labels))
            yield indices, labels, embeddings, reviews

    def __len__(self):
        return len(self.dataset)

    def get_item(self, index):
        return self.dataset[index]

    
    def get_review_by_index(self, index):
        _, _, _, review = self.get_item(index)
        return ' '.join(review)
indexed_test_dataloader = IndexedDataLoader(test_dataloader)
#%%
def get_prediction_indices(model, indexed_dataloader, device):
    model.eval()
    tp_indices, tn_indices, fp_indices, fn_indices = set(), set(), set(), set()
    
    with torch.no_grad():
        for indices, labels, embeddings, _ in indexed_dataloader:
            labels, embeddings = labels.to(device), embeddings.to(device)
            outputs = model(embeddings)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            predicted = outputs.argmax(dim=1)
            true_labels = labels.argmax(dim=1)
            
            for idx, pred, true_label in zip(indices, predicted, true_labels):
                idx = idx.item()
                if pred == 0 and true_label == 0:  # Positive
                    tp_indices.add(idx)
                elif pred == 1 and true_label == 1:  # Negative
                    tn_indices.add(idx)
                elif pred == 0 and true_label == 1:  # False Positive
                    fp_indices.add(idx)
                elif pred == 1 and true_label == 0:  # False Negative
                    fn_indices.add(idx)
    
    return {
        'TP': tp_indices,
        'TN': tn_indices,
        'FP': fp_indices,
        'FN': fn_indices
    }

def analyze_models(models, indexed_dataloader, device):
    results = {}
    for model_name, model in models.items():
        results[model_name] = get_prediction_indices(model, indexed_dataloader, device)
    return results


#%%

def format_output(logits):
    probs = F.softmax(torch.tensor(logits), dim=0)
    return [(f"{logit:.4f}", f"{prob:.4f}") for logit, prob in zip(logits, probs)]

def get_prediction_by_index(model, indexed_dataloader, index):
    # Directly access the underlying dataset
    return predict_raw_review(model, indexed_dataloader.dataset[index])


def predict_raw_review(model, raw_review):
    review, label = raw_review
    embedding = preprocess_review(review).to(device)
    print(embedding.shape)
    
    model.eval()
    with torch.no_grad():
        scores, sub_scores = model(embedding)        
        
        predicted = scores.argmax(dim=1).item()
    
    return {
        'review': review,
        'true_label': label.title(),
        'predicted_label': 'Positive' if predicted == 0 else 'Negative',
        'model_output': scores.squeeze().tolist(),
        'sub_scores': sub_scores.squeeze().tolist() if sub_scores is not None else None
    }

def display_prediction_with_sub_scores_by_index(model, indexed_dataloader, index, device):
    results = get_prediction_by_index(model, indexed_dataloader, index)
    display_prediction_with_sub_scores(results)

def display_prediction_with_sub_scores(results):
    from loader import tokinize
    print(f"Review (index {results.get('index', None)}):")
    print(results['review'])
    print(f"\nTokenized review: {' '.join(tokinize(results['review']))}")
    print(f"\nTrue label: {results['true_label']}")
    print(f"Predicted label: {results['predicted_label']}")

    print("\nOverall output:")
    overall_output = format_output(results['model_output'])
    print(f"Positive: {overall_output[0]}, Negative: {overall_output[1]}")

    if results['sub_scores'] is not None:
        print("\nSub-scores for each word:")
        for word, sub_score in zip(tokinize(results['review']), results['sub_scores']):
            word_output = format_output(sub_score)
            print(f"{word:<15}: Pos: {word_output[0]}, Neg: {word_output[1]}")

def analyze_specific_indices(models, indexed_dataloader, indices_to_check, device):
    for index in indices_to_check:
        print(f"\n{'='*50}")
        print(f"Checking index {index}:")
        for model_name, model in models.items():
            print(f"\nPrediction for {model_name}:")
            display_prediction_with_sub_scores_by_index(model, indexed_dataloader, index, device)




#%%
models = {
    'MLP': mlp,
    'SelfAttentionMLP': self_attention_mlp_pos
}
results = analyze_models(models, indexed_test_dataloader, device)

list(results['MLP']['TP'])[:5], list(results['MLP']['TN'])[:5]

display_prediction_with_sub_scores(get_prediction_by_index(mlp, indexed_test_dataloader, 1))
def compare_review_results(models, review, sentiment):
    for model_name, model in models.items():
        results = predict_raw_review(model, (review, sentiment))
        print(f"\nPrediction for {model_name}:")
        display_prediction_with_sub_scores(results)  

#%%
FP = "Although all the reviews praised the movie with high praises, I thought it was terrible"
compare_review_results(models, FP, 'negative')


TP = "Incredible movie, good actors, great screenplay, not bad at all"


compare_review_results(models, TP, 'positive')

hard_pos = "Although the choice of this subject by most directors will lead to a terrible movie, because of the complicated emotions the characters exhibit. This time it was not the case, a great, stunning and very good movie, I will remember this talented director."
compare_review_results(models, hard_pos, 'positive')


FN = "The movie perfectly depicted the harsh and cruel reality of life in the favelas, I was so immersed I felt like I was there. I couldn't stop crying from how good it was."

FN = "I usually don't like movies that depict the harsh reality of life, but this one was so well made I couldn't stop crying."

FN = "People with no eyes for art will not appreciate this movie, they will say it is trash,but it's a masterpiece"

FN = "The movie is not bad it is actually terrible, if you don't have eyes and can't appreciate brilliance, masterpiece at display, I couldn't stop crying, it was so good"
"We are faced with all the suffering and terrible consequences this place has to offer, however the main character has a dream of leaving, brilliant piece of art, incredibly touching and moving I couldn't stop crying"


FN = 'The brilliant actors: johnny dep, morgan freeman, angelina jolie, the amazing leonardo dicaprio, and brad pit mad the movie even though the screenplay was pretty terrible'
compare_review_results(models, FN, 'positive')



FN = "The movie is not bad it is actually terrible, if you don't have eyes and can't appreciate brilliance, masterpiece at display, I couldn't stop crying, it was so good"
FN = "I usually like movies with jim carrey, but this one just didn't do it for me."
compare_review_results(models, FN, 'negative')


#%%
MLP_FN_ATTEN_TP = list(results['MLP']['FN'].intersection(results['SelfAttentionMLP']['TP']))
MLP_FP_ATTEN_TN = list(results['MLP']['FP'].intersection(results['SelfAttentionMLP']['TN']))


MLP_FN_ATTEN_TP_examples =sorted(MLP_FN_ATTEN_TP)[:5]
MLP_FP_ATTEN_TN_examples = sorted(MLP_FP_ATTEN_TN)[:5]
print(MLP_FN_ATTEN_TP_examples, MLP_FP_ATTEN_TN_examples)

analyze_specific_indices(models, indexed_test_dataloader, MLP_FN_ATTEN_TP_examples, device)
analyze_specific_indices(models, indexed_test_dataloader, MLP_FP_ATTEN_TN_examples, device)


def calculate_confidence_scores(outputs, true_labels):
    probabilities = F.softmax(outputs, dim=1)
    predicted_labels = outputs.argmax(dim=1)
    correct_mask = (predicted_labels == true_labels)
    
    confidence_scores = probabilities.gather(1, true_labels.unsqueeze(1)).squeeze(1)
    confidence_scores = torch.where(correct_mask, confidence_scores, 1 - confidence_scores)
    
    return confidence_scores
#%%

import torch
import numpy as np
from tqdm.auto import tqdm

def find_divergent_examples(models, indexed_dataloader, device, top_k=10):
    mlp_model = models['MLP']
    attention_model = models['SelfAttentionMLP']
    
    mlp_incorrect = []
    attention_correct = []
    
    for indices, labels, embeddings, _ in tqdm(indexed_dataloader):
        indices, labels, embeddings = indices.to(device), labels.to(device), embeddings.to(device)
        true_labels = labels.argmax(dim=1)
        
        with torch.no_grad():
            mlp_outputs = mlp_model(embeddings)
            attention_outputs = attention_model(embeddings)
            
            if isinstance(mlp_outputs, tuple):
                mlp_outputs = mlp_outputs[0]
            if isinstance(attention_outputs, tuple):
                attention_outputs = attention_outputs[0]
            
            mlp_preds = mlp_outputs.argmax(dim=1)
            attention_preds = attention_outputs.argmax(dim=1)
            
            mlp_probs = torch.softmax(mlp_outputs, dim=1)
            attention_probs = torch.softmax(attention_outputs, dim=1)
            
            mlp_scores = mlp_probs[torch.arange(len(true_labels)), true_labels]
            attention_scores = attention_probs[torch.arange(len(true_labels)), true_labels]
            
            mlp_incorrect.extend([(idx.item(), score.item(), label.item()) 
                                  for idx, score, label in zip(indices[mlp_preds != true_labels], 
                                                               mlp_scores[mlp_preds != true_labels],
                                                               true_labels[mlp_preds != true_labels])])
            
            attention_correct.extend([(idx.item(), score.item(), label.item()) 
                                      for idx, score, label in zip(indices[attention_preds == true_labels], 
                                                                   attention_scores[attention_preds == true_labels],
                                                                   true_labels[attention_preds == true_labels])])
    
    # Convert to numpy arrays for faster operations
    mlp_incorrect = np.array(mlp_incorrect)
    attention_correct = np.array(attention_correct)
    
    # Find indices present in both arrays
    common_indices = np.intersect1d(mlp_incorrect[:, 0], attention_correct[:, 0])
    
    # Filter arrays to keep only common indices
    mlp_filtered = mlp_incorrect[np.isin(mlp_incorrect[:, 0], common_indices)]
    attention_filtered = attention_correct[np.isin(attention_correct[:, 0], common_indices)]
    
    # Sort both arrays by index for easier comparison
    mlp_filtered = mlp_filtered[mlp_filtered[:, 0].argsort()]
    attention_filtered = attention_filtered[attention_filtered[:, 0].argsort()]
    
    # Calculate divergence
    divergence = attention_filtered[:, 1] - mlp_filtered[:, 1]
    
    # Split into positive and negative examples
    pos_mask = attention_filtered[:, 2] == 0
    neg_mask = attention_filtered[:, 2] == 1
    
    # Get top k for positive and negative
    top_divergent_pos = attention_filtered[pos_mask, 0][np.argsort(divergence[pos_mask])[-top_k:]].astype(int)
    top_divergent_neg = attention_filtered[neg_mask, 0][np.argsort(divergence[neg_mask])[-top_k:]].astype(int)
    
    return top_divergent_pos.tolist(), top_divergent_neg.tolist()

# Usage
models = {
    'MLP': mlp,
    'SelfAttentionMLP': self_attention_mlp_pos
}

top_divergent_pos, top_divergent_neg = find_divergent_examples(models, indexed_test_dataloader, device, top_k=10)
print(top_divergent_pos, top_divergent_neg)


[top_divergent_neg[0]]


analyze_specific_indices(models, indexed_test_dataloader, top_divergent_neg, device)

top_divergent_pos2

FP = "Although all the reviews praised the movie with high praises, I thought it was terrible"
preprocess_review(FP)

analyze_specific_indices(models, indexed_test_dataloader, top_divergent_pos2, device)

analyze_specific_indices(models, indexed_test_dataloader, examples_to_check, device)



