# Comparative Sentiment Analysis: Recurrent vs. Attention-Based Architectures

This repository implements and evaluates four distinct neural architectures for binary sentiment classification on the IMDB Movie Reviews dataset. *The project explores the trade-offs between sequential modeling (RNN/GRU) and localized contextual reasoning (Self-Attention) using frozen GloVe embeddings.


## 📌 Problem Statement

*The objective is to predict movie viewer sentiment (Positive/Negative) based on text reviews truncated to the first 100 words. *We evaluate how different architectural inductive biases—recurrency, global pooling, and local self-attention—impact the model's ability to capture semantic nuance.


## 🏗 Model Architectures

*We implement and compare four strategies:

1. ***Elman RNN**: A standard recurrent neural network to capture sequential dependencies.

2. ***GRU (Gated Recurrent Unit)**: A gated architecture designed to mitigate the vanishing gradient problem.

3. ***MLP + Global Average Pooling**: Every word is mapped to a sub-prediction scalar via an MLP; the final prediction is the sum of these scores.

4. ***Local Self-Attention + MLP**: Adds a restricted self-attention layer (window size 5) with positional encoding before the MLP to allow for cross-word reasoning.


## 🧪 Experimental Setup

* ***Embeddings**: Pre-trained 100-dimensional GloVe vectors (non-trainable.

* ***Hidden Dimensions**: Evaluated across $\{64, 128\}$ to optimize test error.

* ***Input**: $Batch \times Time \times Features$ (100-word sequences.

### Performance Summary

| Model Architecture | Hidden Dim | Test Accuracy | F1-Score | Key Observation |

| :--- | :--- | :--- | :--- | :--- |

| Elman RNN | 64/128 | % | | [Observation on vanishing gradients] |

| GRU | 64/128 | % | | [Observation on long-term memory] |

| MLP (Global Pool) | - | % | | [Observation on "Bag of Words" behavior] |

| Attention + MLP | - | % | | [Observation on contextualized scores] |


## 📊 Results & Visualization

* ***Training Dynamics**: Accuracy and Loss curves comparing RNN vs. GRU convergence.

* ***Sub-Prediction Analysis**: For Strategy 3 & 4, we visualize the "sentiment contribution" of individual words to understand why the model succeeds or fails.


## 🔍 Error Analysis (TP, TN, FP, FN)

*We analyze four specific test cases to understand model behavior:

* **True Positive**: Clear sentiment, high sub-scores on adjectives.

* **True Negative**: Explicitly negative vocabulary.

* **False Positive**: Cases where sarcasm or negations (e.g., "not good") confuse the MLP.

* ***False Negative**: Reviews where the 100-word truncation loses critical context.

## 📁 Repository Structure

```
├── main/                # Core implementation
│   ├── RNN.py
│   ├── self_attention.py
│   ├── MLP.py
│   ├── GRU.py         # Missing Righ now.
│   ├── loader.py       # Data preprocessing & GloVe mapping
│   └── utils.py        # Helper functions
├── main.py             # Training and evaluation entry point
└── results/            # Detailed PDF analysis and figures
