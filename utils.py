import torch
import numpy as np
import matplotlib.pyplot as plt
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_metric_plots(metrics, figsize=(10, 15), save_dir=None):
    plt.figure(figsize=figsize)

    def plot_metric(ax, train_data, test_data, title, ylabel):
        max_epoch = 0
        train_x, train_y = [], []
        test_x, test_y = [], []

        for epoch, (train_epoch_metrics, test_epoch_metrics) in enumerate(zip(train_data, test_data)):
            if len(train_epoch_metrics) > 0 or len(test_epoch_metrics) > 0:
                max_epoch = epoch + 1
                train_x.extend(np.linspace(epoch, epoch + 1, len(train_epoch_metrics), endpoint=False))
                train_y.extend(train_epoch_metrics)
                test_x.extend(np.linspace(epoch, epoch + 1, len(test_epoch_metrics), endpoint=False))
                test_y.extend(test_epoch_metrics)

        ax.plot(train_x, train_y, label='Train', alpha=0.7)
        ax.plot(test_x, test_y, label='Test', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set x-ticks to show epoch numbers
        ax.set_xticks(range(max_epoch + 1))
        ax.set_xticklabels(range(max_epoch + 1))

    # Plot loss
    ax1 = plt.subplot(2, 1, 1)
    plot_metric(ax1, metrics['train_losses'], metrics['test_losses'], 'Loss over epochs', 'Loss')

    # Plot accuracy
    ax2 = plt.subplot(2, 1, 2)
    plot_metric(ax2, metrics['train_accuracies'], metrics['test_accuracies'], 'Accuracy over epochs', 'Accuracy')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def plot_results(results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for model_name, model_results in results.items():
        axs[0, 0].plot(model_results['train_losses'], label=model_name)
        axs[0, 1].plot(model_results['test_losses'], label=model_name)
        axs[1, 0].plot(model_results['train_accuracies'], label=model_name)
        axs[1, 1].plot(model_results['test_accuracies'], label=model_name)

    axs[0, 0].set_title('Train Loss')
    axs[0, 1].set_title('Test Loss')
    axs[1, 0].set_title('Train Accuracy')
    axs[1, 1].set_title('Test Accuracy')

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Value')
        ax.legend()

    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()


def analyze_examples(models, dataloader):
    device = next(iter(models.values())).parameters().__next__().device
    examples = {'TP': None, 'TN': None, 'FP': None, 'FN': None}

    for labels, embeddings, reviews in dataloader:
        labels, embeddings = labels.to(device), embeddings.to(device)

        for model_name, model in models.items():
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels, 1)

            for i in range(len(predicted)):
                if predicted[i] == true_labels[i] == 1 and examples['TP'] is None:
                    examples['TP'] = (reviews[i], outputs[i], model_name)
                elif predicted[i] == true_labels[i] == 0 and examples['TN'] is None:
                    examples['TN'] = (reviews[i], outputs[i], model_name)
                elif predicted[i] == 1 and true_labels[i] == 0 and examples['FP'] is None:
                    examples['FP'] = (reviews[i], outputs[i], model_name)
                elif predicted[i] == 0 and true_labels[i] == 1 and examples['FN'] is None:
                    examples['FN'] = (reviews[i], outputs[i], model_name)

        if all(examples.values()):
            break

    for category, (review, output, model_name) in examples.items():
        print(f"\n{category} Example ({model_name}):")
        print(f"Review: {' '.join(review)}")
        print(f"Model output: {output.tolist()}")

from collections import defaultdict
import torch.nn.functional as F

class MetricLogger:
    def __init__(self, log_interval, num_epochs, log_grad=False):
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.metrics = defaultdict(lambda: [[] for _ in range(num_epochs)])
        self.current_epoch = 0
        self.log_grad = log_grad
        self.grad_log = defaultdict(lambda: [[] for _ in range(num_epochs)])
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.correct = 0
        self.total = 0
        self.n_batches = 0

    def update(self, loss, logits, labels):
        self.total_loss += loss
        probs = F.softmax(logits, dim=1)
        predicted = probs.argmax(dim=1)
        true_labels = labels.argmax(dim=1)
        self.total += labels.size(0)
        self.n_batches += 1
        self.correct += (predicted == true_labels).sum().item()

    def should_log(self, batch_idx, dataloader_length):
        log_interval_batches = dataloader_length // self.log_interval
        return (batch_idx+1) % log_interval_batches == 0 or batch_idx == dataloader_length - 1

    def log_metrics(self, batch_idx, dataloader_length, pbar, mode):
        if self.should_log(batch_idx, dataloader_length):
            current_loss = self.total_loss / self.n_batches
            current_accuracy = self.correct / self.total
            self.metrics[f'{mode}_losses'][self.current_epoch].append(current_loss)
            self.metrics[f'{mode}_accuracies'][self.current_epoch].append(current_accuracy)
            pbar.set_description(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{mode}] Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}")

    def get_metrics(self):
        return self.metrics

    def next_epoch(self):
        self.current_epoch += 1


    def log_gradients(self, model):
        if self.log_grad:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.grad_log[name][self.current_epoch].append({
                        'mean': param.grad.abs().mean().item(),
                        'max': param.grad.abs().max().item()
                    })

    def print_gradient_stats(self):
        if self.log_grad:
            print("\nGradient statistics:")
            for name, grads in self.grad_log.items():
                if grads[self.current_epoch]:  # Check if there are gradients for the current epoch
                    mean_grads = [g['mean'] for g in grads[self.current_epoch]]
                    max_grads = [g['max'] for g in grads[self.current_epoch]]
                    print(f"  {name}:")
                    print(
                        f"    Mean: {np.mean(mean_grads):.6f} (min: {np.min(mean_grads):.6f}, max: {np.max(mean_grads):.6f})")
                    print(
                        f"    Max: {np.mean(max_grads):.6f} (min: {np.min(max_grads):.6f}, max: {np.max(max_grads):.6f})")





def plot_metrics(metrics_dict):
    metrics = ['accuracy', 'loss']

    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True)

    for i, metric_name in enumerate(metrics):
        ax = axs[i] if num_metrics > 1 else axs
        if f'train_{metric_name}' in metrics_dict:
            ax.plot(metrics_dict[f'train_{metric_name}'], label='Train')
        if f'test_{metric_name}' in metrics_dict:
            ax.plot(metrics_dict[f'test_{metric_name}'], label='Test')

        ax.set_title(f'{metric_name.title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.savefig('metrics_plot.png')
    plt.close()