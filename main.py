import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

from loader import get_data_set
from training import train_model, evaluate_model
from utils import set_seed, plot_results, analyze_examples, MetricLogger, create_metric_plots

from collections import defaultdict
import numpy as np

def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, num_epochs, log_interval=30, validate=True, scheduler=None, plot=True):
    logger = MetricLogger(log_interval, num_epochs)
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            gradient_log = defaultdict(list)

            # Train
            training_epoch(model, train_dataloader, optimizer, criterion, logger)

            # Evaluate
            if validate:
                validation_epoch(model, test_dataloader, criterion, logger)

                if scheduler:
                    scheduler.step(logger.get_metrics()['test_losses'][epoch][-1])
            # print("Gradient statistics:")
            # for name, grads in gradient_log.items():
            #     mean_grads = [g['mean'] for g in grads]
            #     max_grads = [g['max'] for g in grads]
            #     print(f"  {name}:")
            #     print(
            #         f"    Mean: {np.mean(mean_grads):.6f} (min: {np.min(mean_grads):.6f}, max: {np.max(mean_grads):.6f})")
            #     print(f"    Max: {np.mean(max_grads):.6f} (min: {np.min(max_grads):.6f}, max: {np.max(max_grads):.6f})")

            metrics = logger.get_metrics()
            print(
                f"Train Loss: {metrics['train_losses'][epoch][-1]:.4f}, Train Accuracy: {metrics['train_accuracies'][epoch][-1]:.4f}")
            if validate:
                print(
                    f"Test Loss: {metrics['test_losses'][epoch][-1]:.4f}, Test Accuracy: {metrics['test_accuracies'][epoch][-1]:.4f}")

            logger.next_epoch()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Returning metrics collected so far.")

    metrics = logger.get_metrics()
    if plot:
        create_metric_plots(metrics)
    return metrics


def training_epoch(model, dataloader, optimizer, criterion, logger):
    model.train()
    device = next(model.parameters()).device
    logger.reset()

    print(f"Training {model.name()} on {device}")
    pbar = tqdm(dataloader, desc=f"Epoch {logger.current_epoch + 1}/{logger.num_epochs} [Train]")
    for batch_idx, (labels, embeddings, _) in enumerate(pbar):
        labels, embeddings = labels.to(device), embeddings.to(device)

        loss, logits = model.train_step(embeddings, labels, optimizer, criterion)
        logger.log_gradients(model)
        logger.update(loss, logits, labels)
        logger.log_metrics(batch_idx, len(dataloader), pbar, "train")


def validation_epoch(model, dataloader, criterion, logger):
    model.eval()
    device = next(model.parameters()).device
    logger.reset()

    print(f"Evaluating {model.name()} on {device}")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {logger.current_epoch + 1}/{logger.num_epochs} [Eval]")
        for batch_idx, (labels, embeddings, _) in enumerate(pbar):
            labels, embeddings = labels.to(device), embeddings.to(device)

            loss, logits = model.validation_step(embeddings, labels, criterion)
            logger.update(loss, logits, labels)
            logger.log_metrics(batch_idx, len(dataloader), pbar, "test")

