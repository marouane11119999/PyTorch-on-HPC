import logging, time, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_loading_benchmark')))

from model import SimpleCNN
from data import get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(lr, batch_size, num_epochs, output_dir):
    data_path = Path("/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_20/projekt/PyTorch-on-HPC/code/jobs/model_training_benchmark/data")
    train_data, test_data = get_dataset(data_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training with lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

    start = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    runtime_ms = 1000 * (time.perf_counter() - start)

    # Evaluate accuracy
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total

    # Save results
    df = pd.DataFrame([{
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "runtime_ms": runtime_ms,
        "accuracy": accuracy
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"results_lr{lr}_bs{batch_size}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")


def get_hyperparams_from_array_id(array_id):
    # Define grid
    learning_rates = [0.01, 0.05, 0.1]
    batch_sizes = [64, 128, 256]
    num_epochs = 5

    grid = [(lr, bs, num_epochs) for lr in learning_rates for bs in batch_sizes]
    return grid[array_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="_benchmark/hyperparam_search")
    args = parser.parse_args()

    lr, batch_size, num_epochs = get_hyperparams_from_array_id(args.array_id)
    output_dir = Path(args.output_dir)
    train_model(lr, batch_size, num_epochs, output_dir)


if __name__ == "__main__":
    main()
