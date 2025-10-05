import logging, time, os, argparse
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_loading_benchmark')))

from data import get_dataset, get_dataloaders
from pathlib import Path
from typing import List
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from model import SimpleCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark_multicore_training(
    train_data: Dataset,
    test_data: Dataset,
    batch_size: int,
    num_epochs: int,
    num_repeats: int,
    num_threads: int
) -> pd.DataFrame:
    device = torch.device("cpu")

    # Set number of threads for OpenMP and PyTorch
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    runtimes = []
    accuracies = []

    train_loader, test_loader = get_dataloaders(
        train_dataset=train_data,
        test_dataset=test_data,
        batch_size=batch_size,
        num_workers=0  # keep data loading serial, to isolate compute scaling
    )

    for rep in range(num_repeats):
        model = SimpleCNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)
        
        # accuracy
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        runtimes.append(runtime)
        accuracies.append(accuracy)
        logger.info(f"[{num_threads} threads] Run {rep+1}/{num_repeats}: {runtime:.1f} ms, Accuracy: {accuracy:.2f}%")

    return pd.DataFrame({
        "num_threads": [num_threads] * num_repeats,
        "runtime_ms": runtimes,
        "accuracy": accuracies
    })

def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_repeats", type=int, default=5)
    return parser.parse_args()

def main():
    args = get_cmdargs()
    data_path = Path("/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_20/projekt/PyTorch-on-HPC/code/jobs/model_training_benchmark")
    output_path = Path("_benchmark/model_training")
    output_path.mkdir(parents=True, exist_ok=True)

    train_data, test_data = get_dataset(path=data_path, download=True)

    all_results = []

    for n_threads in [1, 12, 24]:
        logger.info(f"Running benchmark with {n_threads} threads ...")
        df = run_benchmark_multicore_training(
            train_data=train_data,
            test_data=test_data,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_repeats=args.num_repeats,
            num_threads=n_threads
        )
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)

    # Compute averages for convenience
    summary = (
        df_all.groupby("num_threads")[["runtime_ms", "accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    final_path = output_path / "multicore_model_benchmark.csv"
    df_all.to_csv(final_path, index=False)
    logger.info(f"Results saved to {final_path}")
