import logging, time, os, argparse
import numpy as np
import pandas as pd
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


def run_benchmark_serial_training(
    train_data: Dataset,
    test_data: Dataset,
    batch_size: int,
    num_epochs: int,
    num_repeats: int
) -> pd.DataFrame:
    device = torch.device("cpu")
    torch.set_num_threads(1) # force single-threaded execution
    
    runtimes = []
    accuracies = []
    train_loader, test_loader = get_dataloaders(
        train_dataset=train_data,
        test_dataset=test_data,
        batch_size=batch_size,  
        num_workers=0  # single-threaded data loading
    )
    for rep in range(num_repeats):
        # model setup
        model = SimpleCNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.perf_counter()
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)  # in ms
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        
        runtimes.append(runtime)
        accuracies.append(accuracy)
        logger.info(
            f"Run {rep+1}/{num_repeats}: {runtime:.1f} ms, Accuracy: {accuracy:.2f}%"
        )
    return pd.DataFrame({
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
    data_path = Path("_benchmark/model_training/data")
    output_path = Path("_benchmark/model_training")
    FINAL_PATH = os.path.join(output_path, "baseline_model_serial_benchmark.csv")
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data, test_data = get_dataset(path=data_path)
    model = SimpleCNN()
    
    df_results = run_benchmark_serial_training(
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_repeats=args.num_repeats
    )
    df_results.to_csv(FINAL_PATH, index=False)
    logger.info(f"Results saved to {FINAL_PATH}")

if __name__ == "__main__":
    main()