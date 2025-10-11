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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from model import SimpleCNN


# 



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Distributed setup / cleanup
# ---------------------------
def setup_ddp(rank, world_size):
    dist.init_process_group(backend="mpi")

def cleanup_ddp():
    dist.destroy_process_group()
    

def run_benchmark_ddp_training(
    rank: int,
    world_size: int,
    args,
    train_data: Dataset,
    test_data: Dataset,
    output_path: Path,
    num_threads: int
) -> None:
    """
    Run DDP benchmark on multiple nodes using MPI backend.

    Each rank trains on its subset of data using DistributedSampler.
    Only rank 0 gathers results and writes CSV.
    """

    # set threading
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    device = torch.device("cpu")

    # DistributedSampler ensures each rank gets different data
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    runtimes, accuracies = [], []

    for rep in range(args.num_repeats):
        model = SimpleCNN().to(device)
        model = DDP(model)  # wrap for DDP
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)  # shuffle differently per epoch
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)

        # compute accuracy on test set
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # gather average accuracy across all ranks to rank 0
        accuracy_tensor = torch.tensor(accuracy).to(device)
        dist.reduce(accuracy_tensor, dst=0, op=dist.ReduceOp.AVG)

        if rank == 0:
            accuracy_avg = accuracy_tensor.item()
            runtimes.append(runtime)
            accuracies.append(accuracy_avg)
            logger.info(
                f"[Threads={num_threads}] DDP run {rep+1}/{args.num_repeats} "
                f"on {world_size} ranks: {runtime:.2f} ms, Accuracy: {accuracy_avg:.2f}%"
            )

    # rank 0 saves results
    if rank == 0:
        df = pd.DataFrame({
            "num_ranks": [world_size] * args.num_repeats,
            "num_threads": [num_threads] * args.num_repeats,
            "runtime_ms": runtimes,
            "accuracy": accuracies
        })
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / f"ddp_benchmark_{world_size}_ranks_{num_threads}_threads.csv"
        df.to_csv(final_path, index=False)
        logger.info(f"Results saved to {final_path}")

    
def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_repeats", type=int, default=5)
    return parser.parse_args()

def main():
    args = get_cmdargs()
    data_path = Path("/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_20/projekt/PyTorch-on-HPC/code/jobs/model_training_benchmark/data")
    output_path = Path("_benchmark/2nodes_ddp")
    output_path.mkdir(parents=True, exist_ok=True)

    # get data
    train_data, test_data = get_dataset(data_path)

    
    setup_ddp()  # no rank/world_size arguments needed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"Starting DDP benchmark with {world_size} ranks (this is rank {rank})")
    
    for threads in [12, 24]:
        run_benchmark_ddp_training(
            rank=rank,
            world_size=world_size,
            args=args,
            train_data=train_data,
            test_data=test_data,
            output_path=output_path,
            num_threads=threads
        )
    
if __name__ == "__main__":
    main()