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
    dist.init_process_group(
        "gloo", # the cluster uses MPI backend
        init_method="env://", # use environment variables to setup (will be set by slurm)
        rank=rank, # rank of the current process
        world_size=world_size # total number of processes
        )

def cleanup_ddp():
    dist.destroy_process_group()
    

def run_benchmark_ddp_training(
    rank: int,
    world_size: int,
    args,
    train_data: Dataset,
    test_data: Dataset,
    output_path: Path
) -> None:
    # setup
    setup_ddp(rank, world_size)
    # run each rank on cpu
    device = torch.device("cpu")
    
    # trainers
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True) # ensure each rank gets different data
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0 # keep data loading serial, to isolate compute scaling
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 # keep data loading serial, to isolate compute scaling
    )
    
    # training
    runtimes, accuracies = [], []
    for rep in range(args.num_repeats):
        model = SimpleCNN().to(device)
        model = DDP(model) # wrap model for DDP
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()        
        start_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch) # shuffle data differently at each epoch
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)
        
        # each  rank computes accuracy on its subset of data
        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        
        # gather results from all ranks
        accuracy_tensor = torch.tensor(accuracy).to(device)
        dist.reduce(accuracy_tensor, dst=0, op=dist.ReduceOp.AVG) # avg accuracy to rank 0
        
        if rank == 0: # only rank 0 logs and saves results
            accuracy_avg = accuracy_tensor.item()
            runtimes.append(runtime)
            accuracies.append(accuracy_avg)
            logger.info(f"DDP Training run {rep+1}/{args.num_repeats} on {world_size} ranks: {runtime:.2f} ms, Accuracy: {accuracy:.2f}%")
    
    if rank == 0:
        df = pd.DataFrame({
            "num_ranks": [world_size] * args.num_repeats,
            "runtime_ms": runtimes,
            "accuracy": accuracies
        })
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / f"ddp_benchmark_{world_size}_nodes.csv"
        df.to_csv(final_path, index=False)
        logger.info(f"Results saved to {final_path}")
    
    cleanup_ddp()
    
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

    # get world size from environment (set by SLURM)
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    
    logger.info(f"Starting DDP benchmark with {world_size} ranks (this is rank {rank})")
    
    run_benchmark_ddp_training(
        rank=rank,
        world_size=world_size,
        args=args,
        train_data=train_data,
        test_data=test_data,
        output_path=output_path
    )
    
if __name__ == "__main__":
    main()