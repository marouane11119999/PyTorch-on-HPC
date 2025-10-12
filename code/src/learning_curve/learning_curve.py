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
from torch.utils.data import Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from model import SimpleCNN
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

# ---------------------------
# Distributed setup / cleanup
# ---------------------------
def setup_ddp():
    master_addr = os.environ.get("MASTER_ADDR", "node2-085")  # first node
    master_port = os.environ.get("MASTER_PORT", "12355")
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    logger.info(f"Rank {rank}/{world_size} initializing DDP (MASTER={master_addr}:{master_port})")

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    return rank, world_size

def cleanup_ddp():
    logger.info("Cleaning up DDP process group")
    dist.destroy_process_group()
    
def run_benchmark_ddp_training(rank, world_size, args, train_data, test_data, output_path, num_threads, fraction):

    # set threading
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    device = torch.device("cpu")

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

    logger.info(f"[Rank {rank}] Data loaders initialized (Train: {len(train_loader)}, Test: {len(test_loader)})")

    runtimes, accuracies = [], []

    for rep in range(args.num_repeats):
        model = SimpleCNN().to(device)
        model = DDP(model)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            logger.info(f"[Rank {rank}] Epoch {epoch+1}/{args.num_epochs} starting")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)
        logger.info(f"[Rank {rank}] Repetition {rep+1} finished in {runtime:.2f} ms")

        # compute accuracy
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logger.info(f"[Rank {rank}] Local accuracy: {accuracy:.2f}%")

        # gather average across ranks
        accuracy_tensor = torch.tensor(accuracy).to(device)
        dist.reduce(accuracy_tensor, dst=0, op=dist.ReduceOp.AVG)

        if rank == 0:
            accuracy_avg = accuracy_tensor.item()
            runtimes.append(runtime)
            accuracies.append(accuracy_avg)
            logger.info(f"[Threads={num_threads}] DDP run {rep+1}/{args.num_repeats} "
                        f"on {world_size} ranks, fraction={fraction}: {runtime:.2f} ms, Accuracy: {accuracy_avg:.2f}%")

    # Save results
    if rank == 0:
        df = pd.DataFrame({
            "num_ranks": [world_size] * args.num_repeats,
            "num_threads": [num_threads] * args.num_repeats,
            "train_fraction": [fraction] * args.num_repeats,
            "runtime_ms": runtimes,
            "accuracy": accuracies
        })
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / f"learning_curve_{fraction}_frac_{world_size}_ranks_{num_threads}_threads.csv"
        df.to_csv(final_path, index=False)
        logger.info(f"Results saved to {final_path}")


def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_repeats", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_cmdargs()
    fraction = fractions[args.array_id]

    train_data, test_data = get_dataset(Path("/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_20/projekt/PyTorch-on-HPC/code/jobs/model_training_benchmark/data"))
    num_train_samples = int(len(train_data) * fraction)
    train_data_subset = Subset(train_data, list(range(num_train_samples)))

    rank, world_size = setup_ddp()
    logger.info(f"Starting DDP benchmark with {world_size} ranks (this is rank {rank})")

    slurm_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    logger.info(f"Using {slurm_threads} threads per process (from SLURM_CPUS_PER_TASK)")

    run_benchmark_ddp_training(
        rank=rank,
        world_size=world_size,
        args=args,
        train_data=train_data_subset,
        test_data=test_data,
        output_path=Path(args.output_dir),
        num_threads=slurm_threads,
        fraction=fraction
    )

    cleanup_ddp()
    



if __name__ == "__main__":
    main()