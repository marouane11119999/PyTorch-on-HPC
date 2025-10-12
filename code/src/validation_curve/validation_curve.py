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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Parameter sets (each of length 4)
# ---------------------------
learning_rates  = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes     = [32, 64, 128, 256]
optimizers      = ['SGD', 'Adam', 'RMSprop', 'AdamW']
num_epochs_list = [1, 3, 5, 7]

# ---------------------------
# DDP setup / cleanup
# ---------------------------
def setup_ddp():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
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

# ---------------------------
# Training logic
# ---------------------------
def run_benchmark_ddp_training(rank, world_size, args, train_data, test_data, output_path, num_threads):
    """
    Run a single benchmark varying only the parameter specified by args.arg_type.
    """

    # Thread setup
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    device = torch.device("cpu")

    # Dataloaders
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
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
    logger.info(f"[Rank {rank}] testing {args.arg_type} Using batch size {args.batch_size}, optimizer {args.optimizer_name}, "
                f"lr {args.learning_rate}, epochs {args.num_epochs}")

    runtimes, accuracies = [], []

    for rep in range(args.num_repeats):
        model = SimpleCNN().to(device)
        model = DDP(model)
        # Select optimizer
        if args.optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
        elif args.optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer_name}")

        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        runtime = 1000 * (time.perf_counter() - start_time)
        logger.info(f"[Rank {rank}] Repetition {rep+1} finished in {runtime:.2f} ms")

        # Validation accuracy
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_tensor = torch.tensor(accuracy).to(device)
        dist.reduce(accuracy_tensor, dst=0, op=dist.ReduceOp.AVG)

        if rank == 0:
            acc_avg = accuracy_tensor.item()
            runtimes.append(runtime)
            accuracies.append(acc_avg)
            logger.info(f"[Rank 0] {args.arg_type}={args.param_value} -> Acc={acc_avg:.2f}%, Time={runtime:.2f}ms")

    # Save results
    if rank == 0:
        df = pd.DataFrame({
            "num_ranks": [world_size] * len(runtimes),
            "num_threads": [num_threads] * len(runtimes),
            args.arg_type: [args.param_value] * len(runtimes),
            "runtime_ms": runtimes,
            "accuracy": accuracies
        })
        output_path.mkdir(parents=True, exist_ok=True)
        val_str = str(args.param_value).replace('.', 'p')
        final_path = output_path / f"validation_curve_{args.arg_type}_{val_str}_{world_size}r_{num_threads}t.csv"
        df.to_csv(final_path, index=False)
        logger.info(f"Results saved to {final_path}")

# ---------------------------
# Main
# ---------------------------
def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--arg_type", type=str, required=True,
                        choices=["learning_rate", "batch_size", "optimizer", "num_epochs"])
    parser.add_argument("--num_repeats", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_cmdargs()

    # Select parameter value
    if args.arg_type == "learning_rate":
        args.param_value = learning_rates[args.array_id]
    elif args.arg_type == "batch_size":
        args.param_value = batch_sizes[args.array_id]
    elif args.arg_type == "optimizer":
        args.param_value = optimizers[args.array_id]
    elif args.arg_type == "num_epochs":
        args.param_value = num_epochs_list[args.array_id]
    else:
        raise ValueError(f"Invalid arg_type {args.arg_type}")

    # Default values
    args.learning_rate = 1e-3
    args.batch_size = 128
    args.optimizer_name = "SGD"
    args.num_epochs = 3

    # Override the tested parameter
    if args.arg_type == "learning_rate":
        args.learning_rate = args.param_value
    elif args.arg_type == "batch_size":
        args.batch_size = args.param_value
    elif args.arg_type == "optimizer":
        args.optimizer_name = args.param_value
    elif args.arg_type == "num_epochs":
        args.num_epochs = args.param_value

    # Setup data and DDP
    data_path = Path("/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_20/projekt/PyTorch-on-HPC/code/jobs/model_training_benchmark/data")
    train_data, test_data = get_dataset(data_path)
    rank, world_size = setup_ddp()

    slurm_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    output_path = Path(args.output_dir)

    run_benchmark_ddp_training(rank, world_size, args, train_data, test_data, output_path, slurm_threads)

    cleanup_ddp()


if __name__ == "__main__":
    main()