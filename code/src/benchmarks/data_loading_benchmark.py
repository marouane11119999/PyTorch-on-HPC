import logging, time, os, argparse
import numpy as np
import pandas as pd
from data import get_dataset, get_dataloaders
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(
            num_workers: int,
            train_dataset: Dataset,
            test_dataset: Dataset,
            batch_size: int,
            num_repeats: int
        ) -> pd.DataFrame:
    runtimes = []
    for _ in range(num_repeats):
        loader = get_dataloaders(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers
                        )
        t_start = time.perf_counter()
        for images, labels in loader:
            pass # testing only data loading time
        runtimes.append(1000 * (time.perf_counter() - t_start))
    runtimes = runtimes[1:]  # discard first run (cache warmup)
    logger.info(
        f"{num_workers = }: {np.mean(runtimes):.1f} +/- {np.std(runtimes):.1f} ms"
        f" from {len(runtimes)} runs"
    )
    return pd.DataFrame({
        "num_workers": [num_workers] * len(runtimes),
        "runtime_ms": runtimes
    })

def get_cmdargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--workers_list", type=int, nargs="+", default=[1, 2, 4, 8, 12, 24])
    return parser.parse_args()

def main():
    args = get_cmdargs()
    data_path = Path("_benchmark/data_loading/data")
    output_path = Path("_benchmark/data_loading")
    FINAL_PATH = os.path.join(output_path, "data_loading_benchmark.csv")
    output_path.mkdir(parents=True, exist_ok=True)
    all_results = []
    train, test = get_dataset(
        path=data_path
    )  # download data once
    for nw in args.workers_list:
        df_results = run_benchmark(
            num_workers=nw,
            batch_size=args.batch_size,
            train_dataset=train,
            test_dataset=test,
            num_repeats=args.num_repeats
        )
        all_results.append(df_results)
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(FINAL_PATH, index=False)
    logger.info(f"All benchmarks saved to {output_path}")
    
if __name__ == "__main__":
    main()