import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as snb
from tqdm import tqdm 
import os, logging
from typing import List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gather_multi_benchmark_results(benchmark_dir: str, type : str) -> pd.DataFrame:
    all_files = [os.path.join(benchmark_dir, f) for f in os.listdir(benchmark_dir) if f.endswith('.csv')]
    df_list = []
    work = 60000  # total number of images processed during the benchmark
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            logger.warning(f"Error processing file : {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        # Add a column to indicate the type of multithreading
        # calculating metrics
        df_stats = df.groupby("num_threads")[["runtime_ms","accuracy"]].mean()#.to_frame("T_n")
        df_stats.rename(columns={"runtime_ms": "T_n"}, inplace=True)
        # Calculate Speedup S_n = T1 / Tn
        T1 = df_stats.loc[1, "T_n"]

        df_stats["S_n"] = T1 / df_stats["T_n"]
        
        # Calculate Efficiency E_n = S_n / n
        df_stats["E_n"] = df_stats["S_n"] / df_stats.index
        # Compute Performance P_n = work / Tn (images per millisecond)
        df_stats["P_n"] = work / df_stats["T_n"]
        df_stats['multithreading_type'] = type
        return df_stats.reset_index()
    else:
        logger.error("No valid CSV files found.")
        return pd.DataFrame()
    
def gather_serial_benchmark_results(benchmark_file: str, type : str) -> pd.DataFrame:
    try:
        df_serial = pd.read_csv(benchmark_file)

        df_serial = df_serial.mean().to_frame().T
        num_threads = [pd.NA]*len(df_serial)
        df_serial.insert(0,'num_threads',num_threads)
        df_serial['multithreading_type'] = 'serial'
        df_serial.rename(columns={'runtime_ms':'T_n'})
        return df_serial
    except Exception as e:
        logger.warning(f"Error processing file : {e}")
        

def plot_multithreading_benchmark_results(df_serial: pd.DataFrame, df_multithreading: pd.DataFrame, multithreading_type: str):
    xdf_stats = df_multithreading[df_multithreading['multithreading_type'] == multithreading_type.lower()].copy()
    fig, axes = plt.subplots(1, 5, figsize=(18,5), layout="constrained")
    
    x = xdf_stats['num_threads']
    serial = df_serial['runtime_ms'].values[0]
    serial_acc = df_serial['accuracy'].values[0]
    # Speedup
    axes[0].plot(x, xdf_stats["S_n"], marker="o", label="Multithreading Speedup", linewidth=2)
    axes[0].set_xlabel("Number of Threads")
    axes[0].set_ylabel("Speedup (S_n)")
    axes[0].set_title(f"{multithreading_type} Multithreading Speedup")
    axes[0].grid(True)
    axes[0].legend()

    # Efficiency
    axes[1].plot(x, xdf_stats["E_n"], marker="s", label="Multithreading Efficiency", linewidth=2)
    axes[1].set_xlabel("Number of Threads")
    axes[1].set_ylabel("Efficiency (E_n)")
    axes[1].set_title(f"{multithreading_type} Multithreading Efficiency")
    axes[1].grid(True)
    axes[1].legend()

    # Compute Performance
    axes[2].plot(x, xdf_stats["P_n"], marker="^", label="Multithreading Throughput", linewidth=2)
    axes[2].set_xlabel("Number of Threads")
    axes[2].set_ylabel("Compute Performance (images/ms)")
    axes[2].set_title(f"{multithreading_type} Multithreading Throughput")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(x, xdf_stats["T_n"], marker="^", label="Multithreading Runtime (ms)", linewidth=2)
    axes[3].axhline(y=serial, color='r', linestyle='--', label='Serial Runtime')
    axes[3].set_xlabel("Number of Threads")
    axes[3].set_ylabel("Runtime (ms)")
    axes[3].set_title(f"{multithreading_type} Multithreading Runtime in ms")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(x, xdf_stats["accuracy"], marker="^", label="Multithreading Accuracy", linewidth=2)
    axes[4].axhline(y=serial_acc, color='r', linestyle='--', label='Serial Accuracy')
    axes[4].set_xlabel("Number of Threads")
    axes[4].set_ylabel("Accuracy %")
    axes[4].set_title(f"{multithreading_type} Multithreading Accuracy")
    axes[4].grid(True)
    axes[4].legend()

    plt.show()

def gather_multinode_benchmark_results(benchmark_dirs: List, type : str) -> pd.DataFrame:
    df_list = []
    for dir in benchmark_dirs:
        all_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')]
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
            except Exception as e:
                logger.warning(f"Error processing file : {e}")
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        logger.error("No valid CSV files found.")
        return pd.DataFrame()

def plot_multinode_benchmark_results(df: pd.DataFrame, df_serial: pd.DataFrame, df_multithreading: pd.DataFrame):
    # --- Preprocessing ---
    df_multithreading_min_r = (
        df_multithreading.groupby('num_threads')['T_n']
        .min()
        .reset_index()
        .min()
        .reset_index()
        .T
    )
    df_multithreading_min_r.columns = df_multithreading_min_r.iloc[0]
    df_multithreading_min_r = df_multithreading_min_r[1:]
    
    df_multithreading_max_acc = (
        df_multithreading.groupby('num_threads')['accuracy']
        .max()
        .reset_index()
        .max()
        .reset_index()
        .T
    )
    df_multithreading_max_acc.columns = df_multithreading_max_acc.iloc[0]
    df_multithreading_max_acc = df_multithreading_max_acc[1:]
    
    serial_runtime = df_serial['runtime_ms'].values[0] / 1000
    serial_acc = df_serial['accuracy'].values[0]
    df["runtime_s"] = df["runtime_ms"] / 1000  # convert ms → s

    # --- Create 1 row × 2 columns ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Runtime ---
    ax1.plot(df["num_ranks"], df["runtime_s"], 'o-', label='DDP Runtime (s)')
    ax1.axhline(y=serial_runtime, color='r', linestyle='--', label='Serial Runtime (s)')
    ax1.axhline(y=df_multithreading_min_r['T_n'].values[0]/1000, color='blue', linestyle='--', label='Multithreading Min Runtime (s)')
    ax1.set_xlabel("Number of Ranks (Total)")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("DDP Multinode Training Runtime")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Right: Accuracy ---
    ax2.plot(df["num_ranks"], df["accuracy"], 's--', color='orange', label='DDP Accuracy (%)')
    ax2.axhline(y=serial_acc, color='green', linestyle='--', label='Serial Accuracy (%)')
    ax2.axhline(y=df_multithreading_max_acc['accuracy'].values[0], color='purple', linestyle='--', label='Multithreading Max Accuracy (%)')
    ax2.set_xlabel("Number of Ranks (Total)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("DDP Multinode Training Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
def gather_multiprocthread_benchmark_results(dir_paths, label):
    df_list = []
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                df = pd.read_csv(file_path)
                #df['configuration'] = os.path.basename(dir_path)
                df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    #combined_df['benchmark_type'] = label
    return combined_df

def plot_multiprocthread_benchmark_results(df: pd.DataFrame, df_serial: pd.DataFrame, df_multithreading: pd.DataFrame, df_multinode: pd.DataFrame):
    df_multithreading_min_r = (
            df_multithreading.groupby('num_threads')['T_n']
            .min()
            .reset_index()
            .min()
            .reset_index()
            .T
        )
    df_multithreading_min_r.columns = df_multithreading_min_r.iloc[0]
    df_multithreading_min_r = df_multithreading_min_r[1:]

    df_multithreading_max_acc = (
        df_multithreading.groupby('num_threads')['accuracy']
        .max()
        .reset_index()
        .max()
        .reset_index()
        .T
    )
    df_multithreading_max_acc.columns = df_multithreading_max_acc.iloc[0]
    df_multithreading_max_acc = df_multithreading_max_acc[1:]

    df_multinode_min_r = (
           df_multinode.groupby('num_ranks')['runtime_ms']
           .min()
           .reset_index()
           .min()
           .reset_index()
           .T
        )
    df_multinode_min_r.columns = df_multinode_min_r.iloc[0]
    df_multinode_min_r = df_multinode_min_r[1:]

    df_multinode_max_acc = (
        df_multinode.groupby('num_ranks')['accuracy']
        .max()
        .reset_index()
        .max()
        .reset_index()
        .T
    )
    df_multinode_max_acc.columns = df_multinode_max_acc.iloc[0]
    df_multinode_max_acc = df_multinode_max_acc[1:]


    serial_runtime = df_serial['runtime_ms'].values[0] / 1000
    serial_acc = df_serial['accuracy'].values[0]
    df["runtime_s"] = df["runtime_ms"] / 1000  # convert ms → s
    # --- Create 1 row × 2 columns ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # --- Left: Runtime ---
    ax1.plot(df["num_ranks"], df["runtime_s"], 'o-', label='DDP Runtime (s)')
    ax1.axhline(y=serial_runtime, color='r', linestyle='--', label='Serial Runtime (s)')
    ax1.axhline(y=df_multithreading_min_r['T_n'].values[0]/1000, color='blue', linestyle='--', label='Multithreading Min Runtime (s)')
    ax1.axhline(y=df_multinode_min_r['runtime_ms'].values[0]/1000, color='purple', linestyle='--', label='Multinode Min Runtime (s)')
    ax1.set_xlabel("Number of Ranks (Total)")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("DDP Multinode Training Runtime")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    # --- Right: Accuracy ---
    ax2.plot(df["num_ranks"], df["accuracy"], 's--', color='orange', label='DDP Accuracy (%)')
    ax2.axhline(y=serial_acc, color='green', linestyle='--', label='Serial Accuracy (%)')
    ax2.axhline(y=df_multithreading_max_acc['accuracy'].values[0], color='purple', linestyle='--', label='Multithreading Max Accuracy (%)')
    ax2.axhline(y=df_multinode_max_acc['accuracy'].values[0], color='brown', linestyle='--', label='Multinode Max Accuracy (%)')
    ax2.set_xlabel("Number of Ranks (Total)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("DDP Multinode Training Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def gather_curves_results(benchmark_dirs: List, type : str) -> pd.DataFrame:
    df_list = []
    for dir in benchmark_dirs:
        all_files = [os.path.join(dir[0], f) for f in os.listdir(dir[0]) if f.endswith('.csv')]
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df['parameter'] = dir[1]
                df_list.append(df)
            except Exception as e:
                logger.warning(f"Error processing file : {e}")
    if df_list:
        validation_df = pd.concat(df_list, ignore_index=True)
        validation_df["parameter_value"] = np.select(
            [
                validation_df["parameter"] == "batch_size",
                validation_df["parameter"] == "learning_rate",
                validation_df["parameter"] == "num_epochs",
                validation_df["parameter"] == "optimizer",
            ],
            [
                validation_df["batch_size"],
                validation_df["learning_rate"],
                validation_df["num_epochs"],
                validation_df["optimizer"],
            ],
            default=np.nan,
        )
        return validation_df.drop(columns=["batch_size", "learning_rate", "num_epochs", "optimizer"])

    else:
        logger.error("No valid CSV files found.")
        return pd.DataFrame()
    

def _plot_validation_curve(validation_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))  # Increased height for 2 rows
    
    for idx, (parameter, group_df) in enumerate(validation_df.groupby('parameter')):
        # Check if parameter_value is numeric
        if parameter in ['batch_size', 'learning_rate', 'num_epochs']:
            group_df = group_df.sort_values('parameter_value', ascending=True)
        
        # Convert runtime from ms to seconds
        group_df = group_df.copy()
        group_df['runtime_s'] = group_df['runtime_ms'] / 1000
        
        # Top row: Accuracy plot
        group_df.plot(x='parameter_value', y='accuracy', ax=axes[0, idx], title=f'Validation Curve of {parameter}')
        axes[0, idx].set_xlabel('Parameter Value')
        axes[0, idx].set_ylabel('Accuracy')
        
        # Bottom row: Runtime plot (in seconds)
        group_df.plot(x='parameter_value', y='runtime_s', ax=axes[1, idx], color='orange', title=f'Parameter vs Runtime(s): {parameter}')
        axes[1, idx].set_xlabel('Parameter Value')
        axes[1, idx].set_ylabel('Runtime (s)')
        
        # Add grid for better readability
        axes[0, idx].grid(True, alpha=0.3)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()