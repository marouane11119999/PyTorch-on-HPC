import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as snb
from tqdm import tqdm 
import os, logging

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
        

