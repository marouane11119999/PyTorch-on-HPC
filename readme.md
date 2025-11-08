# PyTorch on HPC: Scaling Machine Learning Workloads

## 1. Ziel des Projektes
Ziel des Projektes ist die Untersuchung der Verwendbarkeit von **PyTorch** im Kontext von **High-Performance Computing (HPC)**.  
Dabei soll anhand des Datensatzes **Fashion MNIST** gezeigt werden, wie verschiedene Parallelisierungsstrategien (OpenMP, MPI/Distributed Data Parallel, Job Arrays) sowie fortgeschrittene Methoden (Learning Curves, Validation Curves, Hyperparameter-Suche) auf einem HPC-Cluster eingesetzt werden können.  

Der Fokus liegt nicht auf den mathematischen Details der Algorithmen, sondern darauf, **wie Nutzerinnen und Nutzer durch den Einsatz eines HPC-Clusters profitieren** – z.B. durch verkürzte Trainingszeiten, effizientere Skalierung oder beschleunigte Experimentreihen.

---

## 2. Clusterarchitektur
Alle Knoten im verwendeten Cluster besitzen folgende Ausstattung:
- **24 Rechenkerne** pro Knoten (2 Sockel × 12 Kerne, jeweils eine NUMA-Domäne)
- **192 GB Arbeitsspeicher** pro Knoten
- **Intel Omni-Path Interconnect** für schnelle MPI-Kommunikation zwischen den Knoten

Diese Architektur ermöglicht sowohl **Shared-Memory-Parallelisierung (OpenMP)** als auch **Message-Passing-Parallelisierung (MPI / DDP)** sowie **Task-Level-Parallelisierung (Job Arrays)**.

---

## 3. Methodik

### 3.1 Datensatz
- **Fashion MNIST** (28×28 Graustufenbilder, 10 Klassen)  
- Trainingsmenge: 60.000 Bilder  
- Testmenge: 10.000 Bilder  

### 3.2 Modell
- Einfaches **Convolutional Neural Network (CNN)** in PyTorch.  
- Optimizer: Adam oder SGD.  
- Loss-Funktion: Cross-Entropy.  

### 3.3 Software
- **PyTorch** für Modelltraining und DDP  
- **Slurm** für Job Scheduling  
- **OpenMP** (über PyTorch Backend) für Multi-Core Training  
- **MPI**-ähnliche Kommunikation über PyTorch Distributed Data Parallel (DDP)  

---

## 4. Experimente
### 4.1 Data Loading
- experimenting data loading parallel mechanisms using Dataloaders class
### 4.2 Baseline (Serial Training)
- Training auf **1 CPU-Kern**.  
- Messung von: Laufzeit, Speicherverbrauch, Genauigkeit.  

### 4.3 Multi-Core Training (OpenMP / Shared Memory)
- Kontrolle der Threads über `OMP_NUM_THREADS`.  
- Läufe mit:
  - 1 Kern
  - 12 Kerne (1 Sockel)
  - 24 Kerne (gesamter Knoten)  
- Vergleich der Skalierung.  

### 4.4 Multi-Node Training (MPI / Distributed Data Parallel)
- Training mit **PyTorch DDP** auf:
  - 1 Knoten
  - 2 Knoten
  - 4 Knoten  
- Analyse von Speedup und Effizienz.  
- Diskussion des Kommunikationsaufwands (Omni-Path).  

### 4.5 Hyperparameter-Suche (Job Arrays)
- Grid aus Parametern, z.B.:
  - Learning Rate: {0.001, 0.01, 0.1}  
  - Batch Size: {32, 64, 128}  
  - Optimizer: {SGD, Adam}  
- Parallele Ausführung per Slurm Job Array.  
- Vergleich der Ergebnisse in Tabelle + Plot.  

### 4.6 Learning Curves
- Training auf **unterschiedlichen Datenmengen** (10%, 25%, 50%, 75%, 100%).  
- Darstellung von Genauigkeit vs. Datensatzgröße.  
- Nutzung von Job Arrays für parallele Läufe.  

### 4.7 Validation Curves
- Variation eines Hyperparameters (z.B. Learning Rate).  
- Training für mehrere Werte {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}.  
- Plot von Validierungsgenauigkeit vs. Hyperparameterwert.  

---

## 5. Ergebnisse (zu ergänzen)
- **Plots**:  
  - Laufzeit vs. Anzahl Threads (OpenMP)  
  - Laufzeit vs. Anzahl Knoten (DDP)  
  - Genauigkeit vs. Datensatzgröße (Learning Curve)  
  - Genauigkeit vs. Hyperparameter (Validation Curve)  
- **Tabellen**: Hyperparameter-Suche (beste Konfigurationen, Laufzeiten, Genauigkeiten)  

---

