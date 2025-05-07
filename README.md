# Deep Learning for Materials Design and Engineering

**Author:** Samiyah Karim
**Programme:** MEng Aerospace Engineering
**Repo:** [Deep-Learning-Materials-Eng](https://github.com/samiyahk/Deep-Learning-Materials-Eng)

---

## Overview

This repository reproduces and extends the pipeline from Jiang et al. (2022) to predict dispersion relations of elastic metamaterials using convolutional neural networks (CNNs). By mapping binary unit‑cell images directly to eigenfrequency curves, the model serves as a fast, data‑driven surrogate for classical FEM/Bloch–Floquet solvers.

Key features:

* **Core‑symmetry augmentations:** 90° rotations, flips, ±5 px periodic shifts
* **Reproducibility:** Controlled bit‑wise seeding for data splits and training
* **Hyperparameter tuning:** Manual one‑factor‑at‑a‑time sweeps and Optuna-based Bayesian optimisation
* **Lightweight workflow:** CPU and GPU, 1 000 samples per branch, full run in ≈1 h 12 min

---

## Repository Structure

* `unit_cell_data.mat`
  MATLAB data file containing unit‑cell parameters and properties (tracked with Git LFS).

* `Training_runs/`
  Model checkpoints and large binary artifacts (Git LFS-managed).

* `Optuna_runs/`
  Hyperparameter search logs and results from Optuna studies.

* `Python Scripts/`
  Data loaders, preprocessing scripts, model definitions, training, and evaluation code.
  
---

## Installation & Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/samiyahk/Deep-Learning-Materials-Eng.git
   cd Deep-Learning-Materials-Eng
   ```

2. **Install Git LFS** (if not already)

   * Follow instructions at [https://git-lfs.github.com/](https://git-lfs.github.com/)

   ```bash
   git lfs install
   git lfs pull
   ```

3. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\\Scripts\\activate  # Windows
   ```

4. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Environment

### Hardware Configuration

* Processor: 12th Gen Intel® Core™ i7-12700H (14 cores, 20 threads; 2.30 GHz base clock)
* Primary GPU: NVIDIA® GeForce RTX 3050 Ti Laptop GPU (4 GB GDDR6; CUDA 12.8)
* Integrated GPU: Intel® Iris® Xe Graphics (shared memory up to 7.8 GB)
* System Memory: 16 GB DDR5-4800 MT/s (2×8 GB modules)
* Storage: 1 TB SK hynix PCIe NVMe SSD (≈ 954 GB usable)

### Software Configuration

* Operating System: Windows 11 Professional
* Development Environment: Spyder 6.0.5
* Environment Manager: Conda
* Python Interpreter: 3.12.7
* Key Libraries:

  * NumPy 1.26.4
  * Pandas 2.2.2
  * PyTorch 2.7.0+cu128
  * Scikit-learn 1.5.1
  * TensorBoard 2.19.0
  * Matplotlib 3.9.2
  * Optuna 4.3.0

## Usage

### Data Preparation

```bash
python "Python Scripts/prepare_data.py" \
  --input unit_cell_data.mat \
  --output data/processed
```

### Model Training

```bash
python "Python Scripts/train.py" \
  --config configs/default.yaml \
  --data data/processed
```

### Hyperparameter Optimisation

```bash
python "Python Scripts/optuna_study.py" \
  --n_trials 50 \
  --config configs/optuna.yaml
```

### Results

* Training checkpoints saved in `Training_runs/`.
* Optuna logs and best-trial summaries in `Optuna_runs/`.
* Final plots and metrics in `results/`.

---

## Performance Highlights

* **Manual tuning baseline:** MAE ≈ 0.058, MAE% ≈ 12.3%, R² ≈ 0.87
* **Optuna best trial:** MAE = 0.0450, MAE% = 7.25%, R² = 0.822, NPI = 0.836

