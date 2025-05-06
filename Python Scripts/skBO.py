import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import optuna

from dataset import loader  # Ensure loader accepts a batch_size argument
from model import CNN        # Ensure CNN __init__ accepts conv‐channel hyperparameters

NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RUN_DIR = os.path.join(os.getcwd(), 'optuna_runs', NOW)
os.makedirs(RUN_DIR, exist_ok=True)

SUBSET_PER_SPLIT = 1000   # None for full set, else integer
EPOCHS = 230
SPLITS = 10
N_TRIALS = 50


def adjust_lr(optimizer, epoch: int, base_lr: float):
    lr = base_lr * (0.1 ** (epoch // 100))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def train_split(split_idx: int, hyperparams: dict):
    # Unpack hyperparameters
    lr           = hyperparams['lr']
    momentum     = hyperparams['momentum']
    weight_decay = hyperparams['weight_decay']
    batch_size   = hyperparams['batch_size']
    conv1_ch     = hyperparams['conv1_channels']
    conv2_ch     = hyperparams['conv2_channels']

    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Initialize model
    model = CNN(num_classes=31,
                conv1_channels=conv1_ch,
                conv2_channels=conv2_ch)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay
    )

    # Data loaders
    test_loader, train_loader = loader(
        split_idx,
        subset_size=SUBSET_PER_SPLIT,
        batch_size=batch_size
    )

    # Training loop
    for epoch in range(EPOCHS):
        adjust_lr(optimizer, epoch, lr)
        model.train()
        for bx, by in train_loader:
            bx, by = bx.cuda(), by.cuda()
            bx = bx.unsqueeze(1) if bx.dim() == 3 else bx

            preds = model(bx)
            loss = torch.mean(torch.abs(preds - by))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.cuda(), by.cuda()
            bx = bx.unsqueeze(1) if bx.dim() == 3 else bx
            preds = model(bx)
            preds_list.append(preds.detach().cpu().numpy())
            labels_list.append(by.detach().cpu().numpy())

    preds_np  = np.vstack(preds_list)
    labels_np = np.vstack(labels_list)

    # Metrics
    mae     = float(np.mean(np.abs(preds_np - labels_np)))
    mae_pct = float(mae / np.mean(labels_np) * 100)
    r2      = float(np.clip(r2_score(labels_np, preds_np), 0, 1))

    return mae, mae_pct, r2

def objective(trial):
    # Suggest hyperparameters
    hyperparams = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'momentum': trial.suggest_uniform('momentum', 0.5, 0.99),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'conv1_channels': trial.suggest_int('conv1_channels', 16, 128),
        'conv2_channels': trial.suggest_int('conv2_channels', 16, 128),
    }

    # Collect metrics per split
    mae_list, mae_pct_list, r2_list = [], [], []
    for si in range(SPLITS):
        mae, mae_pct, r2 = train_split(si, hyperparams)
        mae_list.append(mae)
        mae_pct_list.append(mae_pct)
        r2_list.append(r2)

    # Compute averages
    mae_mean     = np.mean(mae_list)
    mae_pct_mean = np.mean(mae_pct_list)
    r2_mean      = np.mean(r2_list)

    return float(mae_mean), float(mae_pct_mean), float(r2_mean)

if __name__ == '__main__':
    study = optuna.create_study(
        study_name='regression_cv_pareto',
        directions=['minimize', 'minimize', 'maximize']
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # Build DataFrame of all trial summaries
    records = []
    for t in study.trials:
        rec = {'trial': t.number}
        rec.update(t.params)
        rec['avg_MAE'] = t.values[0]
        rec['avg_MAE%'] = t.values[1]
        rec['avg_R2'] = t.values[2]
        records.append(rec)

    df = pd.DataFrame(records)
    excel_path = os.path.join(RUN_DIR, 'all_trials_summary.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Saved consolidated trial summary to {excel_path}")

    # Print Pareto-optimal trials
    print("Pareto-optimal trials:")
    for t in study.best_trials:
        print(f"  Params: {t.params}")
        print(f"  Values [MAE, MAE%, R2]: {t.values}")

