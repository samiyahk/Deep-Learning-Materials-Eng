import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from dataset import loader
from model import CNN

# =================== USER-ADJUSTABLE OPTIONS =====================
SUBSET_PER_SPLIT = 1000 # None for full set, else integer
EPOCHS            = 230
LR                = 0.1
BATCH_SEED        = 1234
OPTIMIZER_NAME    = 'SGD'
MOMENTUM          = 0.9
WEIGHT_DECAY      = 1e-5
# Note: BATCH_SIZE is defined inside dataset.loader (default 128)

NOW     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RUN_DIR = os.path.join(os.getcwd(), 'training_runs', NOW)
os.makedirs(RUN_DIR, exist_ok=True)

def print_log(msg: str, log_file):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

def adjust_lr(optimizer, epoch: int, base_lr: float):
    # decay by 0.1 every 100 epochs
    lr = base_lr * (0.1 ** (epoch // 100))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def train_once(split_idx: int, subset_size: int | None = None):
    print(f"=== Starting training for split {split_idx} ===")
    split_dir = os.path.join(RUN_DIR, f'split_{split_idx}')
    logs_dir  = os.path.join(split_dir, 'logs')
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # reproducibility
    random.seed(BATCH_SEED)
    torch.manual_seed(BATCH_SEED)

    # model & optimizer
    model = CNN(num_classes=31)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # TensorBoard writer
    writer = SummaryWriter(logs_dir)

    # histories
    train_mae_hist,  test_mae_hist      = [], []
    train_mae_per_hist, test_mae_per_hist = [], []
    train_r2_hist,   test_r2_hist       = [], []

    # data loaders
    test_loader, train_loader = loader(split_idx, subset_size=subset_size)

    # training loop
    for epoch in range(EPOCHS):
        adjust_lr(optimizer, epoch, LR)
        model.train()

        run_mae, run_mae_per = 0.0, 0.0
        all_preds, all_labels = [], []

        for bx, by in train_loader:
            bx, by = bx.cuda(), by.cuda()
            bx = bx.unsqueeze(1) if bx.dim()==3 else bx

            preds = model(bx)
            mae   = torch.mean(torch.abs(preds - by))

            optimizer.zero_grad()
            mae.backward()
            optimizer.step()

            # accumulate metrics
            run_mae += mae.item()
            by_adj = torch.cat([by[:, :10], by[:, 11:]], dim=1)
            pr_adj = torch.cat([preds[:, :10], preds[:, 11:]], dim=1)
            run_mae_per += torch.mean(torch.abs(pr_adj - by_adj) / by_adj).item()

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(by.detach().cpu().numpy())

        # epoch-wise training metrics
        train_mae     = run_mae / len(train_loader)
        train_mae_per = run_mae_per / len(train_loader) * 100
        trp = np.vstack(all_preds)
        trl = np.vstack(all_labels)
        trp_adj = np.concatenate([trp[:, :10], trp[:, 11:]], axis=1)
        trl_adj = np.concatenate([trl[:, :10], trl[:, 11:]], axis=1)
        train_r2  = np.clip(r2_score(trl_adj, trp_adj), 0, 1)

        # evaluation
        model.eval()
        run_tmae, run_tmae_per = 0.0, 0.0
        all_tpreds, all_tlabels = [], []

        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.cuda(), by.cuda()
                bx = bx.unsqueeze(1) if bx.dim()==3 else bx
                preds = model(bx)

                run_tmae += torch.mean(torch.abs(preds - by)).item()
                by_adj   = torch.cat([by[:, :10], by[:, 11:]], dim=1)
                pr_adj   = torch.cat([preds[:, :10], preds[:, 11:]], dim=1)
                run_tmae_per += torch.mean(torch.abs(pr_adj - by_adj) / by_adj).item()

                all_tpreds.append(preds.detach().cpu().numpy())
                all_tlabels.append(by.detach().cpu().numpy())

        # epoch-wise validation metrics
        test_mae     = run_tmae / len(test_loader)
        test_mae_per = run_tmae_per / len(test_loader) * 100
        tep = np.vstack(all_tpreds)
        tel = np.vstack(all_tlabels)
        tep_adj = np.concatenate([tep[:, :10], tep[:, 11:]], axis=1)
        tel_adj = np.concatenate([tel[:, :10], tel[:, 11:]], axis=1)
        test_r2  = np.clip(r2_score(tel_adj, tep_adj), 0, 1)

        # log to console & TensorBoard
        log_str = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train MAE: {train_mae:.4f} | Train MAE%: {train_mae_per:.2f} | Train R2: {train_r2:.4f} | "
            f"Test MAE: {test_mae:.4f}  | Test MAE%: {test_mae_per:.2f} | Test R2: {test_r2:.4f}"
        )
        print_log(log_str, open(os.path.join(split_dir, 'log.txt'), 'a'))
        writer.add_scalar('Train/MAE', train_mae, epoch)
        writer.add_scalar('Train/MAE%', train_mae_per, epoch)
        writer.add_scalar('Train/R2', train_r2, epoch)
        writer.add_scalar('Test/MAE', test_mae, epoch)
        writer.add_scalar('Test/MAE%', test_mae_per, epoch)
        writer.add_scalar('Test/R2', test_r2, epoch)

        # store histories
        train_mae_hist.append(train_mae)
        train_mae_per_hist.append(train_mae_per)
        train_r2_hist.append(train_r2)
        test_mae_hist.append(test_mae)
        test_mae_per_hist.append(test_mae_per)
        test_r2_hist.append(test_r2)

    writer.close()

    epochs_arr = np.arange(1, EPOCHS+1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))

    # MAE vs Epoch
    ax1.plot(epochs_arr, train_mae_hist, label='Train MAE')
    ax1.plot(epochs_arr, test_mae_hist,  label='Test MAE')
    ax1.set_title(f'Split {split_idx} – MAE vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(True)
    ax1.text(
        0.02, 0.98,
        f"Final Train={train_mae:.4f}, Test={test_mae:.4f}",
        transform=ax1.transAxes, va='top',
        bbox=dict(facecolor='yellow', alpha=0.5)
    )

    # MAE% vs Epoch
    ax2.plot(epochs_arr, train_mae_per_hist, label='Train MAE%')
    ax2.plot(epochs_arr, test_mae_per_hist,  label='Test MAE%')
    ax2.set_title(f'Split {split_idx} – MAE% vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (%)')
    ax2.legend()
    ax2.grid(True)
    ax2.text(
        0.02, 0.98,
        f"Final Train={train_mae_per:.2f}%, Test={test_mae_per:.2f}%",
        transform=ax2.transAxes, va='top',
        bbox=dict(facecolor='yellow', alpha=0.5)
    )

    # R² vs Epoch
    ax3.plot(epochs_arr, train_r2_hist, '--', label='Train R2')
    ax3.plot(epochs_arr, test_r2_hist,  '-', label='Test R2')
    ax3.set_title(f'Split {split_idx} – R² vs Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R²')
    ax3.legend()
    ax3.grid(True)
    ax3.text(
        0.98, 0.02,
        f"Final Train={train_r2:.4f}, Test={test_r2:.4f}",
        transform=ax3.transAxes, ha='right',
        bbox=dict(facecolor='yellow', alpha=0.5)
    )

    plt.tight_layout()
    plot_path = os.path.join(split_dir, f'training_metrics_split_{split_idx}.png')
    plt.savefig(plot_path)
    plt.close(fig)

    summary = {
        'Split':        split_idx,
        'Subset Size':  subset_size or 'Full',
        'Epochs':       EPOCHS,
        'Learning Rate':LR,
        'Batch Seed':   BATCH_SEED,
        'Optimizer':    OPTIMIZER_NAME,
        'Momentum':     MOMENTUM,
        'Weight Decay': WEIGHT_DECAY,
        'Train MAE':    round(train_mae, 6),
        'Test MAE':     round(test_mae, 6),
        'Train MAE%':   round(train_mae_per, 2),
        'Test MAE%':    round(test_mae_per, 2),
        'Train R2':     round(train_r2, 4),
        'Test R2':      round(test_r2, 4),
    }
    return plot_path, summary


if __name__ == '__main__':
    all_plots, all_summaries = [], []
    for si in range(10):
        p, s = train_once(si, subset_size=SUBSET_PER_SPLIT)
        all_plots.append(p)
        all_summaries.append(s)

    # build DataFrame of all splits
    df = pd.DataFrame(all_summaries)

    # compute Normalized Performance Index (NPI)
    m_min, m_max = df['Test MAE'].min(), df['Test MAE'].max()
    p_min, p_max = df['Test MAE%'].min(), df['Test MAE%'].max()
    r_min, r_max = df['Test R2'].min(),  df['Test R2'].max()

    df['nMAE']   = (m_max - df['Test MAE'])  / (m_max - m_min)
    df['nMAE%']  = (p_max - df['Test MAE%']) / (p_max - p_min)
    df['nR2']    = (df['Test R2'] - r_min)   / (r_max - r_min)
    df['NPI']    = (df['nMAE'] + df['nMAE%'] + df['nR2']) / 3

    # save summary CSV & Excel
    df.to_csv(os.path.join(RUN_DIR, 'training_summary.csv'), index=False)
    excel_path = os.path.join(RUN_DIR, 'training_summary.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)
        wb = writer.book
        ws = writer.sheets['Summary']
        low_good = {
            'type':'3_color_scale','min_type':'min','mid_type':'percentile','mid_value':50,'max_type':'max',
            'min_color':'#63BE7B','mid_color':'#FFEB84','max_color':'#F8696B'
        }
        high_good = {
            'type':'3_color_scale','min_type':'min','mid_type':'percentile','mid_value':50,'max_type':'max',
            'min_color':'#F8696B','mid_color':'#FFEB84','max_color':'#63BE7B'
        }
        for i, col in enumerate(df.columns):
            if col in ['Train MAE','Test MAE','Train MAE%','Test MAE%']:
                ws.conditional_format(1, i, len(df), i, low_good)
            if col in ['Train R2','Test R2','NPI']:
                ws.conditional_format(1, i, len(df), i, high_good)

    print(f"Saved summary (CSV & Excel) to {RUN_DIR}")
