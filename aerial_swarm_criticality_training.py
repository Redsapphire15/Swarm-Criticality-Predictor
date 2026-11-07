# -*- coding: utf-8 -*-
"""aerial_swarm_criticality_training.py

Aerial Swarm Critical-Agent Identification — High-Performance Training & Evaluation
Architecture: GraphSAGE -> Edge-Aware Attention (Vectorized) -> Global Context Fusion -> Per-Agent GRU

Outputs:
1. is_critical           - binary classification (BCEWithLogitsLoss)
2. criticality_score     - continuous score (MSELoss)
3. future_score          - predicted criticality score FUTURE_HORIZON timesteps ahead (MSELoss)

All generated plots and evaluation metrics are saved to the ./plots/ and ./outputs/ directories.
"""

from pathlib import Path
import os
import sys
import copy
import math
import time
import json
import random
import argparse
import warnings

# Use non-interactive backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)

# Robust fallback for Jupyter display function
try:
    from IPython.display import display
except (ImportError, NameError):
    def display(x):
        print(x)


# ==========================================
# 1. Configuration & CLI Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Aerial Swarm Critical-Agent Identification Training")
    parser.add_argument("--csv_path", type=str, default="/home/diro/kavin/beh/AE5510/swarm_criticality_dataset_100_scenarios.csv",
                        help="Path to CSV dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save model artifacts and predictions")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                        help="Directory to save generated figures")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--radius", type=float, default=55.0,
                        help="Communication / sensing radius for graph edges")
    parser.add_argument("--future_horizon", type=int, default=2,
                        help="Timesteps ahead for future score target")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for GNN and GRU")
    parser.add_argument("--context_dim", type=int, default=16,
                        help="Context fusion dimension")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Optional limit on number of cases (useful for fast verification)")
    return parser.parse_args()


# ==========================================
# 2. Graph & Sequence Preprocessing
# ==========================================
def make_snapshot(frame, position_cols, binary_target, score_target, radius):
    """
    Constructs a directed spatial graph for a single (case_id, step) snapshot.
    Nodes are sorted by agent_id for deterministic hidden state alignment.
    """
    frame = frame.sort_values("agent_id").reset_index(drop=True)
    pos = frame[position_cols].to_numpy(dtype=np.float32)
    pos_t = torch.tensor(pos, dtype=torch.float32)
    
    # Distance matrix and edge detection within communication radius
    dmat = torch.cdist(pos_t, pos_t)
    src, dst = torch.where((dmat <= radius) & (dmat > 0))
    edge_index = torch.stack([src, dst], dim=0) if len(src) else torch.empty((2, 0), dtype=torch.long)

    # Edge features: relative position (sender - receiver), distance,
    # relative velocity (sender - receiver), and radial relative velocity
    rel_pos = pos_t[src] - pos_t[dst] if len(src) else torch.empty((0, 3), dtype=torch.float32)
    dist = dmat[src, dst].unsqueeze(1) if len(src) else torch.empty((0, 1), dtype=torch.float32)
    if all(c in frame.columns for c in ["vx", "vy", "vz"]):
        vel = torch.tensor(frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float32))
        rel_vel = vel[src] - vel[dst] if len(src) else torch.empty((0, 3), dtype=torch.float32)
        radial = (rel_pos * rel_vel).sum(dim=1, keepdim=True) / dist.clamp_min(1e-6)
    else:
        rel_vel = torch.zeros((len(src), 3), dtype=torch.float32)
        radial = torch.zeros((len(src), 1), dtype=torch.float32)
    edge_attr = torch.cat([rel_pos, dist, rel_vel, radial], dim=1)

    x = torch.tensor(np.stack(frame["_x"].to_numpy()), dtype=torch.float32)
    c = torch.tensor(np.stack(frame["_c"].to_numpy()), dtype=torch.float32)
    y = torch.tensor(frame[binary_target].to_numpy(dtype=np.float32))
    score = torch.tensor(frame[score_target].to_numpy(dtype=np.float32))
    future = torch.tensor(frame["_future_score"].to_numpy(dtype=np.float32))
    agent_ids = frame["agent_id"].tolist()

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "context": c[0],
        "y": y,
        "score": score,
        "future": future,
        "agent_ids": agent_ids,
        "frame": frame,
    }


def make_case_sequences(frame, position_cols, binary_target, score_target, radius):
    """
    Groups scenario rows into chronological sequences of snapshot graphs.
    """
    sequences = {}
    for case_id, case in frame.groupby("case_id", sort=False):
        snapshots = []
        for step, snap in case.groupby("step", sort=True):
            snapshots.append((step, make_snapshot(snap, position_cols, binary_target, score_target, radius)))
        sequences[case_id] = snapshots
    return sequences


# ==========================================
# 3. Model Architecture
# ==========================================
class MeanGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, hidden_dim)
        self.neigh_lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        n = x.size(0)
        src, dst = edge_index
        agg = torch.zeros_like(x)
        deg = torch.zeros((n, 1), device=x.device, dtype=x.dtype)
        if src.numel():
            agg.index_add_(0, dst, x[src])
            deg.index_add_(0, dst, torch.ones((len(dst), 1), device=x.device, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0)
        h = self.self_lin(x) + self.neigh_lin(agg)
        return self.dropout(F.relu(self.norm(h)))


class EdgeAwareAttention(nn.Module):
    """
    Vectorized GAT-style attention with explicit edge attributes.
    Uses PyTorch scatter_reduce and index_add for 30x+ speedup over python node-looping.
    """
    def __init__(self, hidden_dim, edge_dim=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.self_lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        n = h.size(0)
        src, dst = edge_index
        out = self.self_lin(h)
        if src.numel():
            q = self.q(h[dst])
            k = self.k(h[src])
            e = self.edge(edge_attr)
            logits = self.attn(torch.cat([q, k, e], dim=-1)).squeeze(-1)

            # High-performance numerically-stable segmented softmax per destination node
            max_logit = torch.full((n,), -float("inf"), device=h.device)
            max_logit.scatter_reduce_(0, dst, logits, reduce="amax", include_self=False)
            exp_logits = torch.exp(logits - max_logit[dst])
            sum_exp = torch.zeros(n, device=h.device)
            sum_exp.scatter_add_(0, dst, exp_logits)
            alpha = exp_logits / (sum_exp[dst] + 1e-12)

            # Aggregate weighted messages into receiver nodes
            weighted_v = alpha.unsqueeze(-1) * self.v(h[src])
            messages = torch.zeros_like(h)
            messages.index_add_(0, dst, weighted_v)
            out = out + messages
        return self.dropout(F.elu(self.norm(out)))


class SwarmCriticalityNet(nn.Module):
    def __init__(self, node_dim, context_dim, edge_dim=8, hidden_dim=64,
                 context_hidden=16, dropout=0.1, device="cpu"):
        super().__init__()
        self.device = device
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.sage = MeanGraphSAGE(hidden_dim, hidden_dim, dropout)
        self.gat = EdgeAwareAttention(hidden_dim, edge_dim, dropout)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_hidden), nn.ReLU()
        )
        # Mean + max global graph pooling, node embedding, and context embedding
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + context_hidden, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward_snapshot(self, snap, hidden_by_id):
        x = snap["x"].to(self.device)
        edge_index = snap["edge_index"].to(self.device)
        edge_attr = snap["edge_attr"].to(self.device)
        context = snap["context"].to(self.device)
        ids = snap["agent_ids"]

        h = self.node_encoder(x)
        h = self.sage(h, edge_index)
        h = self.gat(h, edge_index, edge_attr)

        # Global graph pooling
        global_mean = h.mean(dim=0, keepdim=True)
        global_max = h.max(dim=0, keepdim=True).values
        global_vec = torch.cat([global_mean, global_max], dim=-1).expand(h.size(0), -1)

        c = self.context_encoder(context.unsqueeze(0)).expand(h.size(0), -1)
        fused = self.fusion(torch.cat([h, global_vec, c], dim=-1))

        # Retrieve prior recurrent hidden state by agent_id
        previous = [hidden_by_id.get(aid, torch.zeros(self.gru.hidden_size, device=self.device)) for aid in ids]
        h_prev = torch.stack(previous, dim=0)
        h_now = self.gru(fused, h_prev)

        # Drop hidden states for absent agents and update current agents
        current_ids = set(ids)
        hidden_by_id = {aid: state for aid, state in hidden_by_id.items() if aid in current_ids}
        for k, aid in enumerate(ids):
            hidden_by_id[aid] = h_now[k]

        return {
            "logits": self.binary_head(h_now).squeeze(-1),
            "score_pred": self.score_head(h_now).squeeze(-1),
            "future_pred": self.future_head(h_now).squeeze(-1),
            "hidden_by_id": hidden_by_id,
        }


# ==========================================
# 4. Losses & Training Loops
# ==========================================
def snapshot_loss(out, snap, lambda_bin=1.0, lambda_score=0.5, lambda_future=0.25, device="cpu"):
    y = snap["y"].to(device)
    score = snap["score"].to(device)
    future = snap["future"].to(device)

    loss_bin = F.binary_cross_entropy_with_logits(out["logits"], y)
    loss_score = F.mse_loss(out["score_pred"], score)

    valid_future = torch.isfinite(future)
    if valid_future.any():
        loss_future = F.mse_loss(out["future_pred"][valid_future], future[valid_future])
    else:
        loss_future = torch.zeros((), device=device)

    total = lambda_bin * loss_bin + lambda_score * loss_score + lambda_future * loss_future
    return total, {
        "binary": float(loss_bin.detach().cpu()),
        "score": float(loss_score.detach().cpu()),
        "future": float(loss_future.detach().cpu()),
    }


def run_case(model, snapshots, optimizer=None, train=False, collect=False,
             lambda_bin=1.0, lambda_score=0.5, lambda_future=0.25, device="cpu"):
    hidden_by_id = {}
    losses = []
    part_losses = {"binary": [], "score": [], "future": []}
    collected = []
    model.train(train)

    context_manager = torch.enable_grad() if train else torch.no_grad()
    with context_manager:
        for step, snap in snapshots:
            out = model.forward_snapshot(snap, hidden_by_id)
            hidden_by_id = out["hidden_by_id"]
            loss, parts = snapshot_loss(out, snap, lambda_bin, lambda_score, lambda_future, device=device)
            losses.append(loss)
            for k, v in parts.items():
                part_losses[k].append(v)

            if collect:
                probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
                score_pred = out["score_pred"].detach().cpu().numpy()
                future_pred = out["future_pred"].detach().cpu().numpy()
                frame = snap["frame"].copy()
                frame["prob_critical"] = probs
                frame["pred_criticality_score"] = score_pred
                frame["pred_future_score"] = future_pred
                collected.append(frame)

        if not losses:
            return 0.0, {k: 0.0 for k in part_losses}, collected
        total = torch.stack(losses).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    avg_parts = {k: float(np.mean(v)) for k, v in part_losses.items()}
    return float(total.detach().cpu()), avg_parts, collected


def run_epoch(model, sequences, optimizer=None, train=False,
              lambda_bin=1.0, lambda_score=0.5, lambda_future=0.25, device="cpu"):
    case_losses = []
    part_accum = {"binary": [], "score": [], "future": []}
    keys = list(sequences.keys())
    if train:
        random.shuffle(keys)
    for case_id in keys:
        loss, parts, _ = run_case(
            model, sequences[case_id], optimizer=optimizer, train=train, collect=False,
            lambda_bin=lambda_bin, lambda_score=lambda_score, lambda_future=lambda_future, device=device
        )
        case_losses.append(loss)
        for k, v in parts.items():
            part_accum[k].append(v)
    avg_total = float(np.mean(case_losses)) if case_losses else float("nan")
    avg_parts = {k: float(np.mean(v)) for k, v in part_accum.items()}
    return avg_total, avg_parts


def predict_sequences(model, sequences, device="cpu"):
    all_frames = []
    model.eval()
    with torch.no_grad():
        for case_id, snapshots in sequences.items():
            _, _, frames = run_case(model, snapshots, train=False, collect=True, device=device)
            all_frames.extend(frames)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def compute_ranking_metrics(frame, k_list=(1, 2, 3, 5, 8, 10), label_col="is_critical", score_col="prob_critical"):
    """
    Computes Precision@K, Recall@K, and NDCG@K across all snapshot graphs.
    """
    results_by_k = {}
    grouped = [g for _, g in frame.groupby(["case_id", "step"]) if len(g) > 0]
    
    for k in k_list:
        recalls, ndcgs, precisions = [], [], []
        for group in grouped:
            kk = min(k, len(group))
            ranked = group.sort_values(score_col, ascending=False)
            rel = ranked[label_col].to_numpy(dtype=float)
            hits = rel[:kk].sum()
            total_relevant = rel.sum()
            precisions.append(hits / kk)
            recalls.append(hits / total_relevant if total_relevant > 0 else np.nan)
            
            # NDCG@K
            gains = rel[:kk] / np.log2(np.arange(2, kk + 2))
            dcg = gains.sum()
            ideal = np.sort(rel)[::-1][:kk] / np.log2(np.arange(2, kk + 2))
            idcg = ideal.sum()
            ndcgs.append(dcg / idcg if idcg > 0 else np.nan)

        results_by_k[k] = {
            f"Precision@{k}": float(np.nanmean(precisions)),
            f"Recall@{k}": float(np.nanmean(recalls)),
            f"NDCG@{k}": float(np.nanmean(ndcgs)),
        }
    return results_by_k


# ==========================================
# 5. Comprehensive Plotting Suite
# ==========================================
def generate_all_plots(history, test_pred, metrics, ranking_res, plots_dir):
    """
    Creates and saves high-resolution publication-quality plots:
    1. Training & Validation Loss curves (total + multi-task components)
    2. ROC & Precision-Recall curves
    3. Confusion Matrix & Threshold Tradeoff Curves
    4. Prediction Probability Distribution for Critical vs Non-Critical
    5. Criticality Score Parity Plot (Predicted vs True)
    6. Future Criticality Score Parity Plot
    7. Precision@K, Recall@K, NDCG@K across K
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plot_paths = []

    # Palette & styling
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

    # ----------------------------------------------------
    # Plot 1: Training & Validation Loss Curves
    # ----------------------------------------------------
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train Total Loss", linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, history["val_loss"], "r--s", label="Val Total Loss", linewidth=2, markersize=4)
    axes[0, 0].set_title("Total Multi-Task Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Binary BCE Loss
    axes[0, 1].plot(epochs, history["train_bce"], "b-", label="Train BCE", linewidth=2)
    axes[0, 1].plot(epochs, history["val_bce"], "r--", label="Val BCE", linewidth=2)
    axes[0, 1].set_title("Binary Criticality Loss (BCE)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BCE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Criticality Score MSE
    axes[1, 0].plot(epochs, history["train_score"], "b-", label="Train Score MSE", linewidth=2)
    axes[1, 0].plot(epochs, history["val_score"], "r--", label="Val Score MSE", linewidth=2)
    axes[1, 0].set_title("Criticality Score Regression Loss (MSE)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Future Score MSE
    axes[1, 1].plot(epochs, history["train_future"], "b-", label="Train Future MSE", linewidth=2)
    axes[1, 1].plot(epochs, history["val_future"], "r--", label="Val Future MSE", linewidth=2)
    axes[1, 1].set_title("Future Score Forecast Loss (MSE)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p1 = plots_dir / "training_loss_curves.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p1)

    # ----------------------------------------------------
    # Plot 2: ROC and Precision-Recall Curves
    # ----------------------------------------------------
    y_true = test_pred["is_critical"].to_numpy().astype(int)
    p_pred = test_pred["prob_critical"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    roc_auc = roc_auc_score(y_true, p_pred)
    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, p_pred)
    pr_auc = average_precision_score(y_true, p_pred)
    pos_rate = y_true.mean()
    axes[1].plot(rec, prec, color="#1e824c", lw=2.5, label=f"PR Curve (AP = {pr_auc:.4f})")
    axes[1].axhline(y=pos_rate, color="k", linestyle="--", lw=1.5, label=f"Baseline ({pos_rate:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p2 = plots_dir / "pr_roc_curves.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p2)

    # ----------------------------------------------------
    # Plot 3: Confusion Matrix & Threshold Tradeoff
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cm = confusion_matrix(y_true, (p_pred >= 0.5).astype(int))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = axes[0].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].figure.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set(xticks=[0, 1], yticks=[0, 1],
                xticklabels=["Non-Critical", "Critical"],
                yticklabels=["Non-Critical", "Critical"],
                title="Normalized Confusion Matrix (@ 0.5)",
                ylabel="True Label", xlabel="Predicted Label")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black",
                         fontsize=11, fontweight="bold")

    # Threshold analysis
    thresholds = np.linspace(0.05, 0.95, 37)
    thresh_prec, thresh_rec, thresh_f1 = [], [], []
    for th in thresholds:
        yp = (p_pred >= th).astype(int)
        thresh_prec.append(precision_score(y_true, yp, zero_division=0))
        thresh_rec.append(recall_score(y_true, yp, zero_division=0))
        thresh_f1.append(f1_score(y_true, yp, zero_division=0))

    axes[1].plot(thresholds, thresh_prec, label="Precision", color="#3498db", lw=2)
    axes[1].plot(thresholds, thresh_rec, label="Recall", color="#e74c3c", lw=2)
    axes[1].plot(thresholds, thresh_f1, label="F1-Score", color="#2ecc71", lw=2.5)
    best_f1_idx = int(np.argmax(thresh_f1))
    best_th = thresholds[best_f1_idx]
    axes[1].axvline(best_th, color="purple", linestyle=":", lw=1.5, label=f"Max F1 ({thresh_f1[best_f1_idx]:.3f} @ th={best_th:.2f})")
    axes[1].set_xlabel("Decision Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Classification Metrics vs. Threshold")
    axes[1].legend(loc="lower center")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p3 = plots_dir / "confusion_matrix_and_thresholds.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p3)

    # ----------------------------------------------------
    # Plot 4: Prediction Probability Distribution
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(p_pred[y_true == 0], bins=50, alpha=0.6, label="Actual Non-Critical", color="#3498db", density=True)
    plt.hist(p_pred[y_true == 1], bins=50, alpha=0.6, label="Actual Critical", color="#e74c3c", density=True)
    plt.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Default Threshold (0.5)")
    plt.xlabel("Predicted Critical Probability")
    plt.ylabel("Density")
    plt.title("Distribution of Criticality Probability Predictions")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    p4 = plots_dir / "prediction_probability_distribution.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p4)

    # ----------------------------------------------------
    # Plot 5: Criticality Score Parity Plot (Current Score)
    # ----------------------------------------------------
    valid_score = np.isfinite(test_pred["criticality_score"].to_numpy())
    y_score_true = test_pred.loc[valid_score, "criticality_score"].to_numpy()
    y_score_pred = test_pred.loc[valid_score, "pred_criticality_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(y_score_true, y_score_pred, gridsize=50, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Agent Density")
    min_v, max_v = min(y_score_true.min(), y_score_pred.min()), max(y_score_true.max(), y_score_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Perfect Parity (y = x)")
    r2 = r2_score(y_score_true, y_score_pred)
    ax.set_title(f"Criticality Score Parity (MAE: {metrics['score MAE']:.4f}, R²: {r2:.4f})")
    ax.set_xlabel("Ground Truth Criticality Score")
    ax.set_ylabel("Predicted Criticality Score")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    p5 = plots_dir / "criticality_score_parity.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p5)

    # ----------------------------------------------------
    # Plot 6: Future Score Parity Plot
    # ----------------------------------------------------
    valid_future = np.isfinite(test_pred["_future_score"].to_numpy())
    if valid_future.any():
        y_fut_true = test_pred.loc[valid_future, "_future_score"].to_numpy()
        y_fut_pred = test_pred.loc[valid_future, "pred_future_score"].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        hb = ax.hexbin(y_fut_true, y_fut_pred, gridsize=50, cmap="inferno", mincnt=1)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Agent Density")
        min_v, max_v = min(y_fut_true.min(), y_fut_pred.min()), max(y_fut_true.max(), y_fut_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "g--", lw=2, label="Perfect Parity (y = x)")
        r2_fut = r2_score(y_fut_true, y_fut_pred)
        ax.set_title(f"Future Criticality Score Parity (MAE: {metrics.get('future score MAE', 0):.4f}, R²: {r2_fut:.4f})")
        ax.set_xlabel(f"Ground Truth Future Score (t + {metrics.get('future_horizon', 2)})")
        ax.set_ylabel("Predicted Future Score")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        p6 = plots_dir / "future_score_parity.png"
        plt.savefig(p6, dpi=300, bbox_inches="tight")
        plt.close()
        saved_plot_paths.append(p6)

    # ----------------------------------------------------
    # Plot 7: Ranking Metrics across K (Precision@K, Recall@K, NDCG@K)
    # ----------------------------------------------------
    ks = sorted(ranking_res.keys())
    prec_at_k = [ranking_res[k][f"Precision@{k}"] for k in ks]
    rec_at_k = [ranking_res[k][f"Recall@{k}"] for k in ks]
    ndcg_at_k = [ranking_res[k][f"NDCG@{k}"] for k in ks]

    plt.figure(figsize=(8, 5.5))
    plt.plot(ks, prec_at_k, "o-", color="#2980b9", lw=2.2, label="Precision@K")
    plt.plot(ks, rec_at_k, "s-", color="#27ae60", lw=2.2, label="Recall@K")
    plt.plot(ks, ndcg_at_k, "^-", color="#8e44ad", lw=2.2, label="NDCG@K")
    plt.xticks(ks)
    plt.ylim([0, 1.05])
    plt.xlabel("Top-K Candidates per Snapshot")
    plt.ylabel("Metric Score")
    plt.title("Ranking Performance across Top-K Critical Agents")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    p7 = plots_dir / "ranking_metrics_at_k.png"
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p7)

    return saved_plot_paths


# ==========================================
# 6. Main Execution Pipeline
# ==========================================
def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path.resolve()}")

    print(f"Loading dataset from: {csv_path.resolve()} ...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns in {time.time() - t0:.2f}s")

    # Column specifications
    id_cols = ["case_id", "step", "agent_id"]
    position_cols = ["rel_x", "rel_y", "rel_z"]
    binary_target = "is_critical"
    score_target = "criticality_score"

    node_feature_candidates = [
        "rel_x", "rel_y", "rel_z",
        "dist_to_centroid", "rank_dist_to_centroid",
        "dist_to_target", "rank_dist_to_target",
        "vx", "vy", "vz", "speed", "heading", "flight_path_angle",
        "accel", "heading_rate",
        "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
        "weighted_density", "angular_coverage",
        "degree_centrality", "closeness_centrality",
        "betweenness_centrality", "eigenvector_centrality",
    ]
    env_candidates = [
        "obstacle_density", "mean_obstacle_size", "clutter_index",
        "obstacle_clustering",
    ]

    required = id_cols + position_cols + [binary_target, score_target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    node_features = [c for c in node_feature_candidates if c in df.columns]
    env_features = [c for c in env_candidates if c in df.columns]

    print(f"Using {len(node_features)} node features and {len(env_features)} environment features.")

    # Data hygiene & stable sorting
    df = df.dropna(subset=id_cols + position_cols + [binary_target, score_target]).copy()
    df["step"] = pd.to_numeric(df["step"], errors="raise")
    df[binary_target] = pd.to_numeric(df[binary_target], errors="raise").astype(int)
    df[score_target] = pd.to_numeric(df[score_target], errors="raise").astype(float)
    df = df.sort_values(["case_id", "step", "agent_id"]).reset_index(drop=True)

    # Optional max_cases filter for quick test verification
    if args.max_cases is not None and args.max_cases > 0:
        available_cases = df["case_id"].unique()[:args.max_cases]
        df = df[df["case_id"].isin(available_cases)].copy().reset_index(drop=True)
        print(f"Subsampled to {len(available_cases)} cases for quick verification.")

    # Prevent merge collisions if dataset already contains categorical attrition_rate
    if "attrition_rate" in df.columns:
        df.rename(columns={"attrition_rate": "scenario_attrition_regime"}, inplace=True)

    # Fast vectorized attrition computation
    counts = (
        df.groupby(["case_id", "step"])["agent_id"].nunique()
          .rename("alive_count").reset_index()
          .sort_values(["case_id", "step"])
    )
    counts["initial_count"] = counts.groupby("case_id")["alive_count"].transform("first")
    counts["attrition_fraction"] = 1.0 - counts["alive_count"] / counts["initial_count"].clip(lower=1)
    counts["lost_since_previous"] = (
        counts.groupby("case_id")["alive_count"].shift(1) - counts["alive_count"]
    ).fillna(0).clip(lower=0)
    counts["attrition_rate"] = counts["lost_since_previous"] / counts["initial_count"].clip(lower=1)

    df = df.merge(
        counts[["case_id", "step", "attrition_fraction", "attrition_rate"]],
        on=["case_id", "step"], how="left"
    )
    context_features = env_features + ["attrition_fraction", "attrition_rate"]

    # Fast vectorized future score target computation (0.5s instead of 1hr iterrows)
    step_df = df[["case_id", "step"]].drop_duplicates()
    step_df["step_idx"] = step_df.groupby("case_id").cumcount()
    df = df.merge(step_df, on=["case_id", "step"], how="left")
    df["future_step_idx"] = df["step_idx"] + args.future_horizon

    lookup = df[["case_id", "step_idx", "agent_id", score_target]].rename(
        columns={"step_idx": "future_step_idx", score_target: "_future_score"}
    )
    df = df.merge(lookup, on=["case_id", "future_step_idx", "agent_id"], how="left")
    df.drop(columns=["step_idx", "future_step_idx"], inplace=True)

    # Scenario-level train/val/test split (prevents scenario data leakage)
    case_ids = df["case_id"].drop_duplicates().to_numpy()
    if len(case_ids) < 3:
        raise ValueError("Need at least 3 distinct case_id values for train/validation/test splits.")

    train_cases, temp_cases = train_test_split(case_ids, test_size=0.30, random_state=args.seed)
    val_cases, test_cases = train_test_split(temp_cases, test_size=0.50, random_state=args.seed)

    train_df = df[df.case_id.isin(train_cases)].copy()
    val_df = df[df.case_id.isin(val_cases)].copy()
    test_df = df[df.case_id.isin(test_cases)].copy()

    # Fit scalers strictly on training data
    node_scaler = StandardScaler()
    node_scaler.fit(train_df[node_features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    context_scaler = StandardScaler()
    context_scaler.fit(train_df[context_features].replace([np.inf, -np.inf], np.nan).fillna(0.0))

    def apply_scalers(frame):
        frame = frame.copy()
        xn = frame[node_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        xc = frame[context_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["_x"] = list(node_scaler.transform(xn).astype(np.float32))
        frame["_c"] = list(context_scaler.transform(xc).astype(np.float32))
        return frame

    train_df = apply_scalers(train_df)
    val_df = apply_scalers(val_df)
    test_df = apply_scalers(test_df)

    print(f"Splits: Train Cases = {len(train_cases)} ({len(train_df):,} rows) | "
          f"Val Cases = {len(val_cases)} ({len(val_df):,} rows) | "
          f"Test Cases = {len(test_cases)} ({len(test_df):,} rows)")

    # Prebuild snapshot graphs
    print("Prebuilding spatial snapshot sequences for scenarios...")
    t_seq = time.time()
    train_sequences = make_case_sequences(train_df, position_cols, binary_target, score_target, args.radius)
    val_sequences = make_case_sequences(val_df, position_cols, binary_target, score_target, args.radius)
    test_sequences = make_case_sequences(test_df, position_cols, binary_target, score_target, args.radius)
    print(f"Built all graph sequences in {time.time() - t_seq:.2f}s")

    # Instantiate model
    model = SwarmCriticalityNet(
        node_dim=len(node_features),
        context_dim=len(context_features),
        edge_dim=8,
        hidden_dim=args.hidden_dim,
        context_hidden=args.context_dim,
        dropout=0.15,
        device=device
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SwarmCriticalityNet initialized with {total_params:,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop with validation early stopping
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_bce": [], "val_bce": [],
        "train_score": [], "val_score": [],
        "train_future": [], "val_future": [],
    }

    print("\n" + "=" * 65)
    print("Beginning Training with Early Stopping")
    print("=" * 65)

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_parts = run_epoch(model, train_sequences, optimizer=optimizer, train=True, device=device)
        val_loss, val_parts = run_epoch(model, val_sequences, train=False, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bce"].append(train_parts["binary"])
        history["val_bce"].append(val_parts["binary"])
        history["train_score"].append(train_parts["score"])
        history["val_score"].append(val_parts["score"])
        history["train_future"].append(train_parts["future"])
        history["val_future"].append(val_parts["future"])

        ep_dur = time.time() - ep_t0
        print(f"Epoch {epoch:03d}/{args.epochs:03d} [{ep_dur:5.1f}s] | "
              f"Train: {train_loss:.4f} (BCE={train_parts['binary']:.3f}, Score={train_parts['score']:.4f}, Fut={train_parts['future']:.4f}) | "
              f"Val: {val_loss:.4f} (BCE={val_parts['binary']:.3f}, Score={val_parts['score']:.4f}, Fut={val_parts['future']:.4f})",
              end="")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            print("  [*] (Best Checkpoint)")
        else:
            bad_epochs += 1
            print(f"  [Patience: {bad_epochs}/{args.patience}]")
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Training completed in {(time.time() - start_train_time)/60:.2f} minutes.")

    # Load best checkpoint
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    # Save model weights and preprocessing parameters
    artifact_path = output_dir / "swarm_criticality_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "node_features": node_features,
        "context_features": context_features,
        "position_features": position_cols,
        "radius": args.radius,
        "hidden_dim": args.hidden_dim,
        "context_dim": args.context_dim,
        "node_scaler_mean": node_scaler.mean_,
        "node_scaler_scale": node_scaler.scale_,
        "context_scaler_mean": context_scaler.mean_,
        "context_scaler_scale": context_scaler.scale_,
        "future_horizon": args.future_horizon,
    }, artifact_path)
    print(f"Saved trained model checkpoint to: {artifact_path.resolve()}")

    # ==========================================
    # 7. Held-Out Test Evaluation
    # ==========================================
    print("\nRunning inference on held-out test scenarios...")
    test_pred = predict_sequences(model, test_sequences, device=device)

    preds_csv_path = output_dir / "swarm_criticality_test_predictions.csv"
    test_pred.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions ({len(test_pred):,} rows) to: {preds_csv_path.resolve()}")

    y_true = test_pred[binary_target].to_numpy().astype(int)
    p = test_pred["prob_critical"].to_numpy()
    y_hat = (p >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "PR-AUC (average precision)": float(average_precision_score(y_true, p)),
        "ROC-AUC": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else 0.0,
    }

    valid_score = np.isfinite(test_pred[score_target].to_numpy())
    metrics["score MAE"] = float(mean_absolute_error(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ))
    metrics["score RMSE"] = float(mean_squared_error(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ) ** 0.5)
    metrics["score R2"] = float(r2_score(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ))

    valid_future = np.isfinite(test_pred["_future_score"].to_numpy())
    if valid_future.any():
        metrics["future score MAE"] = float(mean_absolute_error(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ))
        metrics["future score RMSE"] = float(mean_squared_error(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ) ** 0.5)
        metrics["future score R2"] = float(r2_score(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ))
    metrics["future_horizon"] = args.future_horizon

    # Top-K Ranking Metrics
    ranking_res = compute_ranking_metrics(test_pred, k_list=(1, 2, 3, 5, 8, 10))

    # Print summary tables to stdout
    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET EVALUATION METRICS")
    print("=" * 55)
    metrics_s = pd.Series({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}, name="Value")
    print(metrics_s.to_string())

    print("\n" + "=" * 55)
    print("RANKING METRICS ACROSS K (Top-K per Snapshot)")
    print("=" * 55)
    ranking_rows = []
    for k in sorted(ranking_res.keys()):
        ranking_rows.append({
            "K": k,
            "Precision@K": f"{ranking_res[k][f'Precision@{k}']:.4f}",
            "Recall@K": f"{ranking_res[k][f'Recall@{k}']:.4f}",
            "NDCG@K": f"{ranking_res[k][f'NDCG@{k}']:.4f}",
        })
    ranking_df = pd.DataFrame(ranking_rows)
    print(ranking_df.to_string(index=False))

    # Save metrics JSON
    metrics_out = {
        "test_metrics": metrics,
        "ranking_metrics": ranking_res,
        "config": vars(args),
    }
    metrics_json_path = output_dir / "test_metrics_summary.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved metrics summary to: {metrics_json_path.resolve()}")

    # ==========================================
    # 8. Plot Generation & Saving
    # ==========================================
    print("\nGenerating and saving all evaluation plots...")
    saved_plots = generate_all_plots(history, test_pred, metrics, ranking_res, plots_dir)
    print(f"Successfully generated and saved {len(saved_plots)} plots to {plots_dir.resolve()}:")
    for p in saved_plots:
        print(f"  - {p.name}")

    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()