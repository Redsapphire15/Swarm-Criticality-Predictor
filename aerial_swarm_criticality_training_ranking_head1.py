# -*- coding: utf-8 -*-
"""aerial_swarm_criticality_training.py

Aerial Swarm Critical-Agent Continuous Severity Ranking & Identification
Architecture: GraphSAGE -> Edge-Aware Attention (GAT, Vectorized) -> Global Context Fusion -> Per-Agent GRU

Primary Objective:
Rank ALL alive agents in descending order of their continuous criticality severity [0, 1].
Top-10 candidate agents are extracted strictly based on predicted continuous criticality scores.

Loss Formulation:
L_total = lambda_reg * L_current_regression
        + lambda_rank * L_rank
        + lambda_future * L_future_regression
        + lambda_cls * L_binary_classification

Where L_rank is the graded pairwise ranking loss:
L_rank = (1 / Z) * sum_{i, j: target_i - target_j > margin} w_ij * softplus(-(pred_i - pred_j) / temperature)
with w_ij = |target_i - target_j|^alpha

Validation Checkpoint Selection: Validation Graded NDCG@10 (tie-breakers: Spearman rho, score MSE).
All generated plots and evaluation metrics are saved to ./plots/ and ./outputs/.
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
import scipy.stats as stats
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
    parser = argparse.ArgumentParser(description="Aerial Swarm Continuous Criticality Ranking Training")
    parser.add_argument("--csv_path", type=str, default="/home/diro/kavin/beh/AE5510/swarm_criticality_dataset_100_scenarios.csv",
                        help="Path to CSV dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save model artifacts and predictions")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                        help="Directory to save generated figures")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val Graded NDCG@10 improvement)")
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

    # Loss weights for Section 4
    parser.add_argument("--lambda_reg", type=float, default=1.0,
                        help="Weight for continuous criticality MSE regression loss")
    parser.add_argument("--lambda_rank", type=float, default=1.0,
                        help="Weight for graded pairwise ranking loss")
    parser.add_argument("--lambda_future", type=float, default=0.25,
                        help="Weight for future score MSE forecast loss")
    parser.add_argument("--lambda_cls", type=float, default=0.5,
                        help="Weight for binary classification BCE loss")

    # Graded ranking hyperparameters for Section 3
    parser.add_argument("--ranking_margin", type=float, default=0.02,
                        help="Minimum target score difference to form training pairs")
    parser.add_argument("--ranking_alpha", type=float, default=1.0,
                        help="Exponent for severity difference weighting w_ij = |target_i - target_j|^alpha")
    parser.add_argument("--ranking_temperature", type=float, default=1.0,
                        help="Temperature scaling in softplus(-diff / temperature)")
    parser.add_argument("--experiment", type=str, default="exp1",
                        choices=["baseline", "exp1", "exp2"],
                        help="Experiment profile: baseline (no ranking loss), exp1 (balanced), exp2 (ranking-tuned)")
    return parser.parse_args()


# ==========================================
# 2. Graph & Sequence Preprocessing
# ==========================================
def make_snapshot(frame, position_cols, binary_target, score_target, radius, device=None):
    """
    Constructs a directed spatial graph for a single (case_id, step) snapshot.
    Nodes are sorted by agent_id for deterministic hidden state alignment.
    Pre-allocates tensors on the target device for high throughput.
    """
    frame = frame.sort_values("agent_id").reset_index(drop=True)
    pos = frame[position_cols].to_numpy(dtype=np.float32)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)

    # Distance matrix and edge detection within communication radius
    dmat = torch.cdist(pos_t, pos_t)
    src, dst = torch.where((dmat <= radius) & (dmat > 0))
    edge_index = torch.stack([src, dst], dim=0) if len(src) else torch.empty((2, 0), dtype=torch.long, device=device)

    # Edge features: relative position, distance, relative velocity, radial velocity
    rel_pos = pos_t[src] - pos_t[dst] if len(src) else torch.empty((0, 3), dtype=torch.float32, device=device)
    dist = dmat[src, dst].unsqueeze(1) if len(src) else torch.empty((0, 1), dtype=torch.float32, device=device)
    if all(c in frame.columns for c in ["vx", "vy", "vz"]):
        vel = torch.tensor(frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float32), device=device)
        rel_vel = vel[src] - vel[dst] if len(src) else torch.empty((0, 3), dtype=torch.float32, device=device)
        radial = (rel_pos * rel_vel).sum(dim=1, keepdim=True) / dist.clamp_min(1e-6)
    else:
        rel_vel = torch.zeros((len(src), 3), dtype=torch.float32, device=device)
        radial = torch.zeros((len(src), 1), dtype=torch.float32, device=device)
    edge_attr = torch.cat([rel_pos, dist, rel_vel, radial], dim=1)

    x = torch.tensor(np.stack(frame["_x"].to_numpy()), dtype=torch.float32, device=device)
    c = torch.tensor(np.stack(frame["_c"].to_numpy()), dtype=torch.float32, device=device)
    y = torch.tensor(frame[binary_target].to_numpy(dtype=np.float32), device=device)
    score = torch.tensor(frame[score_target].to_numpy(dtype=np.float32), device=device)
    future = torch.tensor(frame["_future_score"].to_numpy(dtype=np.float32), device=device)
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


def make_case_sequences(frame, position_cols, binary_target, score_target, radius, device=None):
    """
    Groups scenario rows into chronological sequences of snapshot graphs.
    """
    sequences = {}
    for case_id, case in frame.groupby("case_id", sort=False):
        snapshots = []
        for step, snap in case.groupby("step", sort=True):
            snapshots.append((step, make_snapshot(snap, position_cols, binary_target, score_target, radius, device=device)))
        sequences[case_id] = snapshots
    return sequences


# ==========================================
# 3. Model Architecture (GraphSAGE + GAT + GRU + Heads)
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
    Uses PyTorch scatter_reduce and index_add for high numerical efficiency.
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
    """
    Hierarchical spatial-temporal model:
    Node Encoder -> GraphSAGE -> Edge-Aware GAT -> Global Context Fusion -> Per-Agent GRU
    Prediction Heads:
    1. binary_head: BCE classification logits (probability agent is critical)
    2. score_head: Continuous criticality severity regression in [0, 1] via Sigmoid
    3. future_head: Future horizon criticality regression in [0, 1] via Sigmoid
    """
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
        # Global graph pooling (Mean + Max) and context fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + context_hidden, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        # Continuous criticality severity head [0, 1] - Primary ranking signal
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward_snapshot(self, snap, hidden_by_id):
        x = snap["x"].to(self.device, non_blocking=True)
        edge_index = snap["edge_index"].to(self.device, non_blocking=True)
        edge_attr = snap["edge_attr"].to(self.device, non_blocking=True)
        context = snap["context"].to(self.device, non_blocking=True)
        ids = snap["agent_ids"]

        h = self.node_encoder(x)
        h = self.sage(h, edge_index)
        h = self.gat(h, edge_index, edge_attr)

        # Global graph pooling (Mean + Max)
        global_mean = h.mean(dim=0, keepdim=True)
        global_max = h.max(dim=0, keepdim=True).values
        global_vec = torch.cat([global_mean, global_max], dim=-1).expand(h.size(0), -1)

        c = self.context_encoder(context.unsqueeze(0)).expand(h.size(0), -1)
        fused = self.fusion(torch.cat([h, global_vec, c], dim=-1))

        # Retrieve prior recurrent hidden state by agent_id
        previous = [hidden_by_id.get(aid, torch.zeros(self.gru.hidden_size, device=self.device)) for aid in ids]
        h_prev = torch.stack(previous, dim=0)
        h_now = self.gru(fused, h_prev)

        # Drop hidden states for absent agents and update current alive agents
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
# 4. Graded Pairwise Ranking Loss & Snapshot Loss
# ==========================================
def graded_pairwise_ranking_loss(score_pred, score_target, margin=0.02, alpha=1.0, temperature=1.0):
    """
    Section 3: Graded Pairwise Ranking Loss (Vectorized Mask)
    L_rank = (1 / Z) * sum_{i, j: target_i - target_j > margin} w_ij * softplus(-(pred_i - pred_j) / temperature)
    where: w_ij = |target_i - target_j|^alpha
    Z = sum w_ij + 1e-8 (normalization over valid pairs)
    """
    target_diff = score_target.unsqueeze(1) - score_target.unsqueeze(0)
    mask = (target_diff > margin).float()
    w = target_diff.clamp_min(0.0).pow(alpha) * mask
    pred_diff = score_pred.unsqueeze(1) - score_pred.unsqueeze(0)
    pair_loss = w * F.softplus(-pred_diff / temperature)
    return pair_loss.sum() / (w.sum() + 1e-8)


def snapshot_loss(out, snap, lambda_reg=1.0, lambda_rank=1.0, lambda_future=0.25, lambda_cls=0.5,
                  ranking_margin=0.02, ranking_alpha=1.0, ranking_temperature=1.0, device="cpu"):
    """
    Section 4: Combined multi-task objective:
    L_total = lambda_reg * L_current_regression
            + lambda_rank * L_rank
            + lambda_future * L_future_regression
            + lambda_cls * L_binary_classification
    """
    y = snap["y"].to(device, non_blocking=True)
    score = snap["score"].to(device, non_blocking=True)
    future = snap["future"].to(device, non_blocking=True)

    # 1. Current continuous criticality regression loss (MSE)
    loss_reg = F.mse_loss(out["score_pred"], score)

    # 2. Graded pairwise ranking loss
    loss_rank = graded_pairwise_ranking_loss(
        out["score_pred"], score,
        margin=ranking_margin, alpha=ranking_alpha, temperature=ranking_temperature
    )

    # 3. Future criticality forecast loss (MSE on valid timesteps)
    valid_future = torch.isfinite(future)
    if valid_future.any():
        loss_future = F.mse_loss(out["future_pred"][valid_future], future[valid_future])
    else:
        loss_future = torch.zeros((), device=device)

    # 4. Binary classification loss (BCEWithLogits)
    loss_cls = F.binary_cross_entropy_with_logits(out["logits"], y)

    total = (
        lambda_reg * loss_reg
        + lambda_rank * loss_rank
        + lambda_future * loss_future
        + lambda_cls * loss_cls
    )

    return total, {
        "regression": float(loss_reg.detach().cpu()),
        "ranking": float(loss_rank.detach().cpu()),
        "future": float(loss_future.detach().cpu()),
        "classification": float(loss_cls.detach().cpu()),
    }


def snapshot_ranking_metrics_fast(score_pred, score_target, k=10):
    """
    High-performance GPU evaluation of Graded NDCG@10 and Spearman correlation
    for validation checkpoint monitoring.
    """
    score_target = score_target.to(score_pred.device, non_blocking=True)
    n = score_pred.size(0)
    kk = min(k, n)
    if kk == 0:
        return 0.0, float("nan"), float("nan")

    # Graded NDCG@K
    topk_idx = torch.topk(score_pred, k=kk).indices
    gains = torch.pow(2.0, score_target[topk_idx].clamp_min(0.0)) - 1.0
    discounts = 1.0 / torch.log2(torch.arange(2, kk + 2, device=score_pred.device, dtype=torch.float32))
    dcg = (gains * discounts).sum().item()

    ideal_scores = torch.topk(score_target, k=kk).values
    ideal_gains = torch.pow(2.0, ideal_scores.clamp_min(0.0)) - 1.0
    idcg = (ideal_gains * discounts).sum().item()
    ndcg = (dcg / idcg) if idcg > 0 else float("nan")

    # Spearman rank correlation
    p_np = score_pred.detach().cpu().numpy()
    t_np = score_target.detach().cpu().numpy()
    if n >= 2 and not np.allclose(t_np, t_np[0]) and not np.allclose(p_np, p_np[0]):
        spearman = float(stats.spearmanr(p_np, t_np).statistic)
    else:
        spearman = float("nan")

    return ndcg, spearman, float(F.mse_loss(score_pred, score_target).item())


def run_case(model, snapshots, optimizer=None, train=False, collect=False, eval_ranking=False,
             lambda_reg=1.0, lambda_rank=1.0, lambda_future=0.25, lambda_cls=0.5,
             ranking_margin=0.02, ranking_alpha=1.0, ranking_temperature=1.0, device="cpu"):
    """
    Runs recurrent forward pass across a single scenario.
    Resets recurrent hidden states between scenarios.
    """
    hidden_by_id = {}
    losses = []
    part_losses = {"regression": [], "ranking": [], "future": [], "classification": []}
    collected = []
    case_ndcgs, case_spearmans, case_mses = [], [], []
    model.train(train)

    context_manager = torch.enable_grad() if train else torch.no_grad()
    with context_manager:
        for step, snap in snapshots:
            out = model.forward_snapshot(snap, hidden_by_id)
            hidden_by_id = out["hidden_by_id"]
            loss, parts = snapshot_loss(
                out, snap,
                lambda_reg=lambda_reg, lambda_rank=lambda_rank,
                lambda_future=lambda_future, lambda_cls=lambda_cls,
                ranking_margin=ranking_margin, ranking_alpha=ranking_alpha,
                ranking_temperature=ranking_temperature, device=device
            )
            losses.append(loss)
            for k, v in parts.items():
                part_losses[k].append(v)

            if eval_ranking:
                ndcg, spear, mse_s = snapshot_ranking_metrics_fast(out["score_pred"], snap["score"], k=10)
                case_ndcgs.append(ndcg)
                case_spearmans.append(spear)
                case_mses.append(mse_s)

            if collect:
                probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
                score_pred = out["score_pred"].detach().cpu().numpy()
                future_pred = out["future_pred"].detach().cpu().numpy()
                frame = snap["frame"].copy()
                frame["prob_critical"] = probs
                frame["pred_criticality_score"] = score_pred
                frame["pred_future_score"] = future_pred
                frame["is_alive"] = 1
                collected.append(frame)

        if not losses:
            return 0.0, {k: 0.0 for k in part_losses}, collected, (case_ndcgs, case_spearmans, case_mses)
        total = torch.stack(losses).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    avg_parts = {k: float(np.mean(v)) for k, v in part_losses.items()}
    return float(total.detach().cpu()), avg_parts, collected, (case_ndcgs, case_spearmans, case_mses)


def run_epoch(model, sequences, optimizer=None, train=False, collect=False, eval_ranking=False,
              lambda_reg=1.0, lambda_rank=1.0, lambda_future=0.25, lambda_cls=0.5,
              ranking_margin=0.02, ranking_alpha=1.0, ranking_temperature=1.0, device="cpu"):
    """
    Runs one epoch across all scenarios.
    """
    case_losses = []
    part_accum = {"regression": [], "ranking": [], "future": [], "classification": []}
    all_frames = []
    all_ndcgs, all_spearmans, all_mses = [], [], []
    keys = list(sequences.keys())
    if train:
        random.shuffle(keys)
    for case_id in keys:
        loss, parts, frames, rank_metrics = run_case(
            model, sequences[case_id], optimizer=optimizer, train=train,
            collect=collect, eval_ranking=eval_ranking,
            lambda_reg=lambda_reg, lambda_rank=lambda_rank,
            lambda_future=lambda_future, lambda_cls=lambda_cls,
            ranking_margin=ranking_margin, ranking_alpha=ranking_alpha,
            ranking_temperature=ranking_temperature, device=device
        )
        case_losses.append(loss)
        for k, v in parts.items():
            part_accum[k].append(v)
        if eval_ranking:
            c_n, c_s, c_m = rank_metrics
            all_ndcgs.extend(c_n)
            all_spearmans.extend(c_s)
            all_mses.extend(c_m)
        if collect:
            all_frames.extend(frames)

    avg_total = float(np.mean(case_losses)) if case_losses else float("nan")
    avg_parts = {k: float(np.mean(v)) for k, v in part_accum.items()}
    collected_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    epoch_ranking = {
        "graded_ndcg10": float(np.nanmean(all_ndcgs)) if all_ndcgs else float("nan"),
        "spearman": float(np.nanmean(all_spearmans)) if all_spearmans else float("nan"),
        "score_mse": float(np.nanmean(all_mses)) if all_mses else float("nan"),
    }
    return avg_total, avg_parts, collected_df, epoch_ranking


def predict_sequences(model, sequences, device="cpu"):
    """
    Generates predictions for all scenarios without using ground-truth labels during inference.
    """
    all_frames = []
    model.eval()
    with torch.no_grad():
        for case_id, snapshots in sequences.items():
            _, _, frames, _ = run_case(model, snapshots, train=False, collect=True, eval_ranking=False, device=device)
            all_frames.extend(frames)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ==========================================
# 5. Top-10 Inference & Candidate Extraction
# ==========================================
def generate_top10_candidates(predictions):
    """
    Section 5: Top-10 continuous criticality severity candidate generation.
    For each snapshot (case_id, step):
      1. Rank all alive agents strictly by predicted continuous criticality score descending.
      2. Return the top 10 agents (or all alive agents if < 10).
      3. Assign ranks from 1 to K.
    Output columns:
      case_id, step, rank, agent_id, pred_criticality_score,
      criticality_score, is_critical, pred_future_score, is_alive, prob_critical
    """
    records = []
    if "is_alive" not in predictions.columns:
        predictions = predictions.copy()
        predictions["is_alive"] = 1

    for (case_id, step), group in predictions.groupby(["case_id", "step"], sort=False):
        top_group = group.sort_values("pred_criticality_score", ascending=False).head(10).copy()
        top_group["rank"] = np.arange(1, len(top_group) + 1)
        records.append(top_group)

    if not records:
        return pd.DataFrame(columns=[
            "case_id", "step", "rank", "agent_id", "pred_criticality_score",
            "criticality_score", "is_critical", "pred_future_score", "is_alive", "prob_critical"
        ])

    top10_df = pd.concat(records, ignore_index=True)
    ordered_cols = [
        "case_id", "step", "rank", "agent_id", "pred_criticality_score",
        "criticality_score", "is_critical", "pred_future_score", "is_alive", "prob_critical"
    ]
    ordered_cols = [c for c in ordered_cols if c in top10_df.columns]
    return top10_df[ordered_cols]


# ==========================================
# 6. Evaluation Metrics Suite
# ==========================================
def compute_snapshot_ranking_metrics(pred_scores, true_scores, true_binary=None, k_list=(1, 3, 5, 8, 10)):
    """
    Section 6: Evaluates continuous ranking metrics for a single swarm snapshot:
    - Spearman rank correlation
    - Kendall's Tau
    - Graded NDCG@1, @3, @5, @8, @10
    - Top-10 severity MAE and RMSE
    - Mean absolute pairwise ordering error
    - Secondary binary Precision@10 and Recall@10
    """
    n = len(pred_scores)
    # Sort agents in descending order of predicted continuous criticality
    order = np.argsort(pred_scores)[::-1]
    sorted_pred = pred_scores[order]
    sorted_true = true_scores[order]

    # 1. Spearman & Kendall
    if n >= 2 and not np.allclose(true_scores, true_scores[0]) and not np.allclose(pred_scores, pred_scores[0]):
        spearman = float(stats.spearmanr(pred_scores, true_scores).statistic)
        kendall = float(stats.kendalltau(pred_scores, true_scores).statistic)
    else:
        spearman = float("nan")
        kendall = float("nan")

    # 2. Graded NDCG@K
    max_k = max(k_list)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    ndcgs = {}
    ideal_scores = np.sort(true_scores)[::-1]
    for k in k_list:
        kk = min(k, n)
        gains = (2.0 ** np.clip(sorted_true[:kk], 0.0, None)) - 1.0
        dcg = (gains * discounts[:kk]).sum()
        ideal_gains = (2.0 ** np.clip(ideal_scores[:kk], 0.0, None)) - 1.0
        idcg = (ideal_gains * discounts[:kk]).sum()
        ndcgs[k] = float(dcg / idcg) if idcg > 0 else float("nan")

    # 3. Top-10 Severity MAE & RMSE
    kk10 = min(10, n)
    top10_mae = float(np.mean(np.abs(sorted_pred[:kk10] - sorted_true[:kk10])))
    top10_rmse = float(np.sqrt(np.mean((sorted_pred[:kk10] - sorted_true[:kk10]) ** 2)))

    # 4. Mean absolute pairwise ordering error
    # For pairs where true_i > true_j, fraction where pred_i < pred_j (inversion rate)
    t_diff = true_scores[:, None] - true_scores[None, :]
    p_diff = pred_scores[:, None] - pred_scores[None, :]
    pair_mask = t_diff > 0.001
    if pair_mask.any():
        inversions = (p_diff[pair_mask] < 0).astype(float) + 0.5 * (p_diff[pair_mask] == 0).astype(float)
        ordering_error = float(inversions.mean())
    else:
        ordering_error = 0.0

    # 5. Secondary binary Precision@10 & Recall@10
    if true_binary is not None:
        sorted_bin = true_binary[order]
        hits = sorted_bin[:kk10].sum()
        tot_pos = true_binary.sum()
        p10 = float(hits / kk10)
        r10 = float(hits / tot_pos) if tot_pos > 0 else float("nan")
    else:
        p10, r10 = float("nan"), float("nan")

    return {
        "spearman": spearman,
        "kendall": kendall,
        "graded_ndcg1": ndcgs.get(1, float("nan")),
        "graded_ndcg3": ndcgs.get(3, float("nan")),
        "graded_ndcg5": ndcgs.get(5, float("nan")),
        "graded_ndcg8": ndcgs.get(8, float("nan")),
        "graded_ndcg10": ndcgs.get(10, float("nan")),
        "top10_mae": top10_mae,
        "top10_rmse": top10_rmse,
        "pairwise_ordering_error": ordering_error,
        "precision10": p10,
        "recall10": r10,
    }


def evaluate_test_ranking(predictions, k_list=(1, 3, 5, 8, 10)):
    """
    Evaluates ranking independently within each test swarm snapshot,
    generates per-snapshot metrics table, and returns aggregated summary.
    """
    rows = []
    for (case_id, step), group in predictions.groupby(["case_id", "step"], sort=False):
        if len(group) == 0:
            continue
        p_scores = group["pred_criticality_score"].to_numpy(dtype=float)
        t_scores = group["criticality_score"].to_numpy(dtype=float)
        t_bin = group["is_critical"].to_numpy(dtype=int) if "is_critical" in group.columns else None

        m = compute_snapshot_ranking_metrics(p_scores, t_scores, t_bin, k_list=k_list)
        m["case_id"] = case_id
        m["step"] = step
        m["num_agents"] = len(group)
        rows.append(m)

    per_snapshot_df = pd.DataFrame(rows)

    # Reorder columns
    id_cols = ["case_id", "step", "num_agents"]
    metric_cols = [c for c in per_snapshot_df.columns if c not in id_cols]
    per_snapshot_df = per_snapshot_df[id_cols + metric_cols]

    # Aggregate across valid snapshots
    agg_summary = {}
    for col in metric_cols:
        vals = per_snapshot_df[col].dropna()
        agg_summary[col] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "median": float(vals.median()) if len(vals) else float("nan"),
            "valid_snapshots": int(len(vals)),
            "total_snapshots": int(len(per_snapshot_df)),
        }

    return per_snapshot_df, agg_summary


def evaluate_ranking_by_attrition(predictions, k=10):
    """
    Section 7: Attrition-wise continuous ranking evaluation (None, Light, Medium, Heavy).
    """
    preds = predictions.copy()
    if "scenario_attrition_regime" in preds.columns:
        preds["regime"] = preds["scenario_attrition_regime"].astype(str).str.lower()
    elif "attrition_fraction" in preds.columns:
        case_max = preds.groupby("case_id")["attrition_fraction"].transform("max")
        regime = pd.Series("none", index=preds.index)
        regime[case_max > 0.0] = "light"
        regime[case_max > 0.25] = "medium"
        regime[case_max > 0.50] = "heavy"
        preds["regime"] = regime
    else:
        preds["regime"] = "none"

    regimes_order = ["none", "light", "medium", "heavy"]
    rows = []
    for reg in regimes_order:
        sub = preds[preds["regime"] == reg]
        if len(sub) == 0:
            continue
        _, agg = evaluate_test_ranking(sub, k_list=(k,))
        rows.append({
            "Attrition Condition": reg.capitalize(),
            f"Graded_NDCG@{k}": agg[f"graded_ndcg{k}"]["mean"],
            "Spearman": agg["spearman"]["mean"],
            "Kendall": agg["kendall"]["mean"],
            "Top10_MAE": agg["top10_mae"]["mean"],
            "Top10_RMSE": agg["top10_rmse"]["mean"],
            "Ordering_Error": agg["pairwise_ordering_error"]["mean"],
            f"Precision@{k}": agg["precision10"]["mean"],
            f"Recall@{k}": agg["recall10"]["mean"],
            "Snapshots": agg[f"graded_ndcg{k}"]["total_snapshots"],
        })
    return pd.DataFrame(rows)


def evaluate_ranking_by_swarm_size(predictions, k=10):
    """
    Section 7: Evaluation across initial swarm sizes.
    """
    preds = predictions.copy()
    if "num_agents" in preds.columns:
        size_col = "num_agents"
    elif "initial_count" in preds.columns:
        size_col = "initial_count"
    else:
        case_init = preds.groupby("case_id")["agent_id"].nunique()
        preds["num_agents"] = preds["case_id"].map(case_init)
        size_col = "num_agents"

    unique_sizes = sorted(preds[size_col].unique())
    rows = []
    for sz in unique_sizes:
        sub = preds[preds[size_col] == sz]
        if len(sub) == 0:
            continue
        _, agg = evaluate_test_ranking(sub, k_list=(k,))
        rows.append({
            "Swarm Size": int(sz),
            f"Graded_NDCG@{k}": agg[f"graded_ndcg{k}"]["mean"],
            "Spearman": agg["spearman"]["mean"],
            "Kendall": agg["kendall"]["mean"],
            "Top10_MAE": agg["top10_mae"]["mean"],
            "Top10_RMSE": agg["top10_rmse"]["mean"],
            "Ordering_Error": agg["pairwise_ordering_error"]["mean"],
            f"Precision@{k}": agg["precision10"]["mean"],
            f"Recall@{k}": agg["recall10"]["mean"],
            "Snapshots": agg[f"graded_ndcg{k}"]["total_snapshots"],
        })
    return pd.DataFrame(rows)


# ==========================================
# 7. Comprehensive Plotting Suite (Section 8)
# ==========================================
def generate_all_plots(history, test_pred, top10_df, per_snapshot_df, agg_summary, attrition_df, plots_dir):
    """
    Section 8: Generates the 7 required visualizations:
    1. Predicted vs ground-truth continuous criticality scatter plot
    2. Predicted vs ground-truth severity ranking comparison
    3. Spearman and Kendall correlation distributions across snapshots
    4. Graded NDCG@K curves
    5. Top-10 predicted vs actual criticality scores for representative snapshots
    6. Pairwise ordering accuracy across severity differences
    7. Performance across none/light/medium/heavy attrition regimes
    Plus supporting loss curves and diagnostic figures.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plot_paths = []
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

    y_score_true = test_pred["criticality_score"].to_numpy(dtype=float)
    y_score_pred = test_pred["pred_criticality_score"].to_numpy(dtype=float)

    # ----------------------------------------------------
    # Plot 1: Predicted vs Ground-Truth Continuous Criticality Scatter
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(y_score_true, y_score_pred, gridsize=50, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Agent Density")
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect Parity (y = x)")

    mae = mean_absolute_error(y_score_true, y_score_pred)
    rmse = mean_squared_error(y_score_true, y_score_pred) ** 0.5
    r2 = r2_score(y_score_true, y_score_pred)
    ax.set_title(f"Predicted vs Ground-Truth Continuous Criticality\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    ax.set_xlabel("Ground-Truth Criticality Severity [0, 1]")
    ax.set_ylabel("Predicted Criticality Severity [0, 1]")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    p1 = plots_dir / "pred_vs_true_criticality_scatter.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p1)

    # ----------------------------------------------------
    # Plot 2: Predicted vs Ground-Truth Severity Ranking Comparison
    # ----------------------------------------------------
    true_ranks = []
    pred_ranks = []
    for _, g in test_pred.groupby(["case_id", "step"]):
        if len(g) >= 2:
            tr = g["criticality_score"].rank(ascending=False, method="min").to_numpy()
            pr = g["pred_criticality_score"].rank(ascending=False, method="min").to_numpy()
            true_ranks.extend(tr)
            pred_ranks.extend(pr)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(true_ranks, pred_ranks, gridsize=40, cmap="Blues", mincnt=1)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Agent Pair Density")
    max_r = max(max(true_ranks), max(pred_ranks))
    ax.plot([1, max_r], [1, max_r], "r--", lw=2, label="Perfect Rank Concordance (y = x)")
    overall_spearman = agg_summary["spearman"]["mean"]
    ax.set_title(f"Severity Ranking Comparison (Mean Spearman ρ: {overall_spearman:.4f})")
    ax.set_xlabel("Ground-Truth Severity Rank (1 = Highest Criticality)")
    ax.set_ylabel("Predicted Severity Rank (1 = Highest Criticality)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    p2 = plots_dir / "severity_ranking_comparison.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p2)

    # ----------------------------------------------------
    # Plot 3: Spearman and Kendall Correlation Distributions Across Snapshots
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sp_vals = per_snapshot_df["spearman"].dropna()
    kd_vals = per_snapshot_df["kendall"].dropna()

    axes[0].hist(sp_vals, bins=30, color="#2980b9", edgecolor="black", alpha=0.75, density=True)
    axes[0].axvline(sp_vals.mean(), color="red", lw=2, label=f"Mean = {sp_vals.mean():.4f}")
    axes[0].axvline(sp_vals.median(), color="orange", lw=2, linestyle="--", label=f"Median = {sp_vals.median():.4f}")
    axes[0].set_title("Spearman Rank Correlation (ρ) Distribution")
    axes[0].set_xlabel("Snapshot Spearman Correlation")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].hist(kd_vals, bins=30, color="#27ae60", edgecolor="black", alpha=0.75, density=True)
    axes[1].axvline(kd_vals.mean(), color="red", lw=2, label=f"Mean = {kd_vals.mean():.4f}")
    axes[1].axvline(kd_vals.median(), color="orange", lw=2, linestyle="--", label=f"Median = {kd_vals.median():.4f}")
    axes[1].set_title("Kendall's Tau (τ) Distribution")
    axes[1].set_xlabel("Snapshot Kendall's Tau")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p3 = plots_dir / "correlation_distributions.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p3)

    # ----------------------------------------------------
    # Plot 4: Graded NDCG@K Curves
    # ----------------------------------------------------
    ks = [1, 3, 5, 8, 10]
    graded_means = [agg_summary[f"graded_ndcg{k}"]["mean"] for k in ks]
    graded_stds = [agg_summary[f"graded_ndcg{k}"]["std"] for k in ks]

    plt.figure(figsize=(8, 5.5))
    plt.plot(ks, graded_means, "o-", color="#8e44ad", lw=2.5, label="Graded NDCG@K (Continuous Severity)")
    plt.fill_between(ks, np.array(graded_means) - np.array(graded_stds),
                     np.array(graded_means) + np.array(graded_stds), color="#8e44ad", alpha=0.2, label="±1 Std Dev")
    plt.xticks(ks)
    plt.ylim([0.7, 1.02])
    plt.xlabel("Top-K Candidate Cutoff (K)")
    plt.ylabel("Graded NDCG@K Score")
    plt.title("Graded NDCG@K across Candidate Cutoffs")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    p4 = plots_dir / "graded_ndcg_at_k_curves.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p4)

    # ----------------------------------------------------
    # Plot 5: Top-10 Predicted vs Actual Criticality Scores for Representative Snapshots
    # ----------------------------------------------------
    # Pick two distinct snapshots: early snapshot and late/attrition snapshot
    cases = list(test_pred["case_id"].unique())
    sample_c1 = cases[0]
    sample_c2 = cases[min(1, len(cases) - 1)]

    snap_df1 = top10_df[top10_df["case_id"] == sample_c1].sort_values("step")
    first_step = snap_df1["step"].iloc[0]
    sample1 = top10_df[(top10_df["case_id"] == sample_c1) & (top10_df["step"] == first_step)].sort_values("rank")

    snap_df2 = top10_df[top10_df["case_id"] == sample_c2].sort_values("step")
    mid_step = snap_df2["step"].iloc[len(snap_df2) // 2]
    sample2 = top10_df[(top10_df["case_id"] == sample_c2) & (top10_df["step"] == mid_step)].sort_values("rank")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sample, title in [(axes[0], sample1, f"Scenario: {sample_c1} (Step: {first_step})"),
                              (axes[1], sample2, f"Scenario: {sample_c2} (Step: {mid_step})")]:
        ranks = sample["rank"].to_numpy()
        labels = [f"Rank {r}: Agent {aid}" for r, aid in zip(ranks, sample["agent_id"])]
        p_sc = sample["pred_criticality_score"].to_numpy()
        t_sc = sample["criticality_score"].to_numpy()
        crits = sample["is_critical"].to_numpy()

        colors = ["#27ae60" if c == 1 else "#e74c3c" for c in crits]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos[::-1], p_sc[::-1], color=colors[::-1], alpha=0.7, edgecolor="black", label="Predicted Severity")
        ax.scatter(t_sc[::-1], y_pos[::-1], color="black", s=60, zorder=5, label="Ground-Truth Severity")

        ax.set_yticks(y_pos[::-1])
        ax.set_yticklabels(labels)
        ax.set_xlim([0, 1.05])
        ax.set_xlabel("Continuous Criticality Severity [0, 1]")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    custom_legend = [
        Patch(facecolor="#27ae60", edgecolor="black", label="Ground Truth: Critical"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Ground Truth: Decoy"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=8, label="True Severity"),
    ]
    axes[0].legend(handles=custom_legend, loc="lower right")
    plt.tight_layout()
    p5 = plots_dir / "top10_representative_snapshots.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p5)

    # ----------------------------------------------------
    # Plot 6: Pairwise Ordering Accuracy Across Severity Differences
    # ----------------------------------------------------
    # Sample snapshot pairs and group by |target_i - target_j|
    diff_bins = [(0.01, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = ["0.01–0.10", "0.10–0.20", "0.20–0.40", "0.40–0.60", "0.60–0.80", "0.80–1.00"]
    bin_correct = [0] * len(diff_bins)
    bin_total = [0] * len(diff_bins)

    # Sample snapshots for fast vectorized calculation
    sample_groups = [g for _, g in test_pred.groupby(["case_id", "step"])]
    if len(sample_groups) > 300:
        sample_groups = sample_groups[::len(sample_groups)//300]

    for g in sample_groups:
        if len(g) < 2:
            continue
        ts = g["criticality_score"].to_numpy()
        ps = g["pred_criticality_score"].to_numpy()
        td = ts[:, None] - ts[None, :]
        pd_ = ps[:, None] - ps[None, :]

        for idx, (low, high) in enumerate(diff_bins):
            mask = (td >= low) & (td < high)
            if mask.any():
                correct = (pd_[mask] > 0).sum()
                total = mask.sum()
                bin_correct[idx] += correct
                bin_total[idx] += total

    accuracies = [((c / t) * 100.0) if t > 0 else 0.0 for c, t in zip(bin_correct, bin_total)]

    plt.figure(figsize=(8.5, 5.5))
    bars = plt.bar(bin_labels, accuracies, color="#3498db", edgecolor="black", alpha=0.85)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{acc:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylim([50, 105])
    plt.xlabel("Ground-Truth Severity Difference |target_i - target_j|")
    plt.ylabel("Pairwise Ordering Accuracy (%)")
    plt.title("Pairwise Ordering Accuracy across Severity Differences")
    plt.grid(True, linestyle=":", alpha=0.6)
    p6 = plots_dir / "pairwise_ordering_accuracy.png"
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p6)

    # ----------------------------------------------------
    # Plot 7: Performance Across Attrition Regimes
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if not attrition_df.empty:
        conds = attrition_df["Attrition Condition"].tolist()
        x_idx = np.arange(len(conds))
        width = 0.2

        ndcg_vals = attrition_df["Graded_NDCG@10"].to_numpy()
        sp_vals_att = attrition_df["Spearman"].to_numpy()
        p10_vals = attrition_df["Precision@10"].to_numpy()
        mae_vals = attrition_df["Top10_MAE"].to_numpy()

        ax.bar(x_idx - 1.5 * width, ndcg_vals, width, label="Graded NDCG@10", color="#8e44ad", edgecolor="black")
        ax.bar(x_idx - 0.5 * width, sp_vals_att, width, label="Spearman ρ", color="#2980b9", edgecolor="black")
        ax.bar(x_idx + 0.5 * width, p10_vals, width, label="Precision@10", color="#27ae60", edgecolor="black")
        ax.bar(x_idx + 1.5 * width, mae_vals, width, label="Top-10 MAE", color="#e67e22", edgecolor="black")

        ax.set_xticks(x_idx)
        ax.set_xticklabels(conds)
        ax.set_ylim([0, 1.15])
        ax.set_ylabel("Metric Value")
        ax.set_title("Continuous Criticality Ranking Performance across Attrition Regimes")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p7 = plots_dir / "attrition_ranking_comparison.png"
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p7)

    # ----------------------------------------------------
    # Supporting Plot: Training Loss Curves
    # ----------------------------------------------------
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train Total Loss", lw=2, ms=3)
    axes[0, 0].plot(epochs, history["val_loss"], "r--s", label="Val Total Loss", lw=2, ms=3)
    axes[0, 0].set_title("Total Multi-Task Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot(epochs, history["train_regression"], "b-", label="Train Regression MSE", lw=2)
    axes[0, 1].plot(epochs, history["val_regression"], "r--", label="Val Regression MSE", lw=2)
    axes[0, 1].set_title("Current Criticality Regression Loss (MSE)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(epochs, history["train_ranking"], "m-o", label="Train Graded Ranking Loss", lw=2, ms=3)
    axes[1, 0].plot(epochs, history["val_ranking"], "c--s", label="Val Graded Ranking Loss", lw=2, ms=3)
    axes[1, 0].set_title("Graded Pairwise Ranking Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    axes[1, 1].plot(epochs, history["train_future"], "b-", label="Train Future MSE", lw=2)
    axes[1, 1].plot(epochs, history["val_future"], "r--", label="Val Future MSE", lw=2)
    axes[1, 1].set_title("Future Score Forecast Loss (MSE)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p_loss = plots_dir / "training_loss_curves.png"
    plt.savefig(p_loss, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p_loss)

    # ----------------------------------------------------
    # Supporting Plot: Ranking Loss and Val Graded NDCG@10 Curve
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5.5))
    plt.plot(epochs, history["train_ranking"], "m-o", label="Train Graded Ranking Loss", lw=2, ms=3)
    plt.plot(epochs, history["val_ranking"], "c--s", label="Val Graded Ranking Loss", lw=2, ms=3)
    if "val_graded_ndcg10" in history and any(not np.isnan(v) for v in history["val_graded_ndcg10"]):
        ax2 = plt.gca().twinx()
        ax2.plot(epochs, history["val_graded_ndcg10"], "g-^", label="Val Graded NDCG@10 (Selection Metric)", lw=2, ms=3)
        if "val_spearman" in history and any(not np.isnan(v) for v in history["val_spearman"]):
            ax2.plot(epochs, history["val_spearman"], "y--d", label="Val Spearman ρ (Diagnostic)", lw=1.8, ms=3)
        ax2.set_ylabel("Validation Ranking Metric", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim([0.0, 1.05])
    plt.title("Graded Ranking Loss and Validation NDCG@10 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Graded Pairwise Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    p_rank = plots_dir / "ranking_loss_curve.png"
    plt.savefig(p_rank, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plot_paths.append(p_rank)

    return saved_plot_paths


# ==========================================
# 8. Main Execution Pipeline
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

    # Adjust loss weights based on experiment profile
    if args.experiment == "baseline":
        args.lambda_rank = 0.0
        print("Experiment Profile: BASELINE (lambda_rank = 0.0, regression + classification only)")
    elif args.experiment == "exp1":
        print(f"Experiment Profile: EXP 1 (Balanced Multi-Task + Graded Ranking, lambda_rank = {args.lambda_rank})")
    elif args.experiment == "exp2":
        args.lambda_reg = 1.0
        args.lambda_rank = 2.0
        args.lambda_future = 0.1
        args.lambda_cls = 0.25
        print(f"Experiment Profile: EXP 2 (Ranking-Tuned: rank={args.lambda_rank}, reg={args.lambda_reg}, cls={args.lambda_cls})")

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
        print(f"Subsampled to {len(available_cases)} cases for verification.")

    if "attrition_rate" in df.columns:
        df.rename(columns={"attrition_rate": "scenario_attrition_regime"}, inplace=True)

    # Vectorized attrition computation
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
        counts[["case_id", "step", "initial_count", "attrition_fraction", "attrition_rate"]],
        on=["case_id", "step"], how="left"
    )
    context_features = env_features + ["attrition_fraction", "attrition_rate"]

    # Future score target alignment
    step_df = df[["case_id", "step"]].drop_duplicates()
    step_df["step_idx"] = step_df.groupby("case_id").cumcount()
    df = df.merge(step_df, on=["case_id", "step"], how="left")
    df["future_step_idx"] = df["step_idx"] + args.future_horizon

    lookup = df[["case_id", "step_idx", "agent_id", score_target]].rename(
        columns={"step_idx": "future_step_idx", score_target: "_future_score"}
    )
    df = df.merge(lookup, on=["case_id", "future_step_idx", "agent_id"], how="left")
    df.drop(columns=["step_idx", "future_step_idx"], inplace=True)

    # Scenario-level train/val/test split
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

    # Prebuild spatial snapshot graphs
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

    # Section 7: Validation Checkpoint Selection using Graded NDCG@10
    best_val_graded_ndcg = -float("inf")
    best_val_spearman = -float("inf")
    best_val_score_mse = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_regression": [], "val_regression": [],
        "train_ranking": [], "val_ranking": [],
        "train_future": [], "val_future": [],
        "train_classification": [], "val_classification": [],
        "val_graded_ndcg10": [], "val_spearman": [], "val_score_mse": [],
    }

    print("\n" + "=" * 75)
    print(f"Beginning Training with Validation Graded NDCG@10 Checkpoint Selection (Patience: {args.patience})")
    print("=" * 75)

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        train_loss, train_parts, _, _ = run_epoch(
            model, train_sequences, optimizer=optimizer, train=True,
            collect=False, eval_ranking=False,
            lambda_reg=args.lambda_reg, lambda_rank=args.lambda_rank,
            lambda_future=args.lambda_future, lambda_cls=args.lambda_cls,
            ranking_margin=args.ranking_margin, ranking_alpha=args.ranking_alpha,
            ranking_temperature=args.ranking_temperature, device=device
        )

        val_loss, val_parts, _, val_rank_dict = run_epoch(
            model, val_sequences, train=False,
            collect=False, eval_ranking=True,
            lambda_reg=args.lambda_reg, lambda_rank=args.lambda_rank,
            lambda_future=args.lambda_future, lambda_cls=args.lambda_cls,
            ranking_margin=args.ranking_margin, ranking_alpha=args.ranking_alpha,
            ranking_temperature=args.ranking_temperature, device=device
        )

        val_graded_ndcg = val_rank_dict["graded_ndcg10"]
        val_spearman = val_rank_dict["spearman"]
        val_score_mse = val_rank_dict["score_mse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_regression"].append(train_parts["regression"])
        history["val_regression"].append(val_parts["regression"])
        history["train_ranking"].append(train_parts["ranking"])
        history["val_ranking"].append(val_parts["ranking"])
        history["train_future"].append(train_parts["future"])
        history["val_future"].append(val_parts["future"])
        history["train_classification"].append(train_parts["classification"])
        history["val_classification"].append(val_parts["classification"])
        history["val_graded_ndcg10"].append(val_graded_ndcg)
        history["val_spearman"].append(val_spearman)
        history["val_score_mse"].append(val_score_mse)

        ep_dur = time.time() - ep_t0
        log_str = (f"Epoch {epoch:03d}/{args.epochs:03d} [{ep_dur:5.1f}s] | "
                   f"Train: {train_loss:.4f} (Reg={train_parts['regression']:.4f}, Rank={train_parts['ranking']:.4f}) | "
                   f"Val: {val_loss:.4f} (Rank={val_parts['ranking']:.4f}) | "
                   f"Val Graded NDCG@10: {val_graded_ndcg:.4f} | Spearman: {val_spearman:.4f}")

        # Checkpoint selection: Primary is Graded NDCG@10, secondary Spearman rho, tertiary score MSE
        improved = False
        if val_graded_ndcg > best_val_graded_ndcg + 1e-4:
            improved = True
        elif abs(val_graded_ndcg - best_val_graded_ndcg) <= 1e-4:
            if val_spearman > best_val_spearman + 1e-4:
                improved = True
            elif abs(val_spearman - best_val_spearman) <= 1e-4 and val_score_mse < best_val_score_mse:
                improved = True

        if improved:
            best_val_graded_ndcg = val_graded_ndcg
            best_val_spearman = val_spearman
            best_val_score_mse = val_score_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            print(log_str + "  [*] (Best Graded NDCG@10 Checkpoint)", flush=True)
        else:
            bad_epochs += 1
            print(log_str + f"  [Patience: {bad_epochs}/{args.patience}]", flush=True)
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (Best Graded NDCG@10: {best_val_graded_ndcg:.4f} at epoch {best_epoch}).", flush=True)
                break

    print(f"Training completed in {(time.time() - start_train_time)/60:.2f} minutes.")

    # Load best checkpoint
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    # Save model weights, ranking configuration, and preprocessing parameters
    artifact_path = output_dir / "swarm_criticality_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "node_features": node_features,
        "context_features": context_features,
        "position_features": position_cols,
        "radius": args.radius,
        "hidden_dim": args.hidden_dim,
        "context_dim": args.context_dim,
        "future_horizon": args.future_horizon,
        "ranking_config": {
            "primary_ranking_signal": "pred_criticality_score",
            "lambda_reg": args.lambda_reg,
            "lambda_rank": args.lambda_rank,
            "lambda_future": args.lambda_future,
            "lambda_cls": args.lambda_cls,
            "ranking_margin": args.ranking_margin,
            "ranking_alpha": args.ranking_alpha,
            "ranking_temperature": args.ranking_temperature,
            "best_epoch": best_epoch,
            "best_val_graded_ndcg": best_val_graded_ndcg,
            "best_val_spearman": best_val_spearman,
            "checkpoint_metric": "val_graded_ndcg_at_10",
        },
        "node_scaler_mean": node_scaler.mean_.tolist(),
        "node_scaler_scale": node_scaler.scale_.tolist(),
        "context_scaler_mean": context_scaler.mean_.tolist(),
        "context_scaler_scale": context_scaler.scale_.tolist(),
    }, artifact_path)
    print(f"Saved trained model checkpoint to: {artifact_path.resolve()}")

    # Save training history JSON
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path.resolve()}")

    # ==========================================
    # 9. Held-Out Test Evaluation
    # ==========================================
    print("\nRunning inference on held-out test scenarios...")
    test_pred = predict_sequences(model, test_sequences, device=device)

    preds_csv_path = output_dir / "swarm_criticality_test_predictions.csv"
    test_pred.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions ({len(test_pred):,} rows) to: {preds_csv_path.resolve()}")

    # Generate Top-10 candidates (ranked strictly by predicted continuous severity)
    top10_df = generate_top10_candidates(test_pred)
    top10_path = output_dir / "top10_ranked_candidates.csv"
    top10_df.to_csv(top10_path, index=False)
    print(f"Saved Top-10 candidates ({len(top10_df):,} rows) to: {top10_path.resolve()}")

    # Comprehensive snapshot-level ranking evaluation (Section 6)
    per_snapshot_df, agg_summary = evaluate_test_ranking(test_pred, k_list=(1, 3, 5, 8, 10))
    snapshot_csv_path = output_dir / "per_snapshot_ranking_metrics.csv"
    per_snapshot_df.to_csv(snapshot_csv_path, index=False)
    print(f"Saved per-snapshot ranking metrics ({len(per_snapshot_df):,} snapshots) to: {snapshot_csv_path.resolve()}")

    # Attrition-wise & Swarm-Size ranking evaluations
    attrition_df = evaluate_ranking_by_attrition(test_pred, k=10)
    attrition_path = output_dir / "ranking_metrics_by_attrition.csv"
    attrition_df.to_csv(attrition_path, index=False)
    print(f"Saved attrition-wise ranking metrics to: {attrition_path.resolve()}")

    swarm_size_df = evaluate_ranking_by_swarm_size(test_pred, k=10)
    swarm_size_path = output_dir / "ranking_metrics_by_swarm_size.csv"
    swarm_size_df.to_csv(swarm_size_path, index=False)
    print(f"Saved swarm-size ranking metrics to: {swarm_size_path.resolve()}")

    # Standard Classification and Regression Metrics
    y_true = test_pred[binary_target].to_numpy().astype(int)
    p = test_pred["prob_critical"].to_numpy()
    y_hat = (p >= 0.5).astype(int)

    cls_reg_metrics = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "PR-AUC (average precision)": float(average_precision_score(y_true, p)),
        "ROC-AUC": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else 0.0,
    }

    valid_score = np.isfinite(test_pred[score_target].to_numpy())
    cls_reg_metrics["score MAE"] = float(mean_absolute_error(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ))
    cls_reg_metrics["score RMSE"] = float(mean_squared_error(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ) ** 0.5)
    cls_reg_metrics["score R2"] = float(r2_score(
        test_pred.loc[valid_score, score_target],
        test_pred.loc[valid_score, "pred_criticality_score"]
    ))

    valid_future = np.isfinite(test_pred["_future_score"].to_numpy())
    if valid_future.any():
        cls_reg_metrics["future score MAE"] = float(mean_absolute_error(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ))
        cls_reg_metrics["future score RMSE"] = float(mean_squared_error(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ) ** 0.5)
        cls_reg_metrics["future score R2"] = float(r2_score(
            test_pred.loc[valid_future, "_future_score"],
            test_pred.loc[valid_future, "pred_future_score"]
        ))
    cls_reg_metrics["future_horizon"] = args.future_horizon

    # Print summary tables to stdout
    print("\n" + "=" * 75)
    print("PRIMARY CONTINUOUS CRITICALITY SEVERITY RANKING METRICS (HELD-OUT TEST SET)")
    print("=" * 75)
    print(f"Spearman Rank Correlation (ρ):        {agg_summary['spearman']['mean']:.4f} ± {agg_summary['spearman']['std']:.4f} (Median: {agg_summary['spearman']['median']:.4f})")
    print(f"Kendall's Tau (τ):                   {agg_summary['kendall']['mean']:.4f} ± {agg_summary['kendall']['std']:.4f} (Median: {agg_summary['kendall']['median']:.4f})")
    print(f"Graded NDCG@10:                      {agg_summary['graded_ndcg10']['mean']:.4f} ± {agg_summary['graded_ndcg10']['std']:.4f} (Median: {agg_summary['graded_ndcg10']['median']:.4f})")
    print(f"Graded NDCG@8:                       {agg_summary['graded_ndcg8']['mean']:.4f} ± {agg_summary['graded_ndcg8']['std']:.4f}")
    print(f"Graded NDCG@5:                       {agg_summary['graded_ndcg5']['mean']:.4f} ± {agg_summary['graded_ndcg5']['std']:.4f}")
    print(f"Graded NDCG@3:                       {agg_summary['graded_ndcg3']['mean']:.4f} ± {agg_summary['graded_ndcg3']['std']:.4f}")
    print(f"Graded NDCG@1:                       {agg_summary['graded_ndcg1']['mean']:.4f} ± {agg_summary['graded_ndcg1']['std']:.4f}")
    print(f"Top-10 Severity MAE:                 {agg_summary['top10_mae']['mean']:.4f} ± {agg_summary['top10_mae']['std']:.4f}")
    print(f"Top-10 Severity RMSE:                {agg_summary['top10_rmse']['mean']:.4f} ± {agg_summary['top10_rmse']['std']:.4f}")
    print(f"Mean Absolute Pairwise Error:        {agg_summary['pairwise_ordering_error']['mean']:.4f} ± {agg_summary['pairwise_ordering_error']['std']:.4f}")
    print(f"Secondary Precision@10:              {agg_summary['precision10']['mean']:.4f}")
    print(f"Secondary Recall@10:                 {agg_summary['recall10']['mean']:.4f}")

    print("\n" + "=" * 75)
    print("SUPPORTING CLASSIFICATION AND REGRESSION METRICS")
    print("=" * 75)
    metrics_s = pd.Series({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in cls_reg_metrics.items()}, name="Value")
    print(metrics_s.to_string())

    print("\n" + "=" * 75)
    print("ATTRITION-WISE CONTINUOUS RANKING PERFORMANCE")
    print("=" * 75)
    print(attrition_df.to_string(index=False))

    # Save metrics JSON
    metrics_out = {
        "continuous_ranking_summary": agg_summary,
        "classification_regression_metrics": cls_reg_metrics,
        "config": vars(args),
    }
    metrics_json_path = output_dir / "test_metrics_summary.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved metrics summary to: {metrics_json_path.resolve()}")

    # ==========================================
    # 10. Visualization Suite (Section 8)
    # ==========================================
    print("\nGenerating and saving all evaluation plots...")
    saved_plots = generate_all_plots(history, test_pred, top10_df, per_snapshot_df, agg_summary, attrition_df, plots_dir)
    print(f"Successfully generated and saved {len(saved_plots)} plots to {plots_dir.resolve()}:")
    for p in saved_plots:
        print(f"  - {p.name}")

    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()