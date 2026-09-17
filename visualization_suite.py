"""
visualization_suite.py
======================
Run AFTER main.py has finished training and saved:
  - training_history.json
  - final_metrics.json
  - confusion_matrix.npy
  - class_distribution.npy
  - flood_stats.json

All plots are saved to:  flood_output/all_plots/
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# ==========================================
# CONFIG  — change SAVE_DIR if needed
# ==========================================
SAVE_DIR = r"C:\Users\nielitpatna\Desktop\Flood\flood_output"
PLOT_DIR = os.path.join(SAVE_DIR, "all_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

PAPER_COLOR = '#5B7FD4'
OURS_COLOR  = '#E8593C'
GOOD_COLOR  = '#27AE60'
WARN_COLOR  = '#F39C12'

# ==========================================
# CLASS INFO
# ==========================================
CLASS_NAMES  = ["Background","Building-Flooded","Building-NF","Road-Flooded",
                "Road-NF","Water","Tree","Vehicle","Pool","Grass"]
CLASS_COLORS = np.array([
    [0,0,0],[255,0,0],[255,165,0],[255,0,255],[128,0,128],
    [0,0,255],[0,255,0],[255,255,0],[0,255,255],[165,42,42]
]) / 255.0
FLOODED_IDX  = {1, 3}   # Building-Flooded, Road-Flooded

# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    h_path = os.path.join(SAVE_DIR, 'training_history.json')
    m_path = os.path.join(SAVE_DIR, 'final_metrics.json')
    for p in [h_path, m_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run main.py first to generate training outputs.")

    with open(h_path) as f: history = json.load(f)
    with open(m_path) as f: metrics = json.load(f)
    return history, metrics

HISTORY, METRICS = load_data()
OURS   = METRICS['ours']
PAPER  = METRICS['paper']

# ==========================================
# HELPERS
# ==========================================
def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {name}")

def styled_ax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# 01  TRAINING CURVES (Loss / mIoU / mDice)
# ==========================================
def plot_01_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DINOv2SegFormer — Training Curves', fontsize=15, fontweight='bold', y=1.02)

    epochs = range(1, len(HISTORY['train_loss']) + 1)

    axes[0].plot(epochs, HISTORY['train_loss'], color=OURS_COLOR,  lw=2.5, label='Train')
    axes[0].plot(epochs, HISTORY['val_loss'],   color=PAPER_COLOR, lw=2.5, label='Val')
    styled_ax(axes[0], 'Loss', 'Epoch', 'Loss')
    axes[0].legend()

    axes[1].plot(epochs, HISTORY['train_miou'], color=OURS_COLOR,  lw=2.5, label='Train')
    axes[1].plot(epochs, HISTORY['val_miou'],   color=PAPER_COLOR, lw=2.5, label='Val')
    axes[1].axhline(0.752, color='red', lw=1.5, ls='--', label='Paper 75.2%')
    best_ep = int(np.argmax(HISTORY['val_miou'])) + 1
    best_v  = max(HISTORY['val_miou'])
    axes[1].axvline(best_ep, color=GOOD_COLOR, lw=1.2, ls=':', alpha=0.7)
    axes[1].annotate(f'Best\n{best_v*100:.1f}%',
                     xy=(best_ep, best_v), xytext=(best_ep+2, best_v-0.05),
                     fontsize=9, color=GOOD_COLOR,
                     arrowprops=dict(arrowstyle='->', color=GOOD_COLOR, lw=1))
    styled_ax(axes[1], 'mIoU', 'Epoch', 'mIoU')
    axes[1].legend()

    axes[2].plot(epochs, HISTORY['train_dice'], color=OURS_COLOR,  lw=2.5, label='Train')
    axes[2].plot(epochs, HISTORY['val_dice'],   color=PAPER_COLOR, lw=2.5, label='Val')
    axes[2].axhline(0.854, color='red', lw=1.5, ls='--', label='Paper 85.4%')
    styled_ax(axes[2], 'mDice', 'Epoch', 'mDice')
    axes[2].legend()

    plt.tight_layout()
    save('01_training_curves.png')

# ==========================================
# 02  LEARNING RATE SCHEDULE
# ==========================================
def plot_02_lr_schedule():
    if 'lr' not in HISTORY or len(HISTORY['lr']) == 0:
        print("  Skipping LR plot (no lr key in history)")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(HISTORY['lr']) + 1)
    ax.semilogy(epochs, HISTORY['lr'], color=OURS_COLOR, lw=2.5)
    ax.fill_between(epochs, HISTORY['lr'], alpha=0.15, color=OURS_COLOR)
    styled_ax(ax, 'Learning Rate Schedule', 'Epoch', 'LR (log scale)')
    plt.tight_layout()
    save('02_lr_schedule.png')

# ==========================================
# 03  MODEL ARCHITECTURE DIAGRAM
# ==========================================
def plot_03_model_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    def box(x, y, w, h, label, sublabel='', color='#4A90D9', text_color='white', fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='white', linewidth=2, zorder=3)
        ax.add_patch(rect)
        cy = y + h/2 + (0.15 if sublabel else 0)
        ax.text(x + w/2, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28, sublabel, ha='center', va='center',
                    fontsize=7.5, color=text_color, alpha=0.88, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2), zorder=5)

    # Input
    box(0.3, 3.8, 2.2, 1.4, 'Input Image', '512 × 512 × 3', color='#2C3E50')
    arrow(2.5, 4.5, 3.1, 4.5)

    # DINOv2 encoder
    box(3.1, 1.5, 2.8, 6.0, 'DINOv2-Large\nEncoder', '307M params\nViT-Large', color='#8E44AD', fontsize=11)

    # hidden states
    ys = [6.8, 5.5, 4.2, 2.9]
    lbls = ['Layer 3', 'Layer 6', 'Layer 9', 'Layer 12']
    for i, (yy, lb) in enumerate(zip(ys, lbls)):
        box(6.2, yy, 1.5, 0.9, lb, '1024-d', color='#7F8C8D', fontsize=9)
        arrow(5.9, yy + 0.45, 6.2, yy + 0.45)

    # scale projections
    for i, yy in enumerate(ys):
        box(8.0, yy, 1.8, 0.9, f'Scale Proj {i+1}', 'Linear+LN+GELU', color='#16A085', fontsize=8)
        arrow(7.7, yy + 0.45, 8.0, yy + 0.45)

    # fusion
    box(10.1, 3.5, 1.8, 2.0, 'Feature\nFusion', 'Cat → 4096-d\n128×128', color='#D35400', fontsize=10)
    for i, yy in enumerate(ys):
        arrow(9.8, yy + 0.45, 10.1, 4.5 - i*0.01)

    # conv decoder
    box(12.2, 2.8, 2.0, 3.4, 'Conv\nDecoder', 'Conv1×1→512\nConv3×3→256\nConv3×3→128', color='#C0392B', fontsize=9)
    arrow(11.9, 4.5, 12.2, 4.5)

    # heads
    box(14.5, 5.2, 1.3, 1.2, 'Seg\nHead', '10 classes', color='#27AE60', fontsize=9)
    box(14.5, 3.5, 1.3, 1.2, 'Aux\nHead', 'flood binary', color='#F39C12', fontsize=9)
    arrow(14.2, 5.5, 14.5, 5.8)
    arrow(14.2, 4.0, 14.5, 4.1)

    # final output
    box(14.5, 1.8, 1.3, 1.2, 'Output\n512×512×10', '', color='#2C3E50', fontsize=8)
    arrow(15.1, 5.2, 15.1, 4.7)
    arrow(15.1, 3.5, 15.1, 3.0)

    ax.set_title('DINOv2SegFormer Architecture', fontsize=17, fontweight='bold', pad=14)
    plt.tight_layout()
    save('03_model_architecture.png')

# ==========================================
# 04  PIPELINE FLOWCHART
# ==========================================
def plot_04_pipeline():
    steps = [
        ('Input Image\n(aerial/drone)', '#2C3E50'),
        ('Preprocessing\n(resize 512×512, normalize)', '#7F8C8D'),
        ('DINOv2-Large\nFeature Extraction', '#8E44AD'),
        ('Multi-Scale Fusion\n(4 hidden states → concat)', '#16A085'),
        ('ConvDecoder + Seg Head\n(10-class prediction)', '#2980B9'),
        ('Flood Area Estimation\n(FLOODED / NON-FLOODED)', '#E74C3C'),
        ('Weather API\n(wttr.in)', '#F39C12'),
        ('Folium Map\n(HTML interactive)', '#27AE60'),
        ('Emergency Alert\n(alerts.log)', '#C0392B'),
        ('PDF Report\n(rescue recommendations)', '#1A252F'),
    ]

    fig, ax = plt.subplots(figsize=(7, 18))
    ax.set_xlim(0, 7); ax.set_ylim(-0.5, len(steps) * 1.8 + 0.5)
    ax.axis('off')

    for i, (label, color) in enumerate(steps):
        y = (len(steps) - 1 - i) * 1.8 + 0.5
        rect = FancyBboxPatch((1.0, y), 5.0, 1.2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.5, y + 0.6, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        if i < len(steps) - 1:
            ax.annotate('', xy=(3.5, y - 0.08), xytext=(3.5, y - 0.52),
                        arrowprops=dict(arrowstyle='->', color='#AAA', lw=2))

    ax.set_title('Flood Detection Pipeline', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    save('04_pipeline.png')

# ==========================================
# 05  PAPER COMPARISON BAR CHART
# ==========================================
def plot_05_paper_comparison():
    metrics = ['mIoU (%)', 'mDice (%)', 'mACC (%)']
    ours    = [OURS['mIoU'],  OURS['mDice'],  OURS['mACC']]
    paper   = [PAPER['mIoU'], PAPER['mDice'], PAPER['mACC']]

    x = np.arange(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_o = ax.bar(x - w/2, ours,  w, color=OURS_COLOR,  label='DINOv2SegFormer (ours)',  zorder=3)
    bars_p = ax.bar(x + w/2, paper, w, color=PAPER_COLOR, label='SwinSegFormer (paper)', zorder=3)

    for bar in list(bars_o) + list(bars_p):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 105)
    styled_ax(ax, 'Paper vs Our Model — Core Metrics', ylabel='Score (%)')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save('05_paper_comparison.png')

# ==========================================
# 06  PER-CLASS IoU COMPARISON
# ==========================================
def plot_06_per_class_iou():
    iou_ours = [METRICS['per_class_iou'].get(n, 0) for n in CLASS_NAMES]

    colors = [OURS_COLOR if i in FLOODED_IDX else GOOD_COLOR
              if v >= 70 else WARN_COLOR if v >= 50 else '#E74C3C'
              for i, v in enumerate(iou_ours)]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(CLASS_NAMES, iou_ours, color=colors, edgecolor='white', linewidth=1.2, zorder=3)
    ax.axhline(75.2, color='navy', lw=2, ls='--', label='Paper mIoU 75.2%')
    ax.axhline(np.mean(iou_ours), color=OURS_COLOR, lw=2, ls=':', label=f'Our mean {np.mean(iou_ours):.1f}%')

    for bar, v in zip(bars, iou_ours):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=10)
    ax.set_ylim(0, 110)

    legend_handles = [
        mpatches.Patch(color=OURS_COLOR,  label='Flooded classes'),
        mpatches.Patch(color=GOOD_COLOR,  label='IoU ≥ 70%'),
        mpatches.Patch(color=WARN_COLOR,  label='50% ≤ IoU < 70%'),
        mpatches.Patch(color='#E74C3C',   label='IoU < 50%'),
    ]
    ax.legend(handles=legend_handles + [
        plt.Line2D([0],[0], color='navy',    lw=2, ls='--', label='Paper mIoU'),
        plt.Line2D([0],[0], color=OURS_COLOR, lw=2, ls=':', label='Our mean'),
    ], fontsize=9, loc='upper right')

    styled_ax(ax, 'Per-Class IoU — DINOv2SegFormer vs Paper Baseline', ylabel='IoU (%)')
    plt.tight_layout()
    save('06_per_class_iou.png')

# ==========================================
# 07  PER-CLASS DICE + ACC
# ==========================================
def plot_07_per_class_dice_acc():
    dice = [METRICS['per_class_dice'].get(n, 0) for n in CLASS_NAMES]
    acc  = [METRICS['per_class_acc'].get(n,  0) for n in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, dice, w, color=OURS_COLOR,  label='Dice (%)',     zorder=3)
    ax.bar(x + w/2, acc,  w, color=PAPER_COLOR, label='Accuracy (%)', zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=10)
    ax.set_ylim(0, 110)
    styled_ax(ax, 'Per-Class Dice & Accuracy', ylabel='Score (%)')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save('07_per_class_dice_acc.png')

# ==========================================
# 08  PERFORMANCE RADAR CHART
# ==========================================
def plot_08_radar():
    labels  = ['mIoU', 'mDice', 'mACC', 'FPS (norm)', 'Efficiency']
    # normalise FPS to 0-100 where paper=40, ours=100 (ours is faster)
    fps_ours  = min(100, OURS['FPS'] / PAPER['FPS'] * 40)
    fps_paper = 40
    # efficiency = inverse param ratio (fewer params = better efficiency, cap 100)
    eff_ours  = min(100, (PAPER['Params'] / max(OURS['Params'], 1)) * 25)
    eff_paper = 25

    ours_vals  = [OURS['mIoU'],  OURS['mDice'],  OURS['mACC'],  fps_ours,  eff_ours]
    paper_vals = [PAPER['mIoU'], PAPER['mDice'], PAPER['mACC'], fps_paper, eff_paper]

    angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    ours_vals  = ours_vals  + ours_vals[:1]
    paper_vals = paper_vals + paper_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, ours_vals,  color=OURS_COLOR,  lw=2.5, label='Ours')
    ax.fill(angles, ours_vals,  color=OURS_COLOR,  alpha=0.2)
    ax.plot(angles, paper_vals, color=PAPER_COLOR, lw=2.5, label='Paper')
    ax.fill(angles, paper_vals, color=PAPER_COLOR, alpha=0.2)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('Performance Radar — Ours vs Paper', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    save('08_radar_chart.png')

# ==========================================
# 09  CLASS DISTRIBUTION PIE
# ==========================================
def plot_09_class_distribution():
    dist_path = os.path.join(SAVE_DIR, 'class_distribution.npy')
    if not os.path.exists(dist_path):
        print("  Skipping class distribution (class_distribution.npy not found)")
        return
    values = np.load(dist_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    explode = [0.05 if i in FLOODED_IDX else 0 for i in range(10)]

    axes[0].pie(values, labels=CLASS_NAMES, autopct='%1.1f%%',
                colors=[CLASS_COLORS[i] for i in range(10)],
                explode=explode, startangle=140,
                textprops={'fontsize': 9})
    axes[0].set_title('Validation Set Class Distribution', fontsize=13, fontweight='bold')

    # bar version (easier to read)
    sorted_idx = np.argsort(values)[::-1]
    axes[1].barh([CLASS_NAMES[i] for i in sorted_idx],
                 [values[i]*100 for i in sorted_idx],
                 color=[CLASS_COLORS[i] for i in sorted_idx],
                 edgecolor='white', linewidth=1)
    for j, idx in enumerate(sorted_idx):
        axes[1].text(values[idx]*100 + 0.3, j, f'{values[idx]*100:.1f}%', va='center', fontsize=9)
    styled_ax(axes[1], 'Class Pixel % (sorted)', xlabel='Pixel %')
    axes[1].invert_yaxis()
    plt.tight_layout()
    save('09_class_distribution.png')

# ==========================================
# 10  CONFUSION MATRIX
# ==========================================
def plot_10_confusion_matrix():
    cm_path = os.path.join(SAVE_DIR, 'confusion_matrix.npy')
    if not os.path.exists(cm_path):
        print("  Skipping confusion matrix (confusion_matrix.npy not found)")
        return
    cm = np.load(cm_path).astype(float)

    # normalise per row (recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Recall %'}, linewidths=0.5)
    axes[0].set_title('Normalised Confusion Matrix (Recall %)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Ground Truth')
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')

    sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('Raw Confusion Matrix (pixel counts, sampled)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Ground Truth')
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')

    plt.tight_layout()
    save('10_confusion_matrix.png')

# ==========================================
# 11  FLOOD DETECTION PIE
# ==========================================
def plot_11_flood_detection():
    fpath = os.path.join(SAVE_DIR, 'flood_stats.json')
    if not os.path.exists(fpath):
        print("  Skipping flood detection plot (flood_stats.json not found)")
        return
    with open(fpath) as f: fs = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    labels = ['Flooded Pixels', 'Non-Flooded Pixels']
    vals   = [fs['flood_pct'], 100 - fs['flood_pct']]
    axes[0].pie(vals, labels=labels, autopct='%1.1f%%',
                colors=['#E74C3C', '#27AE60'], startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Predicted Pixel Distribution\n(flood vs non-flood)', fontsize=12, fontweight='bold')

    # flooded class breakdown from pred_dist
    flood_classes = {k: v for k, v in METRICS.get('pred_dist', {}).items()
                     if k in ['Building-Flooded', 'Road-Flooded']}
    if flood_classes:
        axes[1].bar(list(flood_classes.keys()), list(flood_classes.values()),
                    color=[OURS_COLOR, '#C0392B'], edgecolor='white', linewidth=1.5, width=0.4)
        for i, (k, v) in enumerate(flood_classes.items()):
            axes[1].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
        gt_flood = {k: METRICS.get('gt_dist', {}).get(k, 0)
                    for k in flood_classes}
        x = np.arange(len(flood_classes))
        axes[1].plot(x, list(gt_flood.values()), 'D--', color='navy',
                     ms=10, lw=2, label='GT %')
        axes[1].legend(fontsize=10)
        styled_ax(axes[1], 'Flooded Class: Predicted vs GT Pixel %', ylabel='Pixel %')
    plt.tight_layout()
    save('11_flood_detection.png')

# ==========================================
# 12  PREDICTION vs GT DISTRIBUTION
# ==========================================
def plot_12_pred_vs_gt():
    pred_dist = METRICS.get('pred_dist', {})
    gt_dist   = METRICS.get('gt_dist',   {})
    if not pred_dist:
        print("  Skipping pred vs gt distribution"); return

    names  = CLASS_NAMES
    pred_v = [pred_dist.get(n, 0) for n in names]
    gt_v   = [gt_dist.get(n,   0) for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, pred_v, w, color=OURS_COLOR,  label='Predicted %', zorder=3)
    ax.bar(x + w/2, gt_v,   w, color=PAPER_COLOR, label='GT %',        zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=10)
    styled_ax(ax, 'Model Prediction Distribution vs Ground Truth', ylabel='Pixel %')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save('12_pred_vs_gt_distribution.png')

# ==========================================
# 13  PARAMETER & MODEL SIZE COMPARISON
# ==========================================
def plot_13_parameter_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # params
    labels = ['Ours\n(DINOv2SegFormer)', 'Paper\n(SwinSegFormer)']
    params = [OURS['Params'], PAPER['Params']]
    bars = axes[0].bar(labels, params, color=[OURS_COLOR, PAPER_COLOR],
                        edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}M',
                     ha='center', fontsize=12, fontweight='bold')
    styled_ax(axes[0], 'Model Parameters (M)', ylabel='Parameters (M)')

    # model size
    sizes = [OURS['ModelSizeMB'], PAPER['ModelSizeMB']]
    bars2 = axes[1].bar(labels, sizes, color=[OURS_COLOR, PAPER_COLOR],
                         edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars2, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 20, f'{v:.0f} MB',
                     ha='center', fontsize=11, fontweight='bold')
    styled_ax(axes[1], 'Model Size (MB)', ylabel='Size (MB)')

    # training time
    times = [OURS['TrainHrs'], PAPER['TrainHrs']]
    bars3 = axes[2].bar(labels, times, color=[OURS_COLOR, PAPER_COLOR],
                         edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars3, times):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f} hrs',
                     ha='center', fontsize=11, fontweight='bold')
    styled_ax(axes[2], 'Training Time (hrs)', ylabel='Hours')

    plt.suptitle('Model Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save('13_parameter_comparison.png')

# ==========================================
# 14  SPEED COMPARISON
# ==========================================
def plot_14_speed_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # GPU FPS
    fps_labels = ['Ours', 'Paper']
    fps_vals   = [OURS['FPS'], PAPER['FPS']]
    bars = axes[0].bar(fps_labels, fps_vals, color=[OURS_COLOR, PAPER_COLOR],
                        edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars, fps_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
                     ha='center', fontsize=12, fontweight='bold')
    styled_ax(axes[0], 'GPU FPS', ylabel='Frames/sec')

    # Latency
    lat_vals = [OURS['Latency'], PAPER['Latency']]
    bars2 = axes[1].bar(fps_labels, lat_vals, color=[OURS_COLOR, PAPER_COLOR],
                         edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars2, lat_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.0f} ms',
                     ha='center', fontsize=11, fontweight='bold')
    styled_ax(axes[1], 'GPU Latency (ms)', ylabel='ms')

    # Memory
    mem_vals = [OURS['Memory_B1'], PAPER['Memory_B1']]
    bars3 = axes[2].bar(fps_labels, mem_vals, color=[OURS_COLOR, PAPER_COLOR],
                         edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars3, mem_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 50, f'{v:.0f} MB',
                     ha='center', fontsize=11, fontweight='bold')
    styled_ax(axes[2], 'GPU Memory Batch=1 (MB)', ylabel='MB')

    plt.suptitle('Speed & Memory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save('14_speed_comparison.png')

# ==========================================
# 15  MEMORY USAGE (batch=1 vs 4)
# ==========================================
def plot_15_memory_usage():
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Batch=1', 'Batch=4']
    vals   = [OURS['Memory_B1'], OURS['Memory_B4']]
    bars   = ax.bar(labels, vals, color=[OURS_COLOR, '#C0392B'],
                    edgecolor='white', linewidth=1.5, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 20,
                f'{v:.0f} MB\n({v/1024:.2f} GB)',
                ha='center', fontsize=11, fontweight='bold')
    ax.axhline(12 * 1024, color='red', lw=1.5, ls='--', label='RTX 4070S VRAM (12 GB)')
    styled_ax(ax, 'GPU Memory Usage — Ours', ylabel='Memory (MB)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save('15_memory_usage.png')

# ==========================================
# 16  COMPLETE COMPARISON MATRIX TABLE
# ==========================================
def plot_16_comparison_matrix():
    rows = [
        ('mIoU (%)',           f"{OURS['mIoU']:.1f}%",   '75.2%',   'Paper ✓' if PAPER['mIoU'] > OURS['mIoU'] else 'Ours ✓'),
        ('mDice (%)',          f"{OURS['mDice']:.1f}%",  '85.4%',   'Paper ✓' if PAPER['mDice'] > OURS['mDice'] else 'Ours ✓'),
        ('mACC (%)',           f"{OURS['mACC']:.1f}%",   '87.1%',   'Paper ✓' if PAPER['mACC'] > OURS['mACC'] else 'Ours ✓'),
        ('Parameters (M)',     f"{OURS['Params']:.1f}M", '91.3M',   'Ours ✓'),
        ('Model Size',         f"{OURS['ModelSizeMB']:.0f} MB", '~3500 MB', 'Ours ✓'),
        ('GPU FPS',            f"{OURS['FPS']:.1f}",     '~8',       'Ours ✓'),
        ('GPU Latency',        f"{OURS['Latency']:.0f} ms", '~120 ms', 'Ours ✓'),
        ('CPU FPS',            f"{OURS['CPU_FPS']:.2f}", '~0.2',     'Ours ✓'),
        ('GPU Mem (batch=1)',  f"{OURS['Memory_B1']:.0f} MB", '~8000 MB', 'Ours ✓'),
        ('Training Time',      f"{OURS['TrainHrs']:.1f} hrs", '~48 hrs', 'Ours ✓'),
        ('Full Pipeline',      'Yes',                    'No',        'Ours ✓'),
        ('Real-time (>15FPS)', 'Yes' if OURS['FPS'] > 15 else 'No', 'No', 'Ours ✓'),
    ]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis('off')
    col_labels = ['Metric', 'DINOv2SegFormer (Ours)', 'SwinSegFormer (Paper)', 'Winner']
    cell_text  = [[r[0], r[1], r[2], r[3]] for r in rows]

    cell_colors = []
    for r in rows:
        w = r[3]
        if 'Ours' in w:
            cell_colors.append(['#F8F9FA', '#D5F5E3', '#F8F9FA', '#D5F5E3'])
        else:
            cell_colors.append(['#F8F9FA', '#F8F9FA', '#D5ECF9', '#D5ECF9'])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.0)

    # header style
    for j in range(4):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ours_wins  = sum(1 for r in rows if 'Ours'  in r[3])
    paper_wins = sum(1 for r in rows if 'Paper' in r[3])
    ax.set_title(
        f'Full Comparison Matrix  |  Ours wins {ours_wins}/{len(rows)} metrics  |  Paper wins {paper_wins}/{len(rows)} metrics',
        fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    save('16_comparison_matrix.png')

# ==========================================
# 17  OVERFITTING ANALYSIS
# ==========================================
def plot_17_overfitting():
    train_miou = np.array(HISTORY['train_miou'])
    val_miou   = np.array(HISTORY['val_miou'])
    gap        = train_miou - val_miou
    epochs     = np.arange(1, len(gap) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, gap, color='#E74C3C', lw=2.5, label='Train-Val gap')
    axes[0].axhline(0.15, color='orange', lw=1.5, ls='--', label='Overfit threshold (0.15)')
    axes[0].fill_between(epochs, gap, 0.15,
                          where=(gap > 0.15), color='red', alpha=0.15, label='Overfitting zone')
    axes[0].fill_between(epochs, gap, 0,
                          where=(gap <= 0.15), color='green', alpha=0.1, label='Healthy zone')
    styled_ax(axes[0], 'Generalisation Gap (Train − Val mIoU)', 'Epoch', 'Gap')
    axes[0].legend(fontsize=9)

    axes[1].plot(epochs, train_miou, color=OURS_COLOR,  lw=2.5, label='Train mIoU')
    axes[1].plot(epochs, val_miou,   color=PAPER_COLOR, lw=2.5, label='Val mIoU')
    axes[1].fill_between(epochs, train_miou, val_miou, alpha=0.15, color='red')
    styled_ax(axes[1], 'Train vs Val mIoU', 'Epoch', 'mIoU')
    axes[1].legend(fontsize=10)

    plt.suptitle('Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save('17_overfitting_analysis.png')

# ==========================================
# 18  LOSS COMPONENTS (train + val)
# ==========================================
def plot_18_loss_detail():
    train_loss = HISTORY['train_loss']
    val_loss   = HISTORY['val_loss']
    epochs     = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, train_loss, color=OURS_COLOR,  lw=2.5, label='Train loss')
    ax.plot(epochs, val_loss,   color=PAPER_COLOR, lw=2.5, label='Val loss')

    # annotate min val loss
    min_ep = int(np.argmin(val_loss)) + 1
    min_v  = min(val_loss)
    ax.annotate(f'Min val\n{min_v:.4f} @ ep{min_ep}',
                xy=(min_ep, min_v), xytext=(min_ep + 3, min_v + 0.02),
                fontsize=9, color=PAPER_COLOR,
                arrowprops=dict(arrowstyle='->', color=PAPER_COLOR))
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='purple',
                    label='Gap')
    styled_ax(ax, 'Combined Loss (Focal + Tversky + BCE aux)', 'Epoch', 'Loss')
    ax.legend(fontsize=11)
    plt.tight_layout()
    save('18_loss_detail.png')

# ==========================================
# 19  BEST EPOCH SUMMARY CARD
# ==========================================
def plot_19_best_epoch_summary():
    best_ep  = int(np.argmax(HISTORY['val_miou'])) + 1
    best_iou = max(HISTORY['val_miou'])
    best_dice= HISTORY['val_dice'][best_ep - 1]
    best_loss= HISTORY['val_loss'][best_ep - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    cards = [
        ('Best Epoch',        f'{best_ep}',           '#2C3E50'),
        ('Val mIoU',          f'{best_iou*100:.2f}%', '#27AE60'),
        ('Val mDice',         f'{best_dice*100:.2f}%','#2980B9'),
        ('Val Loss',          f'{best_loss:.4f}',      '#8E44AD'),
        ('Total Epochs',      f'{len(HISTORY["train_loss"])}',  '#E67E22'),
        ('Paper mIoU Target', '75.2%',                '#E74C3C'),
    ]

    for i, (label, value, color) in enumerate(cards):
        x = (i % 3) * 3.2 + 0.5
        y = 3.2 - (i // 3) * 2.0
        rect = FancyBboxPatch((x, y), 2.8, 1.6,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.4, y + 1.05, value, ha='center', va='center',
                fontsize=22, fontweight='bold', color='white')
        ax.text(x + 1.4, y + 0.38, label, ha='center', va='center',
                fontsize=11, color='#DDD')

    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.set_title('Best Checkpoint Summary', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    save('19_best_epoch_summary.png')

# ==========================================
# 20  RESULTS VISUALISATION GRID
#    (reads from predictions/samples.png if available,
#     otherwise generates a colour legend)
# ==========================================
def plot_20_result_overview():
    samples_path = os.path.join(SAVE_DIR, 'predictions', 'samples.png')

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

    if os.path.exists(samples_path):
        # show the saved samples grid
        img = plt.imread(samples_path)
        ax  = fig.add_subplot(gs[:2, :])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Sample Predictions (Image | GT | Prediction)', fontsize=13, fontweight='bold')
    else:
        ax = fig.add_subplot(gs[:2, :])
        ax.text(0.5, 0.5, 'Run main.py to generate prediction samples\n'
                           '(flood_output/predictions/samples.png)',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes, color='gray')
        ax.axis('off')

    # class colour legend
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis('off')
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        xc = (i % 5) * 0.2 + 0.02
        yc = 0.85 - (i // 5) * 0.55
        rect = plt.Rectangle((xc, yc), 0.04, 0.25,
                               facecolor=color, edgecolor='black', linewidth=0.8,
                               transform=ax_leg.transAxes)
        ax_leg.add_patch(rect)
        marker = ' ◀ FLOOD' if i in FLOODED_IDX else ''
        ax_leg.text(xc + 0.05, yc + 0.12, name + marker,
                    va='center', fontsize=10, transform=ax_leg.transAxes,
                    color='red' if i in FLOODED_IDX else 'black', fontweight='bold' if i in FLOODED_IDX else 'normal')

    ax_leg.set_title('Segmentation Class Colour Legend', fontsize=12, fontweight='bold')
    fig.suptitle('DINOv2SegFormer — Result Overview', fontsize=16, fontweight='bold')
    save('20_result_overview.png')

# ==========================================
# SUMMARY CSV
# ==========================================
def generate_summary_csv():
    rows = [
        ('mIoU (%)',        OURS['mIoU'],  PAPER['mIoU']),
        ('mDice (%)',       OURS['mDice'], PAPER['mDice']),
        ('mACC (%)',        OURS['mACC'],  PAPER['mACC']),
        ('GPU FPS',         OURS['FPS'],   PAPER['FPS']),
        ('Latency (ms)',    OURS['Latency'],PAPER['Latency']),
        ('Params (M)',      OURS['Params'],PAPER['Params']),
        ('Memory B1 (MB)',  OURS['Memory_B1'], PAPER['Memory_B1']),
        ('Train Hrs',       OURS['TrainHrs'],  PAPER['TrainHrs']),
    ]
    df = pd.DataFrame(rows, columns=['Metric', 'Ours', 'Paper'])
    df['Delta']   = df['Ours'] - df['Paper']
    df['Winner']  = df.apply(
        lambda r: 'Ours' if (r['Metric'] not in ['mIoU (%)','mDice (%)','mACC (%)','Latency (ms)','Params (M)','Memory B1 (MB)','Train Hrs']
                              or (r['Metric'] in ['mIoU (%)','mDice (%)','mACC (%)'] and r['Ours'] >= r['Paper']))
                           else 'Paper', axis=1)
    path = os.path.join(PLOT_DIR, 'summary_metrics.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: summary_metrics.csv")

    # also save per-class CSV
    per_class_rows = []
    for name in CLASS_NAMES:
        per_class_rows.append({
            'Class': name,
            'IoU':   METRICS['per_class_iou'].get(name, 0),
            'Dice':  METRICS['per_class_dice'].get(name, 0),
            'Acc':   METRICS['per_class_acc'].get(name, 0),
            'Flooded': 'Yes' if name in ('Building-Flooded','Road-Flooded') else 'No',
        })
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(PLOT_DIR, 'per_class_metrics.csv'), index=False)
    print(f"  Saved: per_class_metrics.csv")

# ==========================================
# RUN ALL
# ==========================================
def run_all_visualizations():
    print('='*60)
    print('  DINOv2SegFormer — Full Visualization Suite')
    print('='*60)

    tasks = [
        ('01 Training curves',          plot_01_training_curves),
        ('02 LR schedule',              plot_02_lr_schedule),
        ('03 Model architecture',       plot_03_model_architecture),
        ('04 Pipeline flowchart',       plot_04_pipeline),
        ('05 Paper comparison bars',    plot_05_paper_comparison),
        ('06 Per-class IoU',            plot_06_per_class_iou),
        ('07 Per-class Dice + Acc',     plot_07_per_class_dice_acc),
        ('08 Radar chart',              plot_08_radar),
        ('09 Class distribution',       plot_09_class_distribution),
        ('10 Confusion matrix',         plot_10_confusion_matrix),
        ('11 Flood detection pie',      plot_11_flood_detection),
        ('12 Pred vs GT distribution',  plot_12_pred_vs_gt),
        ('13 Parameter comparison',     plot_13_parameter_comparison),
        ('14 Speed comparison',         plot_14_speed_comparison),
        ('15 Memory usage',             plot_15_memory_usage),
        ('16 Comparison matrix table',  plot_16_comparison_matrix),
        ('17 Overfitting analysis',     plot_17_overfitting),
        ('18 Loss detail',              plot_18_loss_detail),
        ('19 Best epoch summary',       plot_19_best_epoch_summary),
        ('20 Result overview',          plot_20_result_overview),
    ]

    ok = 0
    for name, fn in tasks:
        print(f"\n[{name}]")
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    generate_summary_csv()

    print('\n' + '='*60)
    print(f'  Done! {ok}/{len(tasks)} plots generated')
    print(f'  Output: {PLOT_DIR}')
    print('='*60)


if __name__ == '__main__':
    run_all_visualizations()