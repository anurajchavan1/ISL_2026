"""
generate_metrics.py

Generates a full set of accuracy and performance graphs for both
the dynamic model (TCN / Bi-LSTM) and the static model (Dense NN).

Outputs saved to metrics_output/ folder:
  1_training_curves.png         — loss + accuracy over epochs (dynamic)
  2_confusion_matrix.png        — row-normalised confusion heatmap (dynamic)
  3_per_class_accuracy.png      — per-class bar chart, worst-first (dynamic)
  4_accuracy_distribution.png   — histogram of class accuracy ranges (dynamic)
  5_top_confusions.png          — horizontal bar chart of worst confused pairs
  6_confidence_distribution.png — softmax confidence histogram on test set
  7_model_comparison.png        — side-by-side dynamic vs static key metrics
  8_static_confusion.png        — confusion matrix for static model
  9_static_per_class.png        — per-class F1 for static model

Usage:
  python generate_metrics.py

Requirements:
  pip install matplotlib seaborn scikit-learn tensorflow numpy
"""

import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
# Dynamic model
DYNAMIC_MODEL   = 'isl_tcn_model.h5'       # or isl_bilstm_model.h5
DYNAMIC_CLASSES = 'classes.npy'
DATA_PATH       = os.path.join(os.getcwd(), 'extracted_data')
SEQUENCE_LENGTH = 30
FEATURES        = 720

# Static model
STATIC_MODEL    = 'isl_landmark_model.keras'
STATIC_ENCODER  = 'label_encoder.pickle'
STATIC_DATA     = 'data.pickle'

# Training history (optional — only used for graph 1)
# Set to None if you do not have a saved history file
HISTORY_FILE    = 'training_history.npy'

OUTPUT_DIR      = 'metrics_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'figure.dpi'       : 150,
})

COL_GREEN  = '#639922'
COL_AMBER  = '#EF9F27'
COL_RED    = '#E24B4A'
COL_BLUE   = '#378ADD'
COL_PURPLE = '#7F77DD'
COL_TEAL   = '#1D9E75'
COL_GRAY   = '#888780'


def bar_color(acc):
    if acc >= 0.80: return COL_GREEN
    if acc >= 0.50: return COL_AMBER
    return COL_RED


def save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ==========================================
# LOAD DYNAMIC MODEL + DATA
# ==========================================
print("Loading dynamic model...")
dyn_model   = tf.keras.models.load_model(DYNAMIC_MODEL)
DYN_ACTIONS = np.load(DYNAMIC_CLASSES)

print("Loading dynamic data...")
sequences, labels = [], []
label_map = {l: i for i, l in enumerate(DYN_ACTIONS)}

for action in DYN_ACTIONS:
    p = os.path.join(DATA_PATH, action)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        if not f.endswith('.npy'): continue
        seq = np.load(os.path.join(p, f))
        if seq.shape == (SEQUENCE_LENGTH, FEATURES):
            sequences.append(seq)
            labels.append(label_map[action])

X = np.array(sequences)
y = np.array(labels)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"  Test set: {len(X_test)} samples, {len(DYN_ACTIONS)} classes")

print("Running dynamic model inference...")
dyn_probs  = dyn_model.predict(X_test, batch_size=32, verbose=0)
dyn_pred   = np.argmax(dyn_probs, axis=1)
dyn_true   = y_test
dyn_conf   = np.max(dyn_probs, axis=1)   # per-sample max confidence

overall_acc = np.mean(dyn_pred == dyn_true)
print(f"  Dynamic overall accuracy: {overall_acc:.1%}")

# Per-class accuracy
per_cls_correct = np.zeros(len(DYN_ACTIONS))
per_cls_total   = np.zeros(len(DYN_ACTIONS))
for t, p in zip(dyn_true, dyn_pred):
    per_cls_total[t]   += 1
    if t == p: per_cls_correct[t] += 1
per_cls_acc = np.where(per_cls_total > 0,
                        per_cls_correct / per_cls_total, 0.0)

sort_idx    = np.argsort(per_cls_acc)
sorted_acc  = per_cls_acc[sort_idx]
sorted_lbl  = DYN_ACTIONS[sort_idx]


# ==========================================
# GRAPH 1 — TRAINING CURVES
# ==========================================
print("\nGenerating graph 1: training curves...")

if os.path.exists(HISTORY_FILE):
    hist = np.load(HISTORY_FILE, allow_pickle=True).item()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training history — dynamic model', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(hist['accuracy'],     label='Train accuracy', color=COL_TEAL,   lw=2)
    ax.plot(hist['val_accuracy'], label='Val accuracy',   color=COL_PURPLE, lw=2)
    ax.axhline(overall_acc, color=COL_GRAY, ls='--', lw=1,
               label=f'Test accuracy ({overall_acc:.1%})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over epochs')
    ax.legend(); ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(hist['loss'],     label='Train loss', color=COL_AMBER, lw=2)
    ax.plot(hist['val_loss'], label='Val loss',   color=COL_RED,   lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss over epochs')
    ax.legend()

    save('1_training_curves.png')
else:
    print("  training_history.npy not found — skipping graph 1.")
    print("  To generate it, add this to your training script AFTER model.fit():")
    print("    np.save('training_history.npy', history.history)")


# ==========================================
# GRAPH 2 — CONFUSION MATRIX (dynamic)
# ==========================================
print("Generating graph 2: confusion matrix...")

cm        = confusion_matrix(dyn_true, dyn_pred, labels=range(len(DYN_ACTIONS)))
cm_errors = cm.copy(); np.fill_diagonal(cm_errors, 0)

# Show only classes involved in errors
active = np.unique(np.concatenate(np.where(cm_errors > 0)))
n_act  = len(active)

if n_act > 0:
    cm_sub  = cm[np.ix_(active, active)]
    lbl_sub = DYN_ACTIONS[active]
    row_sum = cm_sub.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm_sub / row_sum

    fs = max(10, 5)
    fig_sz = max(10, n_act * 0.52)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    sns.heatmap(cm_norm, ax=ax, cmap='Blues', vmin=0, vmax=1,
                annot=(n_act <= 40), fmt='.0%' if n_act <= 40 else '',
                xticklabels=lbl_sub, yticklabels=lbl_sub,
                linewidths=0.3, linecolor='#e0e0e0',
                cbar_kws={'label': 'Recall (row-normalised)'})
    ax.set_title(f'Confusion matrix — {n_act} classes with errors\n'
                 f'Dynamic model · overall accuracy {overall_acc:.1%}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=max(6, 10-n_act//12))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=max(6, 10-n_act//12))
    save('2_confusion_matrix.png')


# ==========================================
# GRAPH 3 — PER-CLASS ACCURACY BAR CHART
# ==========================================
print("Generating graph 3: per-class accuracy...")

n_show   = min(60, len(DYN_ACTIONS))
show_acc = sorted_acc[:n_show]
show_lbl = sorted_lbl[:n_show]
colors   = [bar_color(a) for a in show_acc]

fig, ax = plt.subplots(figsize=(max(14, n_show * 0.38), 6))
bars = ax.bar(range(n_show), show_acc * 100, color=colors, width=0.75, zorder=3)
ax.axhline(overall_acc * 100, color=COL_BLUE, lw=1.5, ls='--',
           label=f'Overall avg  {overall_acc:.1%}', zorder=4)
ax.axhline(80, color=COL_GRAY, lw=0.8, ls=':', alpha=0.6, label='80% threshold')
ax.axhline(50, color=COL_GRAY, lw=0.8, ls=':', alpha=0.4, label='50% threshold')

ax.set_xticks(range(n_show))
ax.set_xticklabels(show_lbl, rotation=90, fontsize=7)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 108)
ax.set_title(f'Per-class accuracy — worst {n_show} classes  '
             f'(green ≥ 80%, amber ≥ 50%, red < 50%)',
             fontsize=12, fontweight='bold')

# Patch legend
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor=COL_GREEN, label=f'≥ 80%  ({np.sum(per_cls_acc>=0.80)} classes)'),
    Patch(facecolor=COL_AMBER, label=f'50–79%  ({np.sum((per_cls_acc>=0.50)&(per_cls_acc<0.80))} classes)'),
    Patch(facecolor=COL_RED,   label=f'< 50%  ({np.sum(per_cls_acc<0.50)} classes)'),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color=COL_BLUE, ls='--', label=f'Overall avg {overall_acc:.1%}')
], fontsize=9)

save('3_per_class_accuracy.png')


# ==========================================
# GRAPH 4 — ACCURACY DISTRIBUTION HISTOGRAM
# ==========================================
print("Generating graph 4: accuracy distribution...")

bins   = np.arange(0, 110, 10)
counts = [np.sum((per_cls_acc*100 >= bins[i]) &
                  (per_cls_acc*100 < bins[i+1])) for i in range(len(bins)-1)]
b_colors = [bar_color(b/100) for b in bins[:-1]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(bins[:-1], counts, width=9, color=b_colors,
              align='edge', zorder=3, edgecolor='white')

for bar, cnt in zip(bars, counts):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.3,
                str(cnt), ha='center', va='bottom', fontsize=9)

ax.set_xticks(bins)
ax.set_xticklabels([f'{b}%' for b in bins], fontsize=9)
ax.set_xlabel('Accuracy range'); ax.set_ylabel('Number of classes')
ax.set_title(f'Accuracy distribution across {len(DYN_ACTIONS)} classes — '
             f'dynamic model (overall {overall_acc:.1%})',
             fontsize=12, fontweight='bold')
save('4_accuracy_distribution.png')


# ==========================================
# GRAPH 5 — TOP CONFUSED PAIRS
# ==========================================
print("Generating graph 5: top confusion pairs...")

cm_err = cm.copy(); np.fill_diagonal(cm_err, 0)
flat   = cm_err.flatten()
n_top  = min(20, np.sum(flat > 0))
top    = np.argsort(flat)[::-1][:n_top]
rows, cols = np.unravel_index(top, cm_err.shape)

pair_labels = [f"{DYN_ACTIONS[r]}  →  {DYN_ACTIONS[c]}" for r, c in zip(rows, cols)]
pair_counts = [cm_err[r, c] for r, c in zip(rows, cols)]

# Filter zeros
pair_labels = [l for l, c in zip(pair_labels, pair_counts) if c > 0]
pair_counts = [c for c in pair_counts if c > 0]

if pair_labels:
    fig, ax = plt.subplots(figsize=(10, max(5, len(pair_labels) * 0.45)))
    y_pos   = range(len(pair_labels))
    colors2 = [COL_RED if c >= 2 else COL_AMBER for c in pair_counts]
    ax.barh(y_pos, pair_counts, color=colors2, height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of misclassifications')
    ax.set_title(f'Top {len(pair_labels)} most confused sign pairs — dynamic model',
                 fontsize=12, fontweight='bold')
    ax.axvline(2, color=COL_GRAY, ls='--', lw=0.8, alpha=0.6)
    save('5_top_confusions.png')


# ==========================================
# GRAPH 6 — CONFIDENCE DISTRIBUTION
# ==========================================
print("Generating graph 6: confidence distribution...")

correct_conf   = dyn_conf[dyn_pred == dyn_true]
incorrect_conf = dyn_conf[dyn_pred != dyn_true]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(correct_conf,   bins=30, alpha=0.7, color=COL_GREEN,  label='Correct predictions',   density=True, zorder=3)
ax.hist(incorrect_conf, bins=30, alpha=0.7, color=COL_RED,    label='Incorrect predictions', density=True, zorder=3)
ax.axvline(0.60, color=COL_BLUE,  ls='--', lw=1.5, label='Dynamic threshold (0.60)')
ax.axvline(0.85, color=COL_AMBER, ls='--', lw=1.5, label='Static threshold (0.85)')
ax.set_xlabel('Softmax confidence score')
ax.set_ylabel('Density')
ax.set_title('Confidence distribution — correct vs incorrect predictions',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
save('6_confidence_distribution.png')


# ==========================================
# LOAD STATIC MODEL + DATA
# ==========================================
print("\nLoading static model...")
static_loaded = False

if (os.path.exists(STATIC_MODEL) and
        os.path.exists(STATIC_ENCODER) and
        os.path.exists(STATIC_DATA)):

    stat_model = tf.keras.models.load_model(STATIC_MODEL)
    with open(STATIC_ENCODER, 'rb') as f:
        le = pickle.load(f)
    STAT_CLASSES = list(le.classes_)

    data_dict = pickle.load(open(STATIC_DATA, 'rb'))
    Xs = np.array(data_dict['data'])
    ys = np.array(data_dict['labels'])

    from sklearn.model_selection import train_test_split as tts
    _, Xs_test, _, ys_test = tts(
        Xs, ys, test_size=0.20, random_state=42, stratify=ys
    )

    stat_probs = stat_model.predict(Xs_test, batch_size=32, verbose=0)
    stat_pred  = le.inverse_transform(np.argmax(stat_probs, axis=1))
    stat_acc   = np.mean(stat_pred == ys_test)
    print(f"  Static overall accuracy: {stat_acc:.1%}")
    static_loaded = True
else:
    print("  Static model / data not found — skipping static graphs.")
    print(f"  Expected: {STATIC_MODEL}, {STATIC_ENCODER}, {STATIC_DATA}")
    stat_acc = 0.0


# ==========================================
# GRAPH 7 — MODEL COMPARISON DASHBOARD
# ==========================================
print("Generating graph 7: model comparison...")

fig = plt.figure(figsize=(14, 6))
fig.suptitle('Dynamic vs static model — performance comparison',
             fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# Panel A — Overall accuracy
ax1 = fig.add_subplot(gs[0])
models  = ['Dynamic\n(TCN/Bi-LSTM)', 'Static\n(Dense NN)']
accs    = [overall_acc, stat_acc if static_loaded else 0]
col_acc = [COL_TEAL, COL_CORAL if static_loaded else COL_GRAY]
bars    = ax1.bar(models, [a*100 for a in accs], color=col_acc,
                  width=0.5, zorder=3)
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 110)
ax1.set_title('Overall test accuracy')

# Panel B — Class coverage
ax2 = fig.add_subplot(gs[1])
dyn_green  = np.sum(per_cls_acc >= 0.80)
dyn_amber  = np.sum((per_cls_acc >= 0.50) & (per_cls_acc < 0.80))
dyn_red    = np.sum(per_cls_acc < 0.50)
n_dyn      = len(DYN_ACTIONS)
n_stat     = len(STAT_CLASSES) if static_loaded else 0

x      = np.array([0, 1])
width  = 0.25
bars_g = ax2.bar(x[0],   dyn_green/n_dyn*100,  width, color=COL_GREEN,  label='≥ 80%',   zorder=3)
bars_a = ax2.bar(x[0]+width, dyn_amber/n_dyn*100, width, color=COL_AMBER, label='50-79%', zorder=3)
bars_r = ax2.bar(x[0]+2*width, dyn_red/n_dyn*100, width, color=COL_RED,  label='< 50%',  zorder=3)

if static_loaded:
    report  = classification_report(ys_test, stat_pred,
                                    target_names=STAT_CLASSES, output_dict=True)
    f1s     = [report[c]['f1-score'] for c in STAT_CLASSES if c in report]
    sg      = sum(1 for f in f1s if f >= 0.80)/len(f1s)*100
    sa      = sum(1 for f in f1s if 0.50 <= f < 0.80)/len(f1s)*100
    sr      = sum(1 for f in f1s if f < 0.50)/len(f1s)*100
    ax2.bar(x[1],         sg, width, color=COL_GREEN,  zorder=3)
    ax2.bar(x[1]+width,   sa, width, color=COL_AMBER,  zorder=3)
    ax2.bar(x[1]+2*width, sr, width, color=COL_RED,    zorder=3)

ax2.set_xticks(x + width)
ax2.set_xticklabels(['Dynamic', 'Static'])
ax2.set_ylabel('% of classes')
ax2.set_ylim(0, 110)
ax2.set_title('Class accuracy distribution')
ax2.legend(fontsize=8)

# Panel C — Key metrics table
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
rows  = ['Overall accuracy', 'Classes ≥ 80%', 'Classes < 50%',
         'Total classes', 'Feature dims', 'Model type']
dyn_v = [f'{overall_acc:.1%}',
          f'{dyn_green} / {n_dyn}',
          f'{dyn_red}',
          str(n_dyn), '720', 'TCN / Bi-LSTM']
sta_v = ([f'{stat_acc:.1%}',
           f'{sum(1 for f in f1s if f>=0.80)} / {n_stat}',
           f'{sum(1 for f in f1s if f<0.50)}',
           str(n_stat), '84', 'Dense NN']
          if static_loaded else ['N/A']*6)

table = ax3.table(
    cellText  = [[r, d, s] for r, d, s in zip(rows, dyn_v, sta_v)],
    colLabels = ['Metric', 'Dynamic', 'Static'],
    loc       = 'center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#E6F1FB')
        cell.set_text_props(fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F8F8F8')
    cell.set_edgecolor('#D0D0D0')
ax3.set_title('Summary metrics', pad=12)

save('7_model_comparison.png')


# ==========================================
# GRAPH 8 — STATIC MODEL CONFUSION MATRIX
# ==========================================
if static_loaded:
    print("Generating graph 8: static model confusion matrix...")
    stat_true_idx = le.transform(ys_test)
    stat_pred_idx = le.transform(stat_pred)
    cm_s          = confusion_matrix(stat_true_idx, stat_pred_idx)
    row_s         = cm_s.sum(axis=1, keepdims=True)
    row_s[row_s==0] = 1
    cm_s_norm     = cm_s / row_s

    n_s   = len(STAT_CLASSES)
    fs_s  = max(5, 10 - n_s//8)
    fig_s = max(8, n_s * 0.45)
    fig, ax = plt.subplots(figsize=(fig_s, fig_s))
    sns.heatmap(cm_s_norm, ax=ax, cmap='Purples', vmin=0, vmax=1,
                annot=(n_s <= 30), fmt='.0%' if n_s <= 30 else '',
                xticklabels=STAT_CLASSES, yticklabels=STAT_CLASSES,
                linewidths=0.3, linecolor='#e0e0e0',
                cbar_kws={'label': 'Recall (row-normalised)'})
    ax.set_title(f'Confusion matrix — static model ({n_s} classes)\n'
                 f'Overall accuracy {stat_acc:.1%}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fs_s)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=fs_s)
    save('8_static_confusion.png')


# ==========================================
# GRAPH 9 — STATIC PER-CLASS F1
# ==========================================
if static_loaded:
    print("Generating graph 9: static per-class F1...")
    report   = classification_report(ys_test, stat_pred,
                                     target_names=STAT_CLASSES, output_dict=True)
    f1_vals  = np.array([report[c]['f1-score'] for c in STAT_CLASSES if c in report])
    f1_lbls  = [c for c in STAT_CLASSES if c in report]

    sort_f   = np.argsort(f1_vals)
    f1_s     = f1_vals[sort_f]
    lbl_s    = np.array(f1_lbls)[sort_f]
    col_f    = [bar_color(f) for f in f1_s]

    fig, ax = plt.subplots(figsize=(max(10, len(f1_s)*0.38), 5))
    ax.bar(range(len(f1_s)), f1_s * 100, color=col_f, width=0.75, zorder=3)
    ax.axhline(stat_acc * 100, color=COL_PURPLE, ls='--', lw=1.5,
               label=f'Overall accuracy {stat_acc:.1%}')
    ax.axhline(80, color=COL_GRAY, lw=0.8, ls=':', alpha=0.6)
    ax.set_xticks(range(len(f1_s)))
    ax.set_xticklabels(lbl_s, rotation=90, fontsize=8)
    ax.set_ylabel('F1 Score (%)')
    ax.set_ylim(0, 108)
    ax.set_title('Per-class F1 score — static model (sorted worst-first)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    save('9_static_per_class.png')


# ==========================================
# SUMMARY
# ==========================================
print(f"""
╔══════════════════════════════════════════════════════════╗
  Metrics generation complete
  Output folder : {OUTPUT_DIR}/
  ─────────────────────────────────────────────────────
  Dynamic model : {DYNAMIC_MODEL}
    Overall accuracy  : {overall_acc:.1%}
    Classes ≥ 80%     : {np.sum(per_cls_acc>=0.80)} / {len(DYN_ACTIONS)}
    Classes 50-79%    : {np.sum((per_cls_acc>=0.50)&(per_cls_acc<0.80))} / {len(DYN_ACTIONS)}
    Classes < 50%     : {np.sum(per_cls_acc<0.50)} / {len(DYN_ACTIONS)}
  ─────────────────────────────────────────────────────
  Static model  : {STATIC_MODEL if static_loaded else 'not found'}
    Overall accuracy  : {stat_acc:.1%}
╚══════════════════════════════════════════════════════════╝

To generate graph 1 (training curves), add this to your
training script after model.fit():
  np.save('training_history.npy', history.history)
Then re-run this script.
""")