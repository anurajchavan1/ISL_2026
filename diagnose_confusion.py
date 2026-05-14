"""
diagnose_confusion.py

Run this BEFORE deciding whether to fix the data or fine-tune the model.
Outputs:
  1. confusion_matrix.png   — heatmap of where the model makes mistakes
  2. per_class_accuracy.png — bar chart sorted by accuracy, worst classes first
  3. worst_pairs.txt        — top 30 most confused sign pairs, printed to console

Usage:
    python diagnose_confusion.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.getcwd(), 'extracted_data')
MODEL_PATH      = 'isl_tcn_model.h5'
CLASSES_PATH    = 'classes.npy'
SEQUENCE_LENGTH = 30
FEATURES        = 690
TOP_N_WORST     = 30   # how many worst confused pairs to print

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model and classes...")
model   = tf.keras.models.load_model(MODEL_PATH)
ACTIONS = np.load(CLASSES_PATH)
label_map = {label: i for i, label in enumerate(ACTIONS)}

print("Loading data...")
sequences, labels = [], []
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue
    for f in os.listdir(action_path):
        if not f.endswith('.npy'):
            continue
        seq = np.load(os.path.join(action_path, f))
        if seq.shape == (SEQUENCE_LENGTH, FEATURES):
            sequences.append(seq)
            labels.append(label_map[action])

X = np.array(sequences)
y = np.array(labels)

# Use the same split as training so we evaluate on the held-out test set
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Test set: {len(X_test)} samples across {len(ACTIONS)} classes.")

# ── Predict ───────────────────────────────────────────────────────────────────
print("Running inference on test set (this may take a minute)...")
y_pred_probs = model.predict(X_test, batch_size=32, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_test

# ── Per-class accuracy ────────────────────────────────────────────────────────
per_class_correct = np.zeros(len(ACTIONS))
per_class_total   = np.zeros(len(ACTIONS))
for t, p in zip(y_true, y_pred):
    per_class_total[t]   += 1
    if t == p:
        per_class_correct[t] += 1

per_class_acc = np.where(
    per_class_total > 0,
    per_class_correct / per_class_total,
    0.0
)

overall_acc = np.mean(y_true == y_pred)
print(f"\nOverall test accuracy: {overall_acc:.1%}")
print(f"Classes with 0% accuracy:  {np.sum(per_class_acc == 0.0)}")
print(f"Classes with <50% accuracy: {np.sum(per_class_acc < 0.5)}")
print(f"Classes with >=80% accuracy: {np.sum(per_class_acc >= 0.8)}")

# Sort worst-first for the bar chart
sort_idx   = np.argsort(per_class_acc)
sorted_acc = per_class_acc[sort_idx]
sorted_lbl = ACTIONS[sort_idx]

# Plot: per-class accuracy bar chart (worst 60 shown if > 60 classes)
n_show = min(60, len(ACTIONS))
fig, ax = plt.subplots(figsize=(max(14, n_show * 0.35), 6))
colors = ['#E24B4A' if a < 0.5 else '#EF9F27' if a < 0.8 else '#639922'
          for a in sorted_acc[:n_show]]
ax.bar(range(n_show), sorted_acc[:n_show] * 100, color=colors, width=0.8)
ax.axhline(overall_acc * 100, color='#378ADD', linewidth=1.5,
           linestyle='--', label=f'Overall avg ({overall_acc:.1%})')
ax.set_xticks(range(n_show))
ax.set_xticklabels(sorted_lbl[:n_show], rotation=90, fontsize=7)
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Per-class accuracy — worst {n_show} classes (red < 50%, amber < 80%, green ≥ 80%)')
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150)
plt.close()
print("\nSaved: per_class_accuracy.png")

# ── Confusion matrix (top confused pairs) ────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=range(len(ACTIONS)))

# Zero the diagonal (correct predictions) so we focus on errors
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)

# Find the top-N most confused pairs
flat      = cm_errors.flatten()
top_idx   = np.argsort(flat)[::-1][:TOP_N_WORST]
rows, cols = np.unravel_index(top_idx, cm_errors.shape)

print(f"\n── Top {TOP_N_WORST} most confused sign pairs ──")
print(f"{'True sign':<25} {'Predicted as':<25} {'Count':>6}  {'True class acc':>14}")
print("-" * 75)
with open('worst_pairs.txt', 'w') as f:
    header = f"{'True sign':<25} {'Predicted as':<25} {'Count':>6}  {'True class acc':>14}\n"
    f.write(header)
    f.write("-" * 75 + "\n")
    for r, c in zip(rows, cols):
        count    = cm_errors[r, c]
        true_acc = per_class_acc[r]
        line = (f"{ACTIONS[r]:<25} {ACTIONS[c]:<25} {count:>6}  {true_acc:>13.1%}")
        print(line)
        f.write(line + "\n")
print("\nSaved: worst_pairs.txt")

# ── Confusion matrix heatmap (only classes with errors) ──────────────────────
# To keep the plot readable, show only the N classes involved in the most errors
active_classes = np.unique(np.concatenate([rows, cols]))
n_active = len(active_classes)

if n_active > 0:
    cm_sub = cm[np.ix_(active_classes, active_classes)]
    labels_sub = ACTIONS[active_classes]

    fig_size = max(10, n_active * 0.55)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    # Normalize per row (true class) so color = recall, not raw count
    row_sums = cm_sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm_sub / row_sums

    cmap = plt.cm.Blues
    im   = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label='Recall (row-normalized)')

    tick_pos = range(n_active)
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(labels_sub, rotation=90, fontsize=max(5, 9 - n_active // 20))
    ax.set_yticklabels(labels_sub, fontsize=max(5, 9 - n_active // 20))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion matrix — {n_active} classes with errors (row-normalized recall)')

    # Annotate cells with raw counts where non-zero
    for i in range(n_active):
        for j in range(n_active):
            val = cm_sub[i, j]
            if val > 0:
                color = 'white' if cm_norm[i, j] > 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=max(4, 7 - n_active // 25), color=color)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

# Add this at the end of diagnose_confusion.py, after the existing plots

# Full distribution summary
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print("\n── Full accuracy distribution across all 262 classes ──")
print(f"{'Range':<15} {'Classes':>8} {'% of total':>12}")
print("-" * 38)
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    count = np.sum((per_class_acc * 100 >= lo) & (per_class_acc * 100 < hi))
    if hi == 100:
        count = np.sum(per_class_acc * 100 >= lo)
    pct = count / len(ACTIONS) * 100
    bar = '█' * int(pct / 2)
    print(f"{lo:>3}–{hi:<3}%        {count:>6}      {pct:>6.1f}%  {bar}")

# Also plot ALL classes sorted, not just worst 60
fig, ax = plt.subplots(figsize=(max(20, len(ACTIONS) * 0.18), 6))
colors_all = ['#E24B4A' if a < 0.5 else '#EF9F27' if a < 0.8 else '#639922'
              for a in sorted_acc]
ax.bar(range(len(ACTIONS)), sorted_acc * 100, color=colors_all, width=0.8)
ax.axhline(overall_acc * 100, color='#378ADD', linewidth=1.5,
           linestyle='--', label=f'Overall avg ({overall_acc:.1%})')
ax.set_xticks(range(len(ACTIONS)))
ax.set_xticklabels(sorted_lbl, rotation=90, fontsize=5)
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Per-class accuracy — all {len(ACTIONS)} classes')
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy_full.png', dpi=150)
plt.close()
print("\nSaved: per_class_accuracy_full.png")
# ── Interpretation guide ─────────────────────────────────────────────────────
print("""
── How to read these results ────────────────────────────────────────────────

per_class_accuracy.png:
  Red bars  (<50%)  → these classes are broken. Check: do they have enough
                       videos? Are the signs phonetically similar to their
                       top confused pair?
  Amber bars (<80%) → underperforming but learning something. Fine-tuning
                       and more augmentation will help.
  Green bars (≥80%) → healthy. Don't over-optimize for these.

confusion_matrix.png:
  Bright off-diagonal cell = the model systematically confuses two specific
  signs. If those two signs look similar (same handshape, different motion),
  you need better motion features (joint angles, optical flow). If they look
  completely different, you likely have label noise in the dataset.

worst_pairs.txt:
  Read the "True class acc" column next to each confused pair. If a class
  has 0% accuracy AND its confusion count is high, the feature extractor
  is producing nearly identical vectors for two different signs — that is
  a feature engineering problem, not a capacity problem.

Decision rule:
  ≥30% of errors come from visually similar sign pairs  → data / features
  Errors are scattered across unrelated signs           → model capacity
  A few classes dominate all errors                     → class imbalance
  All classes are bad                                   → re-extract features
""")