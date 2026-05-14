import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_PATH = 'extracted_data'

def plot_per_video_variance(class_folder, class_name):
    path = os.path.join(DATA_PATH, class_folder)
    sequences = []
    filenames = []
    for f in sorted(os.listdir(path)):
        if not f.endswith('.npy'):
            continue
        seq = np.load(os.path.join(path, f))
        sequences.append(seq.flatten())
        filenames.append(f.replace('.npy', ''))

    X = np.array(sequences)
    
    # PCA to 2D to see how spread the videos are
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=100, color='#E24B4A', edgecolors='white')
    for i, name in enumerate(filenames):
        ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    
    # Draw lines from centroid to each point to show spread
    centroid = X_2d.mean(axis=0)
    ax.scatter(*centroid, s=200, color='black', marker='+', zorder=5, label='centroid')
    for point in X_2d:
        ax.plot([centroid[0], point[0]], [centroid[1], point[1]],
                color='gray', alpha=0.3, linewidth=0.8)
    
    variance_explained = pca.explained_variance_ratio_[:2].sum()
    spread = np.mean(np.linalg.norm(X_2d - centroid, axis=1))
    
    ax.set_title(f'{class_name} — per-video spread (PCA, {variance_explained:.0%} variance)\n'
                 f'Mean distance from centroid: {spread:.1f}  (lower = more consistent)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'variance_{class_name.lower()}.png', dpi=150)
    plt.close()
    print(f"Saved: variance_{class_name.lower()}.png")
    print(f"  Mean spread: {spread:.1f}")
    print(f"  Videos ranked by distance from centroid (most outlying first):")
    distances = np.linalg.norm(X_2d - centroid, axis=1)
    for idx in np.argsort(distances)[::-1]:
        print(f"    {filenames[idx]}: {distances[idx]:.1f}")

plot_per_video_variance('4. Money', 'Money')
plot_per_video_variance('8. Blind', 'Blind')