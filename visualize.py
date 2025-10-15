"""Visualization script comparing standard DBSCAN with two-level density DBSCAN."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from incdbscan import IncrementalDBSCAN

# Ensure images directory exists
os.makedirs('images', exist_ok=True)


def plot_clusters(ax, X, labels, title, eps=None, core_points=None):
    """Plot clustering results with optional eps circles."""
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points in black
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]

        ax.scatter(xy[:, 0], xy[:, 1], c=[col],
                   s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
                   label=f'Cluster {int(k)}' if k >= 0 else 'Noise')

    # Optionally draw eps circles around core points
    if eps is not None and core_points is not None:
        for point in core_points:
            circle = Circle(point, eps, fill=False,
                            edgecolor='gray', linestyle='--',
                            linewidth=0.5, alpha=0.3)
            ax.add_patch(circle)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')


def visualize_soft_clustering_probabilities():
    """Visualize soft clustering probabilities with heatmap."""
    np.random.seed(42)

    # Create overlapping clusters scenario
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([3, 0])
    cluster3 = np.random.randn(30, 2) * 0.5 + np.array([1.5, 2.5])

    X = np.vstack([cluster1, cluster2, cluster3])

    eps = 1.2
    eps_soft = 2.5
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_soft=eps_soft)
    dbscan.insert(X)

    # Get hard and soft labels
    hard_labels = dbscan.get_cluster_labels(X)
    soft_probs, cluster_labels = dbscan.get_soft_labels(
        X, kernel='gaussian', include_noise_prob=True)

    n_clusters = len(cluster_labels)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Hard clustering result
    ax1 = fig.add_subplot(gs[0, 0])
    plot_clusters(ax1, X, hard_labels,
                  f'Hard Clustering\n{n_clusters} clusters')

    # 2. Soft clustering probability heatmap
    ax2 = fig.add_subplot(gs[0, 1:])

    # Sort samples by their dominant cluster for better visualization
    dominant_cluster = np.argmax(soft_probs, axis=1)
    sorted_indices = np.argsort(dominant_cluster)

    im = ax2.imshow(soft_probs[sorted_indices].T, aspect='auto',
                    cmap='YlOrRd', interpolation='nearest')
    ax2.set_xlabel('Sample Index (sorted by dominant cluster)')
    ax2.set_ylabel('Cluster / Noise')

    # Set y-axis labels
    ytick_labels = [
        f'Cluster {int(label)}' for label in cluster_labels] + ['Noise']
    ax2.set_yticks(range(len(ytick_labels)))
    ax2.set_yticklabels(ytick_labels)

    ax2.set_title('Soft Clustering Probability Heatmap\n'
                  f'(eps_soft={eps_soft}, kernel=gaussian)',
                  fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Membership Probability', rotation=270, labelpad=15)

    # 3. Probability distribution for each cluster
    ax3 = fig.add_subplot(gs[1, :])

    # Select some representative points
    sample_indices = []
    for label in cluster_labels:
        # Find points with highest probability for this cluster
        cluster_col = np.where(cluster_labels == label)[0][0]
        top_idx = np.argsort(soft_probs[:, cluster_col])[-5:]
        sample_indices.extend(top_idx)

    # Also add some ambiguous points (high entropy)
    entropy = -np.sum(soft_probs * np.log(soft_probs + 1e-10), axis=1)
    ambiguous_indices = np.argsort(entropy)[-5:]
    sample_indices.extend(ambiguous_indices)
    sample_indices = list(set(sample_indices))[:15]  # Limit to 15 points

    # Bar plot for selected samples
    x_pos = np.arange(len(sample_indices))
    width = 0.15

    for i, label in enumerate(cluster_labels):
        cluster_col = np.where(cluster_labels == label)[0][0]
        probs_for_cluster = [soft_probs[idx, cluster_col]
                             for idx in sample_indices]
        ax3.bar(x_pos + i * width, probs_for_cluster, width,
                label=f'Cluster {int(label)}', alpha=0.8)

    # Add noise probability
    noise_probs = [soft_probs[idx, -1] for idx in sample_indices]
    ax3.bar(x_pos + n_clusters * width, noise_probs, width,
            label='Noise', alpha=0.8, color='black')

    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Membership Probability')
    ax3.set_title('Sample Points: Soft Clustering Probabilities',
                  fontweight='bold')
    ax3.set_xticks(x_pos + width * n_clusters / 2)
    ax3.set_xticklabels([str(idx) for idx in sample_indices], rotation=45)
    ax3.legend(loc='upper right', ncol=n_clusters+1, fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.1])

    fig.suptitle(f'Soft Clustering Verification\n'
                 f'(eps={eps}, eps_soft={eps_soft}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.savefig('images/soft_clustering_probabilities.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/soft_clustering_probabilities.png")
    plt.close()


def visualize_soft_clustering_rgb():
    """Visualize soft clustering using RGB color mixing."""
    np.random.seed(123)

    # Create three clusters
    cluster1 = np.random.randn(25, 2) * 0.4 + np.array([0, 0])
    cluster2 = np.random.randn(25, 2) * 0.4 + np.array([3, 0])
    cluster3 = np.random.randn(25, 2) * 0.4 + np.array([1.5, 2.5])

    X = np.vstack([cluster1, cluster2, cluster3])

    eps = 1.2
    eps_soft = 2.0
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_soft=eps_soft)
    dbscan.insert(X)

    # Get soft labels (without noise column for color mixing)
    soft_probs, cluster_labels = dbscan.get_soft_labels(
        X, kernel='gaussian', include_noise_prob=False)

    hard_labels = dbscan.get_cluster_labels(X)
    n_clusters = len(cluster_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hard clustering
    plot_clusters(axes[0], X, hard_labels,
                  f'Hard Clustering\n{n_clusters} clusters')

    # Right: Soft clustering with RGB color mixing
    if n_clusters >= 3:
        # Use first 3 clusters for RGB
        rgb_colors = np.zeros((len(X), 3))
        for i in range(min(3, n_clusters)):
            rgb_colors[:, i] = soft_probs[:, i]

        # Normalize to ensure valid RGB values
        row_sums = rgb_colors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        rgb_colors = rgb_colors / row_sums

        axes[1].scatter(X[:, 0], X[:, 1], c=rgb_colors, s=80,
                        edgecolors='k', linewidth=0.5, alpha=0.8)

        # Add legend showing RGB mapping
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='red', markersize=10,
                       label=f'Cluster {int(cluster_labels[0])} → Red'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='green', markersize=10,
                       label=f'Cluster {int(cluster_labels[1])} → Green'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='blue', markersize=10,
                       label=f'Cluster {int(cluster_labels[2])} → Blue')
        ]
        axes[1].legend(handles=legend_elements, loc='best', fontsize=8)

    elif n_clusters == 2:
        # Use Red and Green for 2 clusters
        rgb_colors = np.zeros((len(X), 3))
        rgb_colors[:, 0] = soft_probs[:, 0]  # Red
        rgb_colors[:, 1] = soft_probs[:, 1]  # Green

        axes[1].scatter(X[:, 0], X[:, 1], c=rgb_colors, s=80,
                        edgecolors='k', linewidth=0.5, alpha=0.8)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='red', markersize=10,
                       label=f'Cluster {int(cluster_labels[0])} → Red'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='green', markersize=10,
                       label=f'Cluster {int(cluster_labels[1])} → Green')
        ]
        axes[1].legend(handles=legend_elements, loc='best', fontsize=8)
    else:
        # Single cluster - just use grayscale
        axes[1].scatter(X[:, 0], X[:, 1], c=soft_probs[:, 0],
                        cmap='gray', s=80, edgecolors='k', linewidth=0.5, alpha=0.8)

    axes[1].set_title(f'Soft Clustering (RGB Color Mixing)\n'
                      f'Mixed colors = overlapping membership',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    fig.suptitle(f'Hard vs Soft Clustering Visualization\n'
                 f'(eps={eps}, eps_soft={eps_soft}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/soft_clustering_rgb.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/soft_clustering_rgb.png")
    plt.close()


def visualize_soft_clustering_kernels():
    """Compare different kernel functions for soft clustering."""
    np.random.seed(42)

    # Create clusters with some overlap
    cluster1 = np.random.randn(30, 2) * 0.4 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.4 + np.array([2.5, 0])

    X = np.vstack([cluster1, cluster2])

    eps = 1.0
    eps_soft = 2.0
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_soft=eps_soft)
    dbscan.insert(X)

    kernels = ['gaussian', 'inverse', 'linear']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, kernel in enumerate(kernels):
        soft_probs, cluster_labels = dbscan.get_soft_labels(
            X, kernel=kernel, include_noise_prob=False)

        n_clusters = len(cluster_labels)

        if n_clusters >= 2:
            # Use red-green mixing for visualization
            rgb_colors = np.zeros((len(X), 3))
            rgb_colors[:, 0] = soft_probs[:, 0]  # Red for cluster 0
            rgb_colors[:, 1] = soft_probs[:, 1]  # Green for cluster 1

            axes[idx].scatter(X[:, 0], X[:, 1], c=rgb_colors, s=80,
                              edgecolors='k', linewidth=0.5, alpha=0.8)

            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='red', markersize=10,
                           label=f'Cluster {int(cluster_labels[0])}'),
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='green', markersize=10,
                           label=f'Cluster {int(cluster_labels[1])}'),
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='yellow', markersize=10,
                           label='Mixed membership')
            ]
            axes[idx].legend(handles=legend_elements, loc='best', fontsize=8)

        axes[idx].set_title(f'{kernel.capitalize()} Kernel',
                            fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal')

    fig.suptitle(f'Soft Clustering: Kernel Comparison\n'
                 f'(eps={eps}, eps_soft={eps_soft}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/soft_clustering_kernels.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/soft_clustering_kernels.png")
    plt.close()


def visualize_soft_clustering_boundary():
    """Visualize soft clustering behavior at cluster boundaries."""
    np.random.seed(42)

    # Create two well-separated clusters
    cluster1 = np.random.randn(30, 2) * 0.35 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.35 + np.array([4, 0])

    X = np.vstack([cluster1, cluster2])

    eps = 1.0
    eps_soft = 2.5
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_soft=eps_soft)
    dbscan.insert(X)

    # Create a grid of query points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get soft labels for grid
    soft_probs_grid, cluster_labels = dbscan.get_soft_labels(
        grid_points, kernel='gaussian', include_noise_prob=True)

    hard_labels = dbscan.get_cluster_labels(X)
    n_clusters = len(cluster_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot probability for each cluster
    for cluster_idx in range(n_clusters):
        probs = soft_probs_grid[:, cluster_idx].reshape(xx.shape)

        im = axes[cluster_idx].contourf(xx, yy, probs, levels=20,
                                        cmap='YlOrRd', alpha=0.7)
        axes[cluster_idx].contour(xx, yy, probs, levels=[0.25, 0.5, 0.75],
                                  colors='black', linewidths=1, alpha=0.5)

        # Overlay actual data points
        mask = hard_labels == cluster_labels[cluster_idx]
        axes[cluster_idx].scatter(X[mask, 0], X[mask, 1], c='blue',
                                  s=30, edgecolors='k', linewidth=0.5,
                                  label=f'Cluster {int(cluster_labels[cluster_idx])} points',
                                  zorder=3)
        axes[cluster_idx].scatter(X[~mask, 0], X[~mask, 1], c='gray',
                                  s=20, alpha=0.3, edgecolors='k', linewidth=0.5,
                                  label='Other points', zorder=3)

        axes[cluster_idx].set_title(f'Cluster {int(cluster_labels[cluster_idx])} Probability',
                                    fontsize=12, fontweight='bold')
        axes[cluster_idx].set_xlabel('X')
        axes[cluster_idx].set_ylabel('Y')
        axes[cluster_idx].legend(loc='best', fontsize=8)
        axes[cluster_idx].set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[cluster_idx])
        cbar.set_label('Probability', rotation=270, labelpad=15)

    # Plot noise probability
    noise_probs = soft_probs_grid[:, -1].reshape(xx.shape)

    im = axes[2].contourf(xx, yy, noise_probs, levels=20,
                          cmap='gray_r', alpha=0.7)
    axes[2].contour(xx, yy, noise_probs, levels=[0.25, 0.5, 0.75],
                    colors='red', linewidths=1, alpha=0.5)

    # Overlay all data points
    axes[2].scatter(X[:, 0], X[:, 1], c='blue', s=30,
                    edgecolors='k', linewidth=0.5, zorder=3,
                    label='Clustered points')

    axes[2].set_title('Noise Probability', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].legend(loc='best', fontsize=8)
    axes[2].set_aspect('equal')

    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Probability', rotation=270, labelpad=15)

    fig.suptitle(f'Soft Clustering: Probability Fields\n'
                 f'(eps={eps}, eps_soft={eps_soft}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/soft_clustering_boundary.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/soft_clustering_boundary.png")
    plt.close()


def visualize_edge_eviction():
    """Visualize progressive edge eviction from different cluster shapes."""
    np.random.seed(42)

    # Create three different cluster shapes
    shapes = []

    # Shape 1: Dense circular cluster
    circular = np.random.randn(40, 2) * 0.5 + np.array([0, 0])
    shapes.append(('Circular Dense Cluster', circular, 1.2, 3))

    # Shape 2: Linear cluster (chain-like)
    linear = np.array([[i, 0.2 * np.sin(i * 0.5)]
                      for i in np.linspace(0, 8, 30)])
    linear += np.random.randn(30, 2) * 0.15
    shapes.append(('Linear Chain Cluster', linear, 1.2, 2))

    # Shape 3: Star-shaped cluster (dense center with arms)
    center = np.random.randn(20, 2) * 0.3
    arms = []
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
        arm = np.array([[np.cos(angle) * r, np.sin(angle) * r]
                        for r in np.linspace(1, 3, 5)])
        arm += np.random.randn(5, 2) * 0.15
        arms.append(arm)
    star = np.vstack([center] + arms)
    shapes.append(('Star-shaped Cluster', star, 1.2, 3))

    # Create figure with subplots for each shape
    fig = plt.figure(figsize=(18, 15))

    for shape_idx, (shape_name, X, eps, min_pts) in enumerate(shapes):
        # Initialize DBSCAN
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)

        initial_labels = dbscan.get_cluster_labels(X)
        unique_clusters = set(initial_labels[initial_labels >= 0])

        if len(unique_clusters) == 0:
            print(f"Warning: {shape_name} formed no clusters, skipping...")
            continue

        cluster_label = int(list(unique_clusters)[0])
        cluster = dbscan.get_cluster(cluster_label)
        initial_size = cluster.size

        # Determine eviction steps
        eviction_steps = [0, 3, 6, 10, 999]  # Last one is "evict to minimum"

        for step_idx, n_evict in enumerate(eviction_steps):
            ax = fig.add_subplot(len(shapes), len(eviction_steps),
                                 shape_idx * len(eviction_steps) + step_idx + 1)

            # Reset for each step
            dbscan_step = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
            dbscan_step.insert(X)

            # Get cluster
            labels_before = dbscan_step.get_cluster_labels(X)
            cluster_label_step = int(
                list(set(labels_before[labels_before >= 0]))[0])
            cluster_step = dbscan_step.get_cluster(cluster_label_step)

            # Evict points
            if n_evict > 0:
                evicted_count = dbscan_step.evict_from_cluster_edge(
                    cluster_label_step, n_evict)
            else:
                evicted_count = 0

            # Get updated labels
            labels_after = dbscan_step.get_cluster_labels(X)

            # Identify evicted points
            evicted_mask = (labels_before >= 0) & (labels_after == -1)
            remaining_mask = labels_after >= 0

            # Plot
            if np.any(remaining_mask):
                ax.scatter(X[remaining_mask, 0], X[remaining_mask, 1],
                           c='blue', s=60, alpha=0.7, edgecolors='k',
                           linewidth=0.5, label='Remaining', zorder=3)

            if np.any(evicted_mask):
                ax.scatter(X[evicted_mask, 0], X[evicted_mask, 1],
                           c='red', s=100, alpha=0.7,
                           linewidth=1.5, marker='x', label='Evicted', zorder=4)

            # Title
            remaining_size = cluster_step.size if cluster_step.size > 0 else 0
            if step_idx == 0:
                title = f'Initial\n{initial_size} points'
            elif n_evict == 999:
                title = f'Max Eviction\n{evicted_count} evicted, {remaining_size} remain'
            else:
                title = f'Request n={n_evict}\n{evicted_count} evicted, {remaining_size} remain'

            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend(loc='best', fontsize=7)

            # Add shape name on leftmost column
            if step_idx == 0:
                ax.text(-0.35, 0.5, shape_name, transform=ax.transAxes,
                        fontsize=11, fontweight='bold', rotation=90,
                        verticalalignment='center')

    fig.suptitle('Progressive Edge Eviction Visualization\n'
                 'Red X = Evicted points, Blue = Remaining cluster',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/edge_eviction_progressive.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/edge_eviction_progressive.png")
    plt.close()


def visualize_edge_eviction_detailed():
    """Detailed visualization of edge eviction with core/border distinction."""
    np.random.seed(123)

    # Create a cluster with clear core-border structure
    core_dense = np.random.randn(25, 2) * 0.4 + np.array([0, 0])
    border_sparse = np.random.randn(15, 2) * 0.3 + np.array([2.5, 0])
    sparse_arm = np.array([[i, 2.5] for i in np.linspace(-1, 1, 8)])
    sparse_arm += np.random.randn(8, 2) * 0.15

    X = np.vstack([core_dense, border_sparse, sparse_arm])

    eps = 1.0
    min_pts = 4

    # Create figure with detailed view
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    eviction_requests = [0, 5, 10, 15, 20, 999]

    for idx, n_evict in enumerate(eviction_requests):
        ax = axes[idx]

        # Reset DBSCAN for each step
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)

        labels_before = dbscan.get_cluster_labels(X)
        unique_clusters = set(labels_before[labels_before >= 0])

        if len(unique_clusters) == 0:
            ax.text(0.5, 0.5, 'No clusters formed',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        cluster_label = int(list(unique_clusters)[0])
        cluster = dbscan.get_cluster(cluster_label)

        # Get core/border status before eviction
        core_mask = np.array([dbscan._objects.get_object(X[i]).is_core
                             if labels_before[i] >= 0 else False
                             for i in range(len(X))])
        border_mask = (labels_before >= 0) & (~core_mask)

        initial_size = cluster.size
        initial_cores = cluster.core_count
        initial_borders = cluster.border_count

        # Perform eviction
        if n_evict > 0:
            evicted_count = dbscan.evict_from_cluster_edge(
                cluster_label, n_evict)
        else:
            evicted_count = 0

        labels_after = dbscan.get_cluster_labels(X)

        # Categorize points
        remaining_core = (labels_after >= 0) & core_mask
        remaining_border = (labels_after >= 0) & border_mask
        evicted_core = (labels_before >= 0) & (labels_after == -1) & core_mask
        evicted_border = (labels_before >= 0) & (
            labels_after == -1) & border_mask

        # Plot
        if np.any(remaining_core):
            ax.scatter(X[remaining_core, 0], X[remaining_core, 1],
                       c='darkblue', s=80, alpha=0.8, edgecolors='k',
                       linewidth=0.5, label='Core (remaining)', zorder=3)

        if np.any(remaining_border):
            ax.scatter(X[remaining_border, 0], X[remaining_border, 1],
                       c='lightblue', s=60, alpha=0.7, edgecolors='k',
                       linewidth=0.5, label='Border (remaining)', zorder=2)

        if np.any(evicted_border):
            ax.scatter(X[evicted_border, 0], X[evicted_border, 1],
                       c='orange', s=60, alpha=0.7, edgecolors='darkorange',
                       linewidth=0.8, marker='x', label='Border (evicted)', zorder=4)

        if np.any(evicted_core):
            ax.scatter(X[evicted_core, 0], X[evicted_core, 1],
                       c='red', s=80, alpha=0.8, edgecolors='darkred',
                       linewidth=1.0, marker='X', label='Core (evicted)', zorder=5)

        # Title with statistics
        remaining_size = cluster.size if cluster.size > 0 else 0
        remaining_cores = cluster.core_count if cluster.size > 0 else 0

        if idx == 0:
            title = (f'Initial State\n'
                     f'{initial_size} pts ({initial_cores} cores, {initial_borders} borders)')
        elif n_evict == 999:
            title = (f'Max Eviction (n=∞)\n'
                     f'{evicted_count} evicted → {remaining_size} remain ({remaining_cores} cores)')
        else:
            title = (f'Request n={n_evict}\n'
                     f'{evicted_count} evicted → {remaining_size} remain ({remaining_cores} cores)')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=7, ncol=2)

    fig.suptitle(f'Edge Eviction with Core/Border Distinction\n'
                 f'(eps={eps}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/edge_eviction_detailed.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/edge_eviction_detailed.png")
    plt.close()


def main():
    """Generate all comparison visualizations."""
    print("=" * 60)
    print("Generating Soft Clustering Visualizations")
    print("=" * 60)
    print()

    print("6. Generating soft clustering probabilities...")
    visualize_soft_clustering_probabilities()

    print("7. Generating soft clustering RGB visualization...")
    visualize_soft_clustering_rgb()

    print("8. Generating soft clustering kernel comparison...")
    visualize_soft_clustering_kernels()

    print("9. Generating soft clustering boundary visualization...")
    visualize_soft_clustering_boundary()

    print()
    print("=" * 60)
    print("Generating Edge Eviction Visualizations")
    print("=" * 60)
    print()

    print("10. Generating progressive edge eviction...")
    visualize_edge_eviction()

    print("11. Generating detailed edge eviction (core/border)...")
    visualize_edge_eviction_detailed()

    print()
    print("=" * 60)
    print("All visualizations generated successfully! ✓")
    print("=" * 60)
    print()
    print("Generated files in 'images/' directory:")
    print("  Soft Clustering:")
    print("    - images/soft_clustering_probabilities.png")
    print("    - images/soft_clustering_rgb.png")
    print("    - images/soft_clustering_kernels.png")
    print("    - images/soft_clustering_boundary.png")
    print("  Edge Eviction:")
    print("    - images/edge_eviction_progressive.png")
    print("    - images/edge_eviction_detailed.png")


if __name__ == "__main__":
    main()
