"""Visualization script comparing standard DBSCAN with two-level density DBSCAN."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from incdbscan import IncrementalDBSCAN

# Ensure images directory exists
os.makedirs('images', exist_ok=True)


def create_bridge_scenario():
    """Create a scenario with two clusters connected by bridge points."""
    np.random.seed(42)

    # Two well-separated clusters
    cluster1 = np.random.randn(30, 2) * 0.4 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.4 + np.array([6, 0])

    # Bridge points that could merge clusters
    bridge = np.array([
        [2.5, 0.2],
        [3.0, -0.1],
        [3.5, 0.15]
    ])

    return cluster1, cluster2, bridge


def create_chain_scenario():
    """Create a scenario with chain-like structure."""
    np.random.seed(123)

    # Three clusters that might chain together
    cluster1 = np.random.randn(25, 2) * 0.35 + np.array([0, 0])
    cluster2 = np.random.randn(25, 2) * 0.35 + np.array([4, 0])
    cluster3 = np.random.randn(25, 2) * 0.35 + np.array([8, 0])

    # Bridge points
    bridge1 = np.array([[1.8, 0], [2.2, 0]])
    bridge2 = np.array([[5.8, 0], [6.2, 0]])

    return cluster1, cluster2, cluster3, bridge1, bridge2


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


def visualize_bridge_comparison():
    """Visualize the bridge scenario comparing standard and two-level DBSCAN."""
    cluster1, cluster2, bridge = create_bridge_scenario()
    X = np.vstack([cluster1, cluster2, bridge])

    eps = 1.6
    eps_merge = 0.8
    min_pts = 3

    # Standard DBSCAN
    dbscan_standard = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps)
    dbscan_standard.insert(X)
    labels_standard = dbscan_standard.get_cluster_labels(X)

    # Two-level density DBSCAN
    dbscan_two_level = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge)
    dbscan_two_level.insert(X)
    labels_two_level = dbscan_two_level.get_cluster_labels(X)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot standard DBSCAN
    plot_clusters(axes[0], X, labels_standard,
                  f'Standard DBSCAN\n(eps={eps}, min_pts={min_pts})')

    # Plot two-level density DBSCAN
    plot_clusters(axes[1], X, labels_two_level,
                  f'Two-Level Density DBSCAN\n(eps={eps}, eps_merge={eps_merge}, min_pts={min_pts})')

    # Add annotations
    n_clusters_std = len(set(labels_standard)) - \
        (1 if -1 in labels_standard else 0)
    n_clusters_two = len(set(labels_two_level)) - \
        (1 if -1 in labels_two_level else 0)

    fig.suptitle(f'Bridge Scenario Comparison\n'
                 f'Standard: {n_clusters_std} clusters | Two-Level: {n_clusters_two} clusters',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/bridge_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/bridge_comparison.png")
    plt.close()


def visualize_chain_comparison():
    """Visualize the chain scenario with multiple bridge points."""
    cluster1, cluster2, cluster3, bridge1, bridge2 = create_chain_scenario()
    X = np.vstack([cluster1, cluster2, cluster3, bridge1, bridge2])

    eps = 1.4
    eps_merge = 0.7
    min_pts = 3

    # Standard DBSCAN
    dbscan_standard = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps)
    dbscan_standard.insert(X)
    labels_standard = dbscan_standard.get_cluster_labels(X)

    # Two-level density DBSCAN
    dbscan_two_level = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge)
    dbscan_two_level.insert(X)
    labels_two_level = dbscan_two_level.get_cluster_labels(X)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot standard DBSCAN
    plot_clusters(axes[0], X, labels_standard,
                  f'Standard DBSCAN\n(eps={eps}, min_pts={min_pts})')

    # Plot two-level density DBSCAN
    plot_clusters(axes[1], X, labels_two_level,
                  f'Two-Level Density DBSCAN\n(eps={eps}, eps_merge={eps_merge}, min_pts={min_pts})')

    # Add annotations
    n_clusters_std = len(set(labels_standard)) - \
        (1 if -1 in labels_standard else 0)
    n_clusters_two = len(set(labels_two_level)) - \
        (1 if -1 in labels_two_level else 0)

    fig.suptitle(f'Chain Scenario Comparison\n'
                 f'Standard: {n_clusters_std} clusters | Two-Level: {n_clusters_two} clusters',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/chain_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/chain_comparison.png")
    plt.close()


def visualize_incremental_process():
    """Visualize the incremental insertion process."""
    np.random.seed(42)

    cluster1 = np.random.randn(25, 2) * 0.35 + np.array([0, 0])
    cluster2 = np.random.randn(25, 2) * 0.35 + np.array([4.5, 0])
    bridge = np.array([[2.0, 0.1], [2.5, -0.1]])

    eps = 1.4
    eps_merge = 0.6
    min_pts = 3

    # Create two DBSCAN instances
    dbscan_std = IncrementalDBSCAN(eps=eps, min_pts=min_pts, eps_merge=eps)
    dbscan_two = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge)

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Step 1: Insert cluster 1
    dbscan_std.insert(cluster1)
    dbscan_two.insert(cluster1)

    X1 = cluster1
    labels_std_1 = dbscan_std.get_cluster_labels(X1)
    labels_two_1 = dbscan_two.get_cluster_labels(X1)

    plot_clusters(axes[0, 0], X1, labels_std_1,
                  'Step 1: Insert Cluster 1\n(Standard DBSCAN)')
    plot_clusters(axes[0, 1], X1, labels_two_1,
                  'Step 1: Insert Cluster 1\n(Two-Level Density)')

    # Step 2: Insert cluster 2
    dbscan_std.insert(cluster2)
    dbscan_two.insert(cluster2)

    X2 = np.vstack([cluster1, cluster2])
    labels_std_2 = dbscan_std.get_cluster_labels(X2)
    labels_two_2 = dbscan_two.get_cluster_labels(X2)

    plot_clusters(axes[1, 0], X2, labels_std_2,
                  'Step 2: Insert Cluster 2\n(Standard DBSCAN)')
    plot_clusters(axes[1, 1], X2, labels_two_2,
                  'Step 2: Insert Cluster 2\n(Two-Level Density)')

    # Step 3: Insert bridge
    dbscan_std.insert(bridge)
    dbscan_two.insert(bridge)

    X3 = np.vstack([cluster1, cluster2, bridge])
    labels_std_3 = dbscan_std.get_cluster_labels(X3)
    labels_two_3 = dbscan_two.get_cluster_labels(X3)

    plot_clusters(axes[2, 0], X3, labels_std_3,
                  'Step 3: Insert Bridge Points\n(Standard DBSCAN)')
    plot_clusters(axes[2, 1], X3, labels_two_3,
                  'Step 3: Insert Bridge Points\n(Two-Level Density)')

    # Highlight bridge points
    axes[2, 0].scatter(bridge[:, 0], bridge[:, 1],
                       s=200, facecolors='none', edgecolors='red',
                       linewidths=2, label='Bridge Points')
    axes[2, 1].scatter(bridge[:, 0], bridge[:, 1],
                       s=200, facecolors='none', edgecolors='red',
                       linewidths=2, label='Bridge Points')

    fig.suptitle(f'Incremental Insertion Process\n'
                 f'(eps={eps}, eps_merge={eps_merge}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/incremental_process.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/incremental_process.png")
    plt.close()


def visualize_parameter_sensitivity():
    """Visualize the effect of different eps_merge values."""
    cluster1, cluster2, bridge = create_bridge_scenario()
    X = np.vstack([cluster1, cluster2, bridge])

    eps = 1.6
    min_pts = 3
    eps_merge_values = [1.6, 1.2, 0.9, 0.6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, eps_merge in enumerate(eps_merge_values):
        dbscan = IncrementalDBSCAN(
            eps=eps, min_pts=min_pts, eps_merge=eps_merge)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        title = f'eps_merge = {eps_merge}\n({n_clusters} clusters, {n_noise} noise)'
        if eps_merge == eps:
            title += '\n(Standard DBSCAN)'

        plot_clusters(axes[idx], X, labels, title)

    fig.suptitle(f'Parameter Sensitivity Analysis\n(eps={eps}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/parameter_sensitivity.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/parameter_sensitivity.png")
    plt.close()


def visualize_eps_circles():
    """Visualize eps and eps_merge circles to show the concept."""
    np.random.seed(42)

    # Create a simple scenario with bridge points for clear demonstration
    cluster1 = np.random.randn(15, 2) * 0.25 + np.array([0, 0])
    cluster2 = np.random.randn(15, 2) * 0.25 + np.array([3.2, 0])
    bridge = np.array([[1.5, 0.1], [1.7, -0.1]])

    X = np.vstack([cluster1, cluster2, bridge])

    eps = 1.25
    eps_merge = 0.55
    min_pts = 3

    # Two-level density DBSCAN
    dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts, eps_merge=eps_merge)
    dbscan.insert(X)
    labels = dbscan.get_cluster_labels(X)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot points
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=80,
                   alpha=0.7, edgecolors='k', linewidth=1,
                   label=f'Cluster {int(k)}' if k >= 0 else 'Noise', zorder=3)

    # Draw eps circles (green) for a few points
    sample_points = [cluster1[0], cluster2[0], bridge[0]]
    for point in sample_points:
        circle_eps = Circle(point, eps, fill=False,
                            edgecolor='green', linestyle='-',
                            linewidth=2, alpha=0.5, zorder=1,
                            label='eps (neighbor range)' if point is sample_points[0] else '')
        ax.add_patch(circle_eps)

    # Draw eps_merge circles (red) for the same points
    for point in sample_points:
        circle_merge = Circle(point, eps_merge, fill=False,
                              edgecolor='red', linestyle='--',
                              linewidth=2, alpha=0.7, zorder=2,
                              label='eps_merge (merge range)' if point is sample_points[0] else '')
        ax.add_patch(circle_merge)

    ax.set_title(f'Two-Level Density Concept\n'
                 f'eps={eps} (neighbor/border assignment) vs eps_merge={eps_merge} (cluster merging)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal')

    # Add text annotation
    ax.text(0.02, 0.98,
            'Green circles (eps): Define neighbors for core point detection\n'
            'Red circles (eps_merge): Define connectivity for cluster merging\n'
            'Border points assigned using eps, but clusters merge only via eps_merge',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('images/eps_circles_concept.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/eps_circles_concept.png")
    plt.close()


def visualize_soft_clustering_probabilities():
    """Visualize soft clustering probabilities with heatmap."""
    np.random.seed(42)

    # Create overlapping clusters scenario
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([3, 0])
    cluster3 = np.random.randn(30, 2) * 0.5 + np.array([1.5, 2.5])

    X = np.vstack([cluster1, cluster2, cluster3])

    eps = 1.2
    eps_merge = 0.6
    eps_soft = 2.5
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge, eps_soft=eps_soft)
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
                 f'(eps={eps}, eps_merge={eps_merge}, eps_soft={eps_soft}, min_pts={min_pts})',
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
    eps_merge = 0.6
    eps_soft = 2.0
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge, eps_soft=eps_soft)
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
                 f'(eps={eps}, eps_merge={eps_merge}, eps_soft={eps_soft}, min_pts={min_pts})',
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
    eps_merge = 0.5
    eps_soft = 2.0
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge, eps_soft=eps_soft)
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
                 f'(eps={eps}, eps_merge={eps_merge}, eps_soft={eps_soft}, min_pts={min_pts})',
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
    eps_merge = 0.5
    eps_soft = 2.5
    min_pts = 3

    # Fit DBSCAN
    dbscan = IncrementalDBSCAN(
        eps=eps, min_pts=min_pts, eps_merge=eps_merge, eps_soft=eps_soft)
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
                 f'(eps={eps}, eps_merge={eps_merge}, eps_soft={eps_soft}, min_pts={min_pts})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/soft_clustering_boundary.png',
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: images/soft_clustering_boundary.png")
    plt.close()


def main():
    """Generate all comparison visualizations."""
    print("=" * 60)
    print("Generating Two-Level Density DBSCAN Visualizations")
    print("=" * 60)
    print()

    print("1. Generating bridge scenario comparison...")
    visualize_bridge_comparison()

    print("2. Generating chain scenario comparison...")
    visualize_chain_comparison()

    print("3. Generating incremental process visualization...")
    visualize_incremental_process()

    print("4. Generating parameter sensitivity analysis...")
    visualize_parameter_sensitivity()

    print("5. Generating eps circles concept diagram...")
    visualize_eps_circles()

    print()
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
    print("All visualizations generated successfully! ✓")
    print("=" * 60)
    print()
    print("Generated files in 'images/' directory:")
    print("  Hard Clustering:")
    print("    - images/bridge_comparison.png")
    print("    - images/chain_comparison.png")
    print("    - images/incremental_process.png")
    print("    - images/parameter_sensitivity.png")
    print("    - images/eps_circles_concept.png")
    print()
    print("  Soft Clustering:")
    print("    - images/soft_clustering_probabilities.png")
    print("    - images/soft_clustering_rgb.png")
    print("    - images/soft_clustering_kernels.png")
    print("    - images/soft_clustering_boundary.png")


if __name__ == "__main__":
    main()
