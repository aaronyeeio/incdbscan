"""Visualization script comparing standard DBSCAN with two-level density DBSCAN."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
    print("All visualizations generated successfully! ✓")
    print("=" * 60)
    print()
    print("Generated files in 'images/' directory:")
    print("  - images/bridge_comparison.png")
    print("  - images/chain_comparison.png")
    print("  - images/incremental_process.png")
    print("  - images/parameter_sensitivity.png")
    print("  - images/eps_circles_concept.png")


if __name__ == "__main__":
    main()
