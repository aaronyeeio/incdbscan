"""
Benchmark for edge eviction performance.

Tests the performance of evict_from_cluster_edge() under different scenarios:
- Varying cluster sizes
- Varying eviction amounts
- Different cluster topologies (dense vs. sparse, with/without articulation points)
"""

import time
import numpy as np
from incdbscan import IncrementalDBSCAN


def benchmark_eviction_cluster_size():
    """Benchmark eviction performance vs cluster size."""
    print("=" * 70)
    print("Benchmark 1: Eviction Performance vs Cluster Size")
    print("=" * 70)
    print()

    cluster_sizes = [100, 500, 1000, 5000]
    eps = 2.0
    min_pts = 5
    n_evict = 100  # Fixed eviction amount

    print(f"Configuration:")
    print(f"  eps: {eps}")
    print(f"  min_pts: {min_pts}")
    print(f"  Eviction request: {n_evict} points")
    print()

    results = []

    for cluster_size in cluster_sizes:
        print(f"Testing cluster size: {cluster_size} points...")

        # Generate 2D data with optimal density (spread=5.0)
        # Balances performance with cluster formation
        np.random.seed(42)
        X = np.random.randn(cluster_size, 2) * 5.0

        # Initialize and insert
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)

        print(f"Inserted {cluster_size} points")

        labels = dbscan.get_cluster_labels(X)
        unique_clusters = set(labels[labels >= 0])

        if len(unique_clusters) == 0:
            print(f"  Warning: No clusters formed")
            continue

        cluster_label = int(list(unique_clusters)[0])
        cluster = dbscan.get_cluster(cluster_label)

        print(f"Found {cluster_label} cluster")

        # Benchmark eviction
        start = time.time()
        evicted_count = dbscan.evict_from_cluster_edge(cluster_label, n_evict)
        eviction_time = time.time() - start

        print(f"  Evicted: {evicted_count} points")
        print(f"  Time: {eviction_time * 1000:.4f} ms")
        print(
            f"  Time per point: {eviction_time / evicted_count * 1000:.4f} ms")
        print()

        results.append((cluster_size, eviction_time, evicted_count))

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"{'Cluster Size':<15} {'Time (ms)':<15} {'Evicted':<15} {'ms/point':<15}")
    print("-" * 70)
    for size, t, evicted in results:
        print(f"{size:<15} {t*1000:<15.4f} {evicted:<15} {t/evicted*1000:<15.4f}")
    print("=" * 70)
    print()


def benchmark_eviction_amount():
    """Benchmark eviction performance vs number of points to evict."""
    print("=" * 70)
    print("Benchmark 2: Eviction Performance vs Eviction Amount")
    print("=" * 70)
    print()

    cluster_size = 5000
    eps = 2.0
    min_pts = 5
    eviction_amounts = [10, 50, 100, 500, 1000, 5000]

    print(f"Configuration:")
    print(f"  Cluster size: {cluster_size} points")
    print(f"  eps: {eps}")
    print(f"  min_pts: {min_pts}")
    print()

    results = []

    for n_evict in eviction_amounts:
        print(f"Testing eviction of {n_evict} points...")

        # Generate 2D data with spread=5.0
        np.random.seed(42)
        X = np.random.randn(cluster_size, 2) * 5.0

        # Initialize
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)

        labels = dbscan.get_cluster_labels(X)
        cluster_label = int(list(set(labels[labels >= 0]))[0])

        # Benchmark
        start = time.time()
        evicted_count = dbscan.evict_from_cluster_edge(cluster_label, n_evict)
        eviction_time = time.time() - start

        print(f"  Evicted: {evicted_count} points")
        print(f"  Time: {eviction_time * 1000:.4f} ms")
        print()

        results.append((n_evict, eviction_time, evicted_count))

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"{'Requested':<15} {'Time (ms)':<15} {'Evicted':<15} {'ms/point':<15}")
    print("-" * 70)
    for requested, t, evicted in results:
        if evicted > 0:
            print(
                f"{requested:<15} {t*1000:<15.4f} {evicted:<15} {t/evicted*1000:<15.4f}")
        else:
            print(f"{requested:<15} {t*1000:<15.4f} {evicted:<15} {'N/A':<15}")
    print("=" * 70)
    print()


def benchmark_topology_comparison():
    """Benchmark eviction on different cluster topologies."""
    print("=" * 70)
    print("Benchmark 3: Eviction Performance vs Cluster Topology")
    print("=" * 70)
    print()

    cluster_size = 1000
    n_evict = 200

    print(f"Configuration:")
    print(f"  Cluster size: {cluster_size} points")
    print(f"  Eviction request: {n_evict} points")
    print()

    topologies = []

    # 1. Spherical cluster (no articulation points)
    np.random.seed(42)
    dense_sphere = np.random.randn(cluster_size, 2) * 5.0
    topologies.append(("Spherical Cluster", dense_sphere, 2.0, 5))

    # 2. Linear chain (many articulation points)
    linear = np.array([[i * 0.8, 0.2 * np.sin(i * 0.3)]
                       for i in range(cluster_size)])
    linear += np.random.randn(cluster_size, 2) * 0.1
    topologies.append(("Linear Chain", linear, 1.0, 2))

    # 3. Star shape (central articulation point)
    center = np.random.randn(cluster_size // 7, 2) * 0.5
    arms = []
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        arm_points = cluster_size // 7
        arm = np.array([[np.cos(angle) * r, np.sin(angle) * r]
                        for r in np.linspace(1, 3, arm_points)])
        arm += np.random.randn(arm_points, 2) * 0.2
        arms.append(arm)
    star = np.vstack([center] + arms)
    topologies.append(("Star Shape", star, 1.5, 3))

    # 4. Sparse uniform (many border points)
    sparse = np.random.randn(cluster_size, 2) * 5.0
    topologies.append(("Sparse Uniform", sparse, 2.5, 5))

    results = []

    for name, X, eps, min_pts in topologies:
        print(f"Testing topology: {name}")
        print(f"  eps={eps}, min_pts={min_pts}")

        # Initialize
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)

        labels = dbscan.get_cluster_labels(X)
        unique_clusters = set(labels[labels >= 0])

        if len(unique_clusters) == 0:
            print(f"  Warning: No clusters formed")
            print()
            continue

        cluster_label = int(list(unique_clusters)[0])
        cluster = dbscan.get_cluster(cluster_label)

        initial_size = cluster.size
        initial_cores = cluster.core_count
        initial_borders = cluster.border_count

        # Benchmark
        start = time.time()
        evicted_count = dbscan.evict_from_cluster_edge(cluster_label, n_evict)
        eviction_time = time.time() - start

        remaining_size = cluster.size if cluster.size > 0 else 0
        remaining_cores = cluster.core_count if cluster.size > 0 else 0

        print(
            f"  Initial: {initial_size} pts ({initial_cores} cores, {initial_borders} borders)")
        print(f"  Evicted: {evicted_count} points")
        print(f"  Remaining: {remaining_size} pts ({remaining_cores} cores)")
        print(f"  Time: {eviction_time * 1000:.4f} ms")
        print()

        results.append((name, eviction_time, evicted_count,
                       initial_cores, initial_borders))

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"{'Topology':<20} {'Time (ms)':<12} {'Evicted':<10} {'Init Cores':<12} {'Init Borders':<12}")
    print("-" * 70)
    for name, t, evicted, cores, borders in results:
        print(f"{name:<20} {t*1000:<12.4f} {evicted:<10} {cores:<12} {borders:<12}")
    print("=" * 70)
    print()


def benchmark_safe_vs_split_eviction():
    """Compare performance of safe eviction vs component splitting."""
    print("=" * 70)
    print("Benchmark 4: Safe Eviction vs Component Splitting")
    print("=" * 70)
    print()

    cluster_size = 2000
    eps = 2.0
    min_pts = 3

    print(f"Configuration:")
    print(f"  Cluster size: {cluster_size} points")
    print(f"  eps: {eps}, min_pts: {min_pts}")
    print()

    # Generate 2D data with spread=5.0
    np.random.seed(42)
    X = np.random.randn(cluster_size, 2) * 5.0

    # Initialize
    dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
    dbscan.insert(X)

    labels = dbscan.get_cluster_labels(X)
    cluster_label = int(list(set(labels[labels >= 0]))[0])
    cluster = dbscan.get_cluster(cluster_label)

    # Test different eviction amounts
    eviction_scenarios = [
        ("Small (safe)", 50),
        ("Medium (safe)", 200),
        ("Large (safe)", 500),
        ("Very large (may split)", 1500),
        ("Maximum", 99999),
    ]

    results = []

    for scenario_name, n_evict in eviction_scenarios:
        # Reset
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        dbscan.insert(X)
        labels = dbscan.get_cluster_labels(X)
        cluster_label = int(list(set(labels[labels >= 0]))[0])

        print(f"Scenario: {scenario_name} (request n={n_evict})")

        # Benchmark
        start = time.time()
        evicted_count = dbscan.evict_from_cluster_edge(cluster_label, n_evict)
        eviction_time = time.time() - start

        cluster = dbscan.get_cluster(cluster_label)
        remaining = cluster.size if cluster.size > 0 else 0

        print(f"  Evicted: {evicted_count} points")
        print(f"  Remaining: {remaining} points")
        print(f"  Time: {eviction_time * 1000:.4f} ms")
        print()

        results.append((scenario_name, n_evict, evicted_count, eviction_time))

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"{'Scenario':<25} {'Requested':<12} {'Evicted':<10} {'Time (ms)':<12}")
    print("-" * 70)
    for name, requested, evicted, t in results:
        print(f"{name:<25} {requested:<12} {evicted:<10} {t*1000:<12.4f}")
    print("=" * 70)
    print()


def benchmark_high_dimensional():
    """Benchmark eviction on high-dimensional data."""
    print("=" * 70)
    print("Benchmark 5: High-Dimensional Data Eviction")
    print("=" * 70)
    print()

    cluster_size = 1000
    n_evict = 200
    eps = 5.0
    min_pts = 5

    dimensions = [2, 10, 50, 100, 256]

    print(f"Configuration:")
    print(f"  Cluster size: {cluster_size} points")
    print(f"  Eviction request: {n_evict} points")
    print(f"  eps: {eps}, min_pts: {min_pts}")
    print()

    results = []

    for n_dims in dimensions:
        print(f"Testing {n_dims} dimensions...")

        # Generate data
        np.random.seed(42)
        X = np.random.randn(cluster_size, n_dims).astype(np.float32) * 2.0

        # Initialize
        dbscan = IncrementalDBSCAN(eps=eps, min_pts=min_pts)

        # Measure insertion time too
        start_insert = time.time()
        dbscan.insert(X)
        insert_time = time.time() - start_insert

        labels = dbscan.get_cluster_labels(X)
        unique_clusters = set(labels[labels >= 0])

        if len(unique_clusters) == 0:
            print(f"  Warning: No clusters formed")
            print()
            continue

        cluster_label = int(list(unique_clusters)[0])

        # Benchmark eviction
        start = time.time()
        evicted_count = dbscan.evict_from_cluster_edge(cluster_label, n_evict)
        eviction_time = time.time() - start

        print(f"  Insert time: {insert_time * 1000:.4f} ms")
        print(f"  Eviction time: {eviction_time * 1000:.4f} ms")
        print(f"  Evicted: {evicted_count} points")
        print()

        results.append((n_dims, insert_time, eviction_time, evicted_count))

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"{'Dimensions':<15} {'Insert (ms)':<15} {'Evict (ms)':<15} {'Evicted':<10}")
    print("-" * 70)
    for dims, insert_t, evict_t, evicted in results:
        print(f"{dims:<15} {insert_t*1000:<15.4f} {evict_t*1000:<15.4f} {evicted:<10}")
    print("=" * 70)
    print()


def main():
    """Run all edge eviction benchmarks."""
    print()
    print("=" * 70)
    print("EDGE EVICTION PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()

    benchmark_eviction_cluster_size()
    benchmark_eviction_amount()
    benchmark_topology_comparison()
    benchmark_safe_vs_split_eviction()
    benchmark_high_dimensional()

    print()
    print("=" * 70)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
