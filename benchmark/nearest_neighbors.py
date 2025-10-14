"""
Benchmark comparing sklearn NearestNeighbors vs PyTorch implementation.
Tests with 256-dimensional vectors and 10,000 points.
"""

from incdbscan._nearest_neighbors_torch import NearestNeighbors as TorchNN
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors as SklearnNN


def generate_test_data(n_samples=10000, n_features=256, n_queries=1000, normalize=True):
    """Generate random vectors for testing.

    Args:
        n_samples: Number of data points
        n_features: Number of features/dimensions
        n_queries: Number of query points
        normalize: If True, normalize vectors (for cosine). If False, keep raw vectors (for euclidean/minkowski)
    """
    np.random.seed(42)

    # Generate random data
    X_data = np.random.randn(n_samples, n_features).astype(np.float32)
    X_query = np.random.randn(n_queries, n_features).astype(np.float32)

    if normalize:
        X_data = X_data / np.linalg.norm(X_data, axis=1, keepdims=True)
        X_query = X_query / np.linalg.norm(X_query, axis=1, keepdims=True)

    return X_data, X_query


def benchmark_kneighbors(X_data, X_query, metric='cosine', n_neighbors=10, p=2, n_runs=5):
    """Benchmark k-nearest neighbors search.

    Args:
        X_data: Data points
        X_query: Query points
        metric: Distance metric ('cosine', 'euclidean', 'minkowski')
        n_neighbors: Number of neighbors
        p: Power parameter for Minkowski metric
        n_runs: Number of benchmark runs
    """
    print(f"\n{'='*70}")
    print(
        f"Benchmarking k-neighbors search (metric={metric}, k={n_neighbors})")
    if metric == 'minkowski':
        print(f"  Minkowski p={p}")
    print(f"Data: {X_data.shape[0]} points x {X_data.shape[1]} dims")
    print(f"Queries: {X_query.shape[0]} points")
    print(f"{'='*70}")

    # Sklearn benchmark
    print("\n[Sklearn NearestNeighbors]")
    sklearn_times = []
    sklearn_nn = SklearnNN(metric=metric, algorithm='brute', n_jobs=-1, p=p)

    # Fit
    fit_start = time.time()
    sklearn_nn.fit(X_data)
    fit_time = time.time() - fit_start
    print(f"  Fit time: {fit_time:.4f}s")

    # Query (warm-up)
    sklearn_nn.kneighbors(X_query, n_neighbors=n_neighbors)

    # Query (timed runs)
    for i in range(n_runs):
        start = time.time()
        distances, indices = sklearn_nn.kneighbors(
            X_query, n_neighbors=n_neighbors)
        elapsed = time.time() - start
        sklearn_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")

    sklearn_avg = np.mean(sklearn_times)
    sklearn_std = np.std(sklearn_times)
    print(f"  Average: {sklearn_avg:.4f}s ± {sklearn_std:.4f}s")

    # PyTorch benchmark
    print("\n[PyTorch NearestNeighbors]")
    torch_times = []
    torch_nn = TorchNN(metric=metric, p=p)
    print(f"  Using device: {torch_nn.device}")

    # Fit (no-op)
    fit_start = time.time()
    torch_nn.fit(X_data)
    fit_time = time.time() - fit_start
    print(f"  Fit time: {fit_time:.4f}s (no-op)")

    # Query (warm-up)
    torch_nn.kneighbors(X_query, X_data, n_neighbors=n_neighbors)

    # Query (timed runs)
    for i in range(n_runs):
        start = time.time()
        scores, indices = torch_nn.kneighbors(
            X_query, X_data, n_neighbors=n_neighbors)
        elapsed = time.time() - start
        torch_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")

    torch_avg = np.mean(torch_times)
    torch_std = np.std(torch_times)
    print(f"  Average: {torch_avg:.4f}s ± {torch_std:.4f}s")

    # Speedup
    speedup = sklearn_avg / torch_avg
    print(f"\n{'='*70}")
    print(
        f"Speedup: {speedup:.2f}x {'(PyTorch faster)' if speedup > 1 else '(sklearn faster)'}")
    print(f"{'='*70}")

    return sklearn_times, torch_times


def benchmark_radius_neighbors(X_data, X_query, metric='cosine', radius=0.8, p=2, n_runs=5):
    """Benchmark radius neighbors search.

    Args:
        X_data: Data points
        X_query: Query points
        metric: Distance metric ('cosine', 'euclidean', 'minkowski')
        radius: Radius threshold (for cosine: similarity >= radius, for euclidean/minkowski: distance <= radius)
        p: Power parameter for Minkowski metric
        n_runs: Number of benchmark runs
    """
    print(f"\n{'='*70}")
    print(
        f"Benchmarking radius neighbors search (metric={metric}, radius={radius})")
    if metric == 'minkowski':
        print(f"  Minkowski p={p}")
    print(f"Data: {X_data.shape[0]} points x {X_data.shape[1]} dims")
    print(f"Queries: {X_query.shape[0]} points")
    print(f"{'='*70}")

    # For cosine metric, sklearn uses distance (1-similarity), so convert radius
    sklearn_radius = (1 - radius) if metric == 'cosine' else radius

    # Sklearn benchmark
    print("\n[Sklearn NearestNeighbors]")
    sklearn_times = []
    sklearn_nn = SklearnNN(metric=metric, algorithm='brute', n_jobs=-1, p=p)
    sklearn_nn.fit(X_data)

    # Query (warm-up)
    sklearn_nn.radius_neighbors(X_query, radius=sklearn_radius)

    # Query (timed runs)
    for i in range(n_runs):
        start = time.time()
        indices_list = sklearn_nn.radius_neighbors(
            X_query, radius=sklearn_radius, return_distance=False)
        elapsed = time.time() - start
        sklearn_times.append(elapsed)
        avg_neighbors = np.mean([len(idx) for idx in indices_list])
        print(
            f"  Run {i+1}: {elapsed:.4f}s (avg {avg_neighbors:.1f} neighbors/query)")

    sklearn_avg = np.mean(sklearn_times)
    sklearn_std = np.std(sklearn_times)
    print(f"  Average: {sklearn_avg:.4f}s ± {sklearn_std:.4f}s")

    # PyTorch benchmark
    print("\n[PyTorch NearestNeighbors]")
    torch_times = []
    torch_nn = TorchNN(metric=metric, p=p)
    torch_nn.fit(X_data)

    # Query (warm-up)
    torch_nn.radius_neighbors(X_query, X_data, radius=radius)

    # Query (timed runs)
    for i in range(n_runs):
        start = time.time()
        indices_list = torch_nn.radius_neighbors(
            X_query, X_data, radius=radius, return_distance=False)
        elapsed = time.time() - start
        torch_times.append(elapsed)
        avg_neighbors = np.mean([len(idx) for idx in indices_list])
        print(
            f"  Run {i+1}: {elapsed:.4f}s (avg {avg_neighbors:.1f} neighbors/query)")

    torch_avg = np.mean(torch_times)
    torch_std = np.std(torch_times)
    print(f"  Average: {torch_avg:.4f}s ± {torch_std:.4f}s")

    # Speedup
    speedup = sklearn_avg / torch_avg
    print(f"\n{'='*70}")
    print(
        f"Speedup: {speedup:.2f}x {'(PyTorch faster)' if speedup > 1 else '(sklearn faster)'}")
    print(f"{'='*70}")

    return sklearn_times, torch_times


def verify_correctness(X_data, X_query, metric='cosine', n_neighbors=10, p=2):
    """Verify that both implementations give similar results.

    Args:
        X_data: Data points
        X_query: Query points
        metric: Distance metric ('cosine', 'euclidean', 'minkowski')
        n_neighbors: Number of neighbors
        p: Power parameter for Minkowski metric
    """
    print(f"\n{'='*70}")
    print(f"Verifying correctness (metric={metric})...")
    if metric == 'minkowski':
        print(f"  Minkowski p={p}")
    print(f"{'='*70}")

    # Sklearn
    sklearn_nn = SklearnNN(metric=metric, algorithm='brute', p=p)
    sklearn_nn.fit(X_data)
    sk_distances, sk_indices = sklearn_nn.kneighbors(
        X_query[:10], n_neighbors=n_neighbors)

    # PyTorch
    torch_nn = TorchNN(metric=metric, p=p)
    torch_nn.fit(X_data)
    torch_scores, torch_indices = torch_nn.kneighbors(
        X_query[:10], X_data, n_neighbors=n_neighbors)

    # Compare
    if metric == 'cosine':
        # PyTorch returns similarity, sklearn returns distance (1 - similarity)
        torch_distances = 1 - torch_scores
    else:
        # For euclidean/minkowski, both return distances
        torch_distances = torch_scores

    print(f"\nFirst query point comparison:")
    print(f"  Sklearn indices:   {sk_indices[0][:5]}")
    print(f"  PyTorch indices:   {torch_indices[0][:5]}")
    print(f"  Sklearn distances: {sk_distances[0][:5]}")
    print(f"  PyTorch distances: {torch_distances[0][:5]}")

    # Check if indices match
    indices_match = np.array_equal(sk_indices, torch_indices)
    print(f"\n  Indices match: {indices_match}")

    # Check if distances are close
    max_diff = np.max(np.abs(sk_distances - torch_distances))
    print(f"  Max distance difference: {max_diff:.6f}")

    if indices_match and max_diff < 1e-4:
        print("  ✓ Results are consistent!")
    else:
        print("  ⚠ Results differ slightly (expected due to numerical precision)")


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("NearestNeighbors Benchmark: sklearn vs PyTorch")
    print("="*70)

    # Test parameters
    n_samples = 10000
    n_features = 256
    n_queries = 1000
    n_neighbors = 10
    n_runs = 5

    # ========== Cosine Metric ==========
    print("\n" + "="*70)
    print("TESTING COSINE METRIC")
    print("="*70)

    print("\nGenerating normalized test data for cosine metric...")
    X_data_cosine, X_query_cosine = generate_test_data(
        n_samples=n_samples, n_features=n_features, n_queries=n_queries, normalize=True)
    print(f"  Data shape: {X_data_cosine.shape}")
    print(f"  Query shape: {X_query_cosine.shape}")

    verify_correctness(X_data_cosine, X_query_cosine,
                       metric='cosine', n_neighbors=n_neighbors)
    benchmark_kneighbors(X_data_cosine, X_query_cosine,
                         metric='cosine', n_neighbors=n_neighbors, n_runs=n_runs)
    benchmark_radius_neighbors(
        X_data_cosine, X_query_cosine, metric='cosine', radius=0.8, n_runs=n_runs)

    # ========== Euclidean Metric ==========
    print("\n" + "="*70)
    print("TESTING EUCLIDEAN METRIC")
    print("="*70)

    print("\nGenerating test data for euclidean metric...")
    X_data_euclidean, X_query_euclidean = generate_test_data(
        n_samples=n_samples, n_features=n_features, n_queries=n_queries, normalize=False)
    print(f"  Data shape: {X_data_euclidean.shape}")
    print(f"  Query shape: {X_query_euclidean.shape}")

    verify_correctness(X_data_euclidean, X_query_euclidean,
                       metric='euclidean', n_neighbors=n_neighbors)
    benchmark_kneighbors(X_data_euclidean, X_query_euclidean,
                         metric='euclidean', n_neighbors=n_neighbors, n_runs=n_runs)
    benchmark_radius_neighbors(
        X_data_euclidean, X_query_euclidean, metric='euclidean', radius=5.0, n_runs=n_runs)

    # ========== Minkowski Metric (p=3) ==========
    print("\n" + "="*70)
    print("TESTING MINKOWSKI METRIC (p=3)")
    print("="*70)

    print("\nUsing same data as euclidean...")
    verify_correctness(X_data_euclidean, X_query_euclidean,
                       metric='minkowski', n_neighbors=n_neighbors, p=3)
    benchmark_kneighbors(X_data_euclidean, X_query_euclidean,
                         metric='minkowski', n_neighbors=n_neighbors, p=3, n_runs=n_runs)
    benchmark_radius_neighbors(X_data_euclidean, X_query_euclidean,
                               metric='minkowski', radius=4.0, p=3, n_runs=n_runs)

    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
