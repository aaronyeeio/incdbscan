import time
import numpy as np
from incdbscan import IncrementalDBSCAN


def benchmark_insert_delete():
    """
    Benchmark insertion and deletion performance on 256-dimensional data.
    Base dataset: 10000 points
    Test: insert 10 points, delete 10 points
    """
    print("=" * 60)
    print("IncrementalDBSCAN Time Benchmark")
    print("=" * 60)

    # Configuration
    n_base = 50000
    n_test = 50
    n_dims = 256
    eps = 1.0
    min_pts = 5

    print(f"\nConfiguration:")
    print(f"  Base dataset size: {n_base} points")
    print(f"  Test batch size: {n_test} points")
    print(f"  Dimensions: {n_dims}")
    print(f"  eps: {eps}")
    print(f"  min_pts: {min_pts}")
    print()

    # Generate base dataset
    print("Generating base dataset...")
    np.random.seed(42)
    base_data = np.random.randn(n_base, n_dims).astype(np.float32)

    # Generate test data
    test_insert_data = np.random.randn(n_test, n_dims).astype(np.float32)
    # Use first 10 points for deletion
    test_delete_data = base_data[:n_test].copy()

    # Initialize algorithm
    print("Initializing IncrementalDBSCAN...")
    algo = IncrementalDBSCAN(eps=eps, min_pts=min_pts)

    # Insert base dataset
    print(f"Inserting {n_base} base points...")
    start = time.time()
    algo.insert(base_data)
    base_insert_time = time.time() - start
    print(f"  Time: {base_insert_time:.4f} seconds")
    print()

    # Benchmark batch insertion
    print(f"Benchmarking BATCH insertion of {n_test} points...")
    start = time.time()
    algo.insert(test_insert_data)
    batch_insert_time = time.time() - start
    print(f"  Time: {batch_insert_time:.6f} seconds")
    print(f"  Average per point: {batch_insert_time / n_test * 1000:.4f} ms")
    print()

    # Benchmark batch deletion
    print(f"Benchmarking BATCH deletion of {n_test} points...")
    start = time.time()
    algo.delete(test_delete_data)
    batch_delete_time = time.time() - start
    print(f"  Time: {batch_delete_time:.6f} seconds")
    print(f"  Average per point: {batch_delete_time / n_test * 1000:.4f} ms")
    print()

    # Benchmark single insertion
    print(f"Benchmarking SINGLE insertion of {n_test} points...")
    single_insert_data = np.random.randn(n_test, n_dims).astype(np.float32)
    start = time.time()
    for point in single_insert_data:
        algo.insert(point.reshape(1, -1))
    single_insert_time = time.time() - start
    print(f"  Time: {single_insert_time:.6f} seconds")
    print(f"  Average per point: {single_insert_time / n_test * 1000:.4f} ms")
    print()

    # Benchmark single deletion
    print(f"Benchmarking SINGLE deletion of {n_test} points...")
    single_delete_data = base_data[n_test:n_test * 2].copy()
    start = time.time()
    for point in single_delete_data:
        algo.delete(point.reshape(1, -1))
    single_delete_time = time.time() - start
    print(f"  Time: {single_delete_time:.6f} seconds")
    print(f"  Average per point: {single_delete_time / n_test * 1000:.4f} ms")
    print()

    # Summary
    print("=" * 60)
    print("Summary:")
    print(
        f"  Batch insert {n_test} points: {batch_insert_time:.6f} seconds ({batch_insert_time * 1000:.4f} ms)")
    print(
        f"  Single insert {n_test} points: {single_insert_time:.6f} seconds ({single_insert_time * 1000:.4f} ms)")
    print(f"  Speedup: {single_insert_time / batch_insert_time:.2f}x")
    print()
    print(
        f"  Batch delete {n_test} points: {batch_delete_time:.6f} seconds ({batch_delete_time * 1000:.4f} ms)")
    print(
        f"  Single delete {n_test} points: {single_delete_time:.6f} seconds ({single_delete_time * 1000:.4f} ms)")
    print(f"  Speedup: {single_delete_time / batch_delete_time:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_insert_delete()
