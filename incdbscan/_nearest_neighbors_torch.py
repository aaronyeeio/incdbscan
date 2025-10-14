import torch
import numpy as np


class NearestNeighbors:
    """
    PyTorch-based neighbor searcher supporting multiple distance metrics.
    Compatible with sklearn NearestNeighbors interface.
    """

    def __init__(self, metric="cosine", device=None, p=2, **kwargs):
        """
        Args:
            metric: "cosine", "euclidean", "manhattan", or "minkowski"
            device: PyTorch device (default: "cuda" if available else "cpu")
            p: Power parameter for Minkowski metric (default: 2)
        """
        if metric not in ["cosine", "euclidean", "manhattan", "minkowski"]:
            raise ValueError(
                f"Metric '{metric}' not supported. Choose from: cosine, euclidean, manhattan, minkowski")
        self.metric = metric
        self.p = p
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self._fit_X = None

    def fit(self, X, y=None):
        """
        Store the data for later queries.
        Compatible with sklearn interface.
        """
        self._fit_X = X
        return self

    def _compute_scores(self, X_query_tensor, X_data_tensor):
        """
        Compute distance for all metrics (lower is better).

        Returns:
            scores: distance (lower is better)
            is_similarity: Always False (all metrics return distances)
        """
        if self.metric == "cosine":
            X_data_norm = torch.nn.functional.normalize(
                X_data_tensor, p=2, dim=1)
            X_query_norm = torch.nn.functional.normalize(
                X_query_tensor, p=2, dim=1)
            cosine_sim = torch.matmul(X_query_norm, X_data_norm.T)
            scores = 1.0 - cosine_sim  # Convert similarity to distance
            return scores, False
        elif self.metric == "euclidean":
            scores = torch.cdist(X_query_tensor, X_data_tensor, p=2)
            return scores, False
        elif self.metric == "manhattan":
            scores = torch.cdist(X_query_tensor, X_data_tensor, p=1)
            return scores, False
        elif self.metric == "minkowski":
            scores = torch.cdist(X_query_tensor, X_data_tensor, p=self.p)
            return scores, False

    def kneighbors(self, X_query, X_data, n_neighbors=5, return_distance=True):
        """
        Find k nearest neighbors using the configured metric.

        Args:
            X_query: Query points (n_queries, n_features)
            X_data: Data points to search in (n_samples, n_features)
            n_neighbors: Number of neighbors to return
            return_distance: If True, return (distances, indices), else only indices

        Returns:
            If return_distance=True: (distances/similarities, indices)
            If return_distance=False: indices only
        """
        if len(X_data) == 0:
            if return_distance:
                return np.array([]).reshape(len(X_query), 0), np.array([]).reshape(len(X_query), 0)
            return np.array([]).reshape(len(X_query), 0)

        # Convert to tensors
        X_data_tensor = torch.from_numpy(np.asarray(
            X_data, dtype=np.float32)).to(self.device)
        X_query_tensor = torch.from_numpy(np.asarray(
            X_query, dtype=np.float32)).to(self.device)

        # Compute distances
        scores, _ = self._compute_scores(
            X_query_tensor, X_data_tensor)

        # Get top-k (smallest distances)
        k = min(n_neighbors, len(X_data))
        topk_scores, indices = torch.topk(
            scores, k=k, dim=1, largest=False)

        # Convert to numpy
        indices_np = indices.cpu().numpy()
        scores_np = topk_scores.cpu().numpy()

        if return_distance:
            return scores_np, indices_np
        return indices_np

    def radius_neighbors(self, X_query=None, X_data=None, radius=None, return_distance=False):
        """
        Find all neighbors within radius using the configured metric.

        Args:
            X_query: Query points (n_queries, n_features). If None, use fitted data.
            X_data: Data points to search in (n_samples, n_features). If None, use fitted data.
            radius: Distance threshold (<= radius are neighbors) for all metrics
            return_distance: If True, return (distances, indices), else only indices

        Returns:
            If return_distance=True: (list of distances arrays, list of indices arrays)
            If return_distance=False: list of indices arrays
        """
        if X_query is None:
            X_query = self._fit_X
        if X_data is None:
            X_data = self._fit_X

        if len(X_data) == 0:
            if return_distance:
                return [np.array([]) for _ in range(len(X_query))], [np.array([]) for _ in range(len(X_query))]
            return [np.array([]) for _ in range(len(X_query))]

        # Convert data to tensor once
        X_data_tensor = torch.from_numpy(np.asarray(
            X_data, dtype=np.float32)).to(self.device)

        n_queries = len(X_query)
        result_indices = [np.array([], dtype=np.int64)
                          for _ in range(n_queries)]
        result_scores = [np.array([], dtype=np.float32)
                         for _ in range(n_queries)]

        # Process in batches to avoid creating matrices larger than INT_MAX
        batch_size = min(1000, max(1, int(2_000_000_000 // len(X_data))))

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            X_query_batch = X_query[start_idx:end_idx]

            X_query_tensor = torch.from_numpy(np.asarray(
                X_query_batch, dtype=np.float32)).to(self.device)

            # Compute distances
            scores, _ = self._compute_scores(
                X_query_tensor, X_data_tensor)

            # Find matching indices (distance <= radius)
            mask = scores <= radius
            query_indices, data_indices = torch.nonzero(mask, as_tuple=True)

            if len(query_indices) > 0:
                # Sort by query index for efficient grouping
                sorted_indices = torch.argsort(query_indices)
                query_indices = query_indices[sorted_indices]
                data_indices = data_indices[sorted_indices]
                score_values = scores[query_indices, data_indices]

                # Transfer to CPU
                query_indices_np = query_indices.cpu().numpy()
                data_indices_np = data_indices.cpu().numpy()
                score_values_np = score_values.cpu().numpy()

                # Group by query index
                if len(query_indices_np) > 0:
                    split_points = np.where(
                        np.diff(query_indices_np) != 0)[0] + 1
                    split_points = np.concatenate(
                        [[0], split_points, [len(query_indices_np)]])

                    for i in range(len(split_points) - 1):
                        start, end = split_points[i], split_points[i + 1]
                        if start < end:
                            # Adjust for batch offset
                            q_idx = query_indices_np[start] + start_idx
                            result_indices[q_idx] = data_indices_np[start:end]
                            result_scores[q_idx] = score_values_np[start:end]

        if return_distance:
            return result_scores, result_indices
        return result_indices
