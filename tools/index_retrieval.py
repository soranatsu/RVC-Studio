"""Numerically safe FAISS feature retrieval for RVC index mixing."""

from __future__ import annotations

import numpy as np


def retrieve_index_features(index, vectors, query, k: int = 8):
    """Retrieve and distance-weight index features without propagating bad values.

    Exact zero-distance matches are retained as exact matches (and averaged if
    several occur).  Rows with no usable neighbour fall back to the original
    query row.
    """
    bank = np.asarray(vectors)
    original = np.asarray(query)
    if bank.ndim != 2 or original.ndim != 2:
        raise ValueError("vectors and query must be 2-D arrays")
    if bank.shape[1] != original.shape[1]:
        raise ValueError("vectors and query feature dimensions differ")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")

    result = original.copy()
    n = original.shape[0]
    stats = {
        "search_frames": int(n),
        "used_frames": 0,
        "invalid_neighbors": 0,
        "exact_match_frames": 0,
        "fallback_frames": int(n),
        "nonfinite_query_frames": 0,
    }
    finite_query = np.isfinite(original).all(axis=1)
    stats["nonfinite_query_frames"] = int((~finite_query).sum())
    if not finite_query.size or not finite_query.any() or bank.shape[0] == 0:
        return result, stats

    # FAISS expects float32 and cannot search NaN rows.  Keep the original
    # query for fallback; only finite rows enter the search.
    search_rows = np.flatnonzero(finite_query)
    distances, ids = index.search(
        np.ascontiguousarray(original[search_rows], dtype=np.float32), int(k)
    )
    distances = np.asarray(distances)
    ids = np.asarray(ids)
    if distances.shape != ids.shape or distances.ndim != 2:
        raise ValueError("index.search must return matching 2-D distance/id arrays")

    for local_row, row in enumerate(search_rows):
        row_dist = distances[local_row]
        row_ids = ids[local_row]
        valid = np.isfinite(row_dist) & (row_dist >= 0)
        valid &= np.isfinite(row_ids)
        valid &= (row_ids >= 0) & (row_ids < bank.shape[0])
        valid &= np.array(
            [np.isfinite(bank[int(i)]).all() if ok else False
             for i, ok in zip(row_ids, valid)], dtype=bool
        )
        stats["invalid_neighbors"] += int((~valid).sum())
        if not valid.any():
            continue

        valid_dist = row_dist[valid].astype(np.float64, copy=False)
        valid_ids = row_ids[valid].astype(np.int64, copy=False)
        exact = valid_dist <= 1e-12
        if exact.any():
            result[row] = bank[valid_ids[exact]].mean(axis=0, dtype=np.float64)
            stats["exact_match_frames"] += 1
        else:
            # L2 distances are non-negative.  The floor prevents a very close
            # non-exact match from overflowing inverse-square weights.
            weights = 1.0 / np.maximum(valid_dist, 1e-12) ** 2
            total = weights.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            weights /= total
            mixed = np.sum(
                bank[valid_ids].astype(np.float64) * weights[:, None], axis=0
            )
            if not np.isfinite(mixed).all():
                continue
            result[row] = mixed

        stats["used_frames"] += 1

    stats["fallback_frames"] = int(n - stats["used_frames"])
    return result.astype(original.dtype, copy=False), stats

