# Copyright © 2025 Apple Inc.

"""
Symbolic analysis and CTSF construction for tile-based sparse Cholesky.

This module runs on CPU (numpy) and produces the metadata arrays that
the SparseCholeskyFactor GPU primitive consumes.

Terminology
-----------
- **Tile**: a ``tile_size × tile_size`` dense block.
- **Tile matrix**: the original N×N matrix viewed as an ``Nt × Nt`` grid of
  tiles, where ``Nt = ceil(N / tile_size)``.
- **CTSF** (Contiguous Tile Storage Format): nonzero tiles packed into a flat
  buffer with block-CSC indexing (``col_ptr``, ``row_idx``).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Convert a scalar CSC matrix to a tile-level adjacency structure
# ---------------------------------------------------------------------------

def tile_structure_from_csc(
    indptr: np.ndarray,
    indices: np.ndarray,
    N: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine which tiles of a CSC matrix are structurally nonzero.

    Only the lower triangle is kept (for Cholesky).

    Parameters
    ----------
    indptr, indices : CSC index arrays of the *full* (symmetric) matrix.
    N : matrix dimension.
    tile_size : tile side length.

    Returns
    -------
    tile_col_ptr : int32 array of length ``Nt + 1``
    tile_row_idx : int32 array (sorted per column, lower triangle only)
    """
    Nt = (N + tile_size - 1) // tile_size

    # Collect the set of nonzero tile (row, col) pairs, lower triangle
    tile_set: set = set()
    for scalar_col in range(N):
        tc = scalar_col // tile_size
        for idx in range(indptr[scalar_col], indptr[scalar_col + 1]):
            scalar_row = indices[idx]
            tr = scalar_row // tile_size
            if tr >= tc:
                tile_set.add((tr, tc))
            if tc >= tr:
                tile_set.add((tc, tr))  # symmetrise → take lower

    # Always include diagonal tiles
    for j in range(Nt):
        tile_set.add((j, j))

    # Build CSC-style structure
    cols: dict[int, list[int]] = {j: [] for j in range(Nt)}
    for tr, tc in tile_set:
        if tr >= tc:
            cols[tc].append(tr)

    tile_col_ptr = np.zeros(Nt + 1, dtype=np.int32)
    row_lists: list[list[int]] = []
    for j in range(Nt):
        rows_sorted = sorted(cols[j])
        row_lists.append(rows_sorted)
        tile_col_ptr[j + 1] = tile_col_ptr[j] + len(rows_sorted)

    tile_row_idx = np.concatenate(row_lists).astype(np.int32) if row_lists else np.array([], dtype=np.int32)
    return tile_col_ptr, tile_row_idx


# ---------------------------------------------------------------------------
# 2. Symbolic factorisation – compute fill-in at the tile level
# ---------------------------------------------------------------------------

def symbolic_factorization(
    tile_col_ptr: np.ndarray,
    tile_row_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symbolic Cholesky on the tile adjacency graph (fill-in computation).

    Parameters
    ----------
    tile_col_ptr, tile_row_idx : tile-level lower-triangular CSC structure of A.

    Returns
    -------
    L_col_ptr, L_row_idx : tile-level CSC structure of L (with fill-in).
    """
    Nt = len(tile_col_ptr) - 1

    # Mutable adjacency: col → sorted set of row indices (lower triangle)
    adj: list[set[int]] = [set() for _ in range(Nt)]
    for j in range(Nt):
        for idx in range(tile_col_ptr[j], tile_col_ptr[j + 1]):
            adj[j].add(int(tile_row_idx[idx]))

    # Standard symbolic elimination
    for j in range(Nt):
        rows_j = sorted(r for r in adj[j] if r > j)
        if not rows_j:
            continue
        # The minimum row above j in column j becomes the "parent" in the
        # elimination tree.  All other rows in column j create fill-in in
        # the parent's column.
        parent = rows_j[0]
        for r in rows_j[1:]:
            adj[parent].add(r)
        # Also propagate existing fill-in from column j to parent
        for r in rows_j:
            adj[parent].add(r)

    # Rebuild CSC arrays
    L_col_ptr = np.zeros(Nt + 1, dtype=np.int32)
    all_rows: list[int] = []
    for j in range(Nt):
        rows_sorted = sorted(adj[j])
        all_rows.extend(rows_sorted)
        L_col_ptr[j + 1] = L_col_ptr[j] + len(rows_sorted)

    L_row_idx = np.array(all_rows, dtype=np.int32)
    return L_col_ptr, L_row_idx


# ---------------------------------------------------------------------------
# 3. Build the Contiguous Tile Storage Format (CTSF)
# ---------------------------------------------------------------------------

def build_ctsf(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    N: int,
    tile_size: int,
    L_col_ptr: np.ndarray,
    L_row_idx: np.ndarray,
) -> np.ndarray:
    """Pack scalar CSC values into CTSF tiles.

    Tiles corresponding to fill-in positions are zero-initialised.

    Parameters
    ----------
    indptr, indices, data : CSC arrays of the *lower triangle* of A
        (or full symmetric – both halves are read).
    N : matrix order.
    tile_size : tile side length.
    L_col_ptr, L_row_idx : tile-level CSC structure of L (from symbolic).

    Returns
    -------
    tiles : float32 array of shape ``(num_tiles, tile_size, tile_size)``
    """
    n = tile_size
    num_tiles = int(L_col_ptr[-1])
    tiles = np.zeros((num_tiles, n, n), dtype=np.float32)

    # Build a fast lookup: (tile_row, tile_col) → tile index
    tile_idx_map: dict[tuple[int, int], int] = {}
    for j in range(len(L_col_ptr) - 1):
        for idx in range(L_col_ptr[j], L_col_ptr[j + 1]):
            tile_idx_map[(int(L_row_idx[idx]), j)] = idx

    # Scatter scalar values into tiles
    for scalar_col in range(N):
        tc = scalar_col // n
        lc = scalar_col % n
        for ptr in range(indptr[scalar_col], indptr[scalar_col + 1]):
            scalar_row = int(indices[ptr])
            tr = scalar_row // n
            lr = scalar_row % n
            val = float(data[ptr])
            # Lower triangle
            if tr >= tc:
                key = (tr, tc)
                if key in tile_idx_map:
                    tiles[tile_idx_map[key], lr, lc] += val
            # Mirror to lower if upper entry provided
            if tc >= tr and (tc, tr) != (tr, tc):
                key = (tc, tr)
                if key in tile_idx_map:
                    tiles[tile_idx_map[key], lc, lr] += val

    return tiles


# ---------------------------------------------------------------------------
# 4. Build the dispatch task schedule
# ---------------------------------------------------------------------------

def build_task_schedule(
    L_col_ptr: np.ndarray,
    L_row_idx: np.ndarray,
    tile_size: int,
) -> dict:
    """Pre-compute all offset arrays for the GPU dispatch primitive.

    Returns a dict of numpy int32 arrays ready to be passed as MLX arrays:
        potrf_offsets, trsm_l_offsets, trsm_b_offsets, trsm_col_ptr,
        syrk_c_offsets, syrk_a_offsets, syrk_col_ptr,
        gemm_c_offsets, gemm_a_offsets, gemm_b_offsets, gemm_col_ptr
    """
    n = tile_size
    nn = n * n
    Nt = len(L_col_ptr) - 1

    # Build (row, col) → tile_flat_offset lookup
    def _tile_off(tile_idx: int) -> int:
        return tile_idx * nn

    tile_idx_map: dict[tuple[int, int], int] = {}
    for j in range(Nt):
        for idx in range(L_col_ptr[j], L_col_ptr[j + 1]):
            tile_idx_map[(int(L_row_idx[idx]), j)] = idx

    # Collect per-column operations
    potrf_list: list[int] = []
    trsm_l_list: list[int] = []
    trsm_b_list: list[int] = []
    trsm_cptr: list[int] = [0]
    syrk_c_list: list[int] = []
    syrk_a_list: list[int] = []
    syrk_cptr: list[int] = [0]
    gemm_c_list: list[int] = []
    gemm_a_list: list[int] = []
    gemm_b_list: list[int] = []
    gemm_cptr: list[int] = [0]

    for j in range(Nt):
        # Diagonal tile index
        diag_idx = tile_idx_map[(j, j)]
        potrf_list.append(_tile_off(diag_idx))

        # Off-diagonal tiles in column j (rows > j)
        col_start = L_col_ptr[j]
        col_end = L_col_ptr[j + 1]
        offdiag_indices = []
        offdiag_rows = []
        for idx in range(col_start, col_end):
            r = int(L_row_idx[idx])
            if r > j:
                offdiag_indices.append(idx)
                offdiag_rows.append(r)

        # trsm: each off-diagonal tile in column j
        for idx in offdiag_indices:
            trsm_l_list.append(_tile_off(diag_idx))   # L = diagonal tile
            trsm_b_list.append(_tile_off(idx))          # B = off-diagonal tile
        trsm_cptr.append(len(trsm_l_list))

        # syrk + gemm: for each pair of off-diagonal tiles (i, k) in column j
        for ii, (idx_i, row_i) in enumerate(zip(offdiag_indices, offdiag_rows)):
            # syrk: L(row_i, row_i) -= L(row_i, j) · L(row_i, j)^T
            diag_target = tile_idx_map.get((row_i, row_i))
            if diag_target is not None:
                syrk_c_list.append(_tile_off(diag_target))
                syrk_a_list.append(_tile_off(idx_i))

            # gemm: for k > i in same column
            for idx_k, row_k in zip(
                offdiag_indices[ii + 1 :], offdiag_rows[ii + 1 :]
            ):
                # L(row_k, row_i) -= L(row_k, j) · L(row_i, j)^T
                target = tile_idx_map.get((row_k, row_i))
                if target is not None:
                    gemm_c_list.append(_tile_off(target))
                    gemm_a_list.append(_tile_off(idx_k))
                    gemm_b_list.append(_tile_off(idx_i))

        syrk_cptr.append(len(syrk_c_list))
        gemm_cptr.append(len(gemm_c_list))

    def _i32(lst):
        return np.array(lst, dtype=np.int32) if lst else np.zeros(0, dtype=np.int32)

    return {
        "num_tile_cols": Nt,
        "potrf_offsets": _i32(potrf_list),
        "trsm_l_offsets": _i32(trsm_l_list),
        "trsm_b_offsets": _i32(trsm_b_list),
        "trsm_col_ptr": _i32(trsm_cptr),
        "syrk_c_offsets": _i32(syrk_c_list),
        "syrk_a_offsets": _i32(syrk_a_list),
        "syrk_col_ptr": _i32(syrk_cptr),
        "gemm_c_offsets": _i32(gemm_c_list),
        "gemm_a_offsets": _i32(gemm_a_list),
        "gemm_b_offsets": _i32(gemm_b_list),
        "gemm_col_ptr": _i32(gemm_cptr),
    }
