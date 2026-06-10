"""Graph-level metadata feature computation for dataset analysis.

All metrics are computed on **undirected** graphs.  Self-loops are removed
before any computation.

Notation (per graph):
    n = |V|  (number of nodes)
    m = |E|  (number of edges, undirected, no self-loops)

Feature catalogue
-----------------
Global (per split):
    num_graphs              number of graphs in the split
    total_num_nodes         sum of n across the split

Per-graph → reduced to (mean, std) across the split:
    num_nodes               n
    num_edges               m
    edge_density            2m / (n(n−1))
    avg_degree              2m / n
    degree_assortativity    Pearson r of (deg(u), deg(v)) over edges
    pseudo_diameter         BFS-based lower bound on graph diameter
    avg_clustering_coef     mean local clustering coefficient
    transitivity            3·#triangles / #triads
    degeneracy              max k-core number
    gini_degree             Gini coefficient of degree sequence
    spectral_gap            algebraic connectivity (2nd-smallest Laplacian eigenval)
    spectral_radius         largest adjacency-matrix eigenvalue
    num_connected_components number of connected components

Scalar (mean across graphs, no std):
    attribute_assortativity Pearson r of scalar node attribute over edges;
                            uses data.x[:, 0] if available, else node degree.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_undirected_nx(data) -> nx.Graph:
    """Convert a PyG Data object to a plain undirected NetworkX Graph.

    Self-loops are removed so that all topology metrics operate on a clean
    simple graph.  Multi-edges that arise from parallel directed edges in the
    PyG edge_index are collapsed automatically by the undirected conversion.
    """
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _safe(value: Any, fallback: float = float("nan")) -> float:
    """Return a finite float, or *fallback* if value is NaN/Inf/None."""
    if value is None:
        return fallback
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _gini(values: list[float]) -> float:
    """Gini coefficient of a sequence of non-negative values."""
    arr = np.sort(np.abs(np.array(values, dtype=float)))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def _pseudo_diameter(G: nx.Graph) -> float:
    """BFS-based pseudo-diameter (lower bound on exact diameter).

    For each connected component the approximation proceeds in two BFS passes:
      1. From an arbitrary seed u, find the farthest node v.
      2. From v, find the farthest distance d.
    The pseudo-diameter is the maximum d over all components.

    Returns 0 for the trivial graph (n ≤ 1).
    """
    if G.number_of_nodes() <= 1:
        return 0.0

    max_d = 0
    for component in nx.connected_components(G):
        if len(component) == 1:
            continue
        sub = G.subgraph(component)
        seed = next(iter(component))
        lengths_from_seed = nx.single_source_shortest_path_length(sub, seed)
        far_node = max(lengths_from_seed, key=lengths_from_seed.get)
        lengths_from_far = nx.single_source_shortest_path_length(sub, far_node)
        max_d = max(max_d, max(lengths_from_far.values()))

    return float(max_d)


def _degeneracy(G: nx.Graph) -> float:
    """K-core degeneracy: the maximum core number across all nodes."""
    if G.number_of_nodes() == 0:
        return 0.0
    core_numbers = nx.core_number(G)
    return float(max(core_numbers.values())) if core_numbers else 0.0


def _spectral_gap(G: nx.Graph) -> float:
    """Algebraic connectivity (2nd-smallest eigenvalue of the Laplacian).

    Equals 0 for disconnected graphs.  Uses dense eigen-decomposition for
    small graphs (n ≤ 500) and the sparse shift-invert method otherwise.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    if n <= 500:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigs = np.linalg.eigvalsh(L)
        eigs_sorted = np.sort(eigs)
        return float(eigs_sorted[1]) if n >= 2 else 0.0
    else:
        from scipy.sparse.linalg import eigsh
        L = nx.laplacian_matrix(G).astype(float)
        try:
            vals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-4)
            return float(np.sort(vals)[1])
        except Exception:
            return float("nan")


def _spectral_radius(G: nx.Graph) -> float:
    """Largest eigenvalue of the adjacency matrix."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    if n <= 500:
        A = nx.adjacency_matrix(G).toarray().astype(float)
        eigs = np.linalg.eigvalsh(A)
        return float(np.max(eigs))
    else:
        from scipy.sparse.linalg import eigsh
        A = nx.adjacency_matrix(G).astype(float)
        try:
            vals = eigsh(A, k=1, which="LM", return_eigenvectors=False, tol=1e-4)
            return float(vals[0])
        except Exception:
            return float("nan")


def _attribute_assortativity(G: nx.Graph, data) -> float | None:
    """Numeric assortativity coefficient using data.x[:, 0] as node attribute.

    Falls back to degree assortativity when node features are absent.
    Returns None for graphs with no edges (assortativity undefined).
    """
    if G.number_of_edges() == 0:
        return None

    if data.x is not None and data.x.numel() > 0:
        x_col = data.x[:, 0].detach().float().numpy()
        for i in G.nodes():
            G.nodes[i]["_attr"] = float(x_col[i]) if i < len(x_col) else 0.0
        attr_name = "_attr"
    else:
        for i, deg in G.degree():
            G.nodes[i]["_attr"] = float(deg)
        attr_name = "_attr"

    try:
        return float(nx.numeric_assortativity_coefficient(G, attr_name))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_graph_stats(data, enabled: dict) -> dict[str, float | None]:
    """Compute all enabled per-graph statistics for a single PyG Data object.

    Parameters
    ----------
    data:
        A ``torch_geometric.data.Data`` graph.
    enabled:
        Dict mapping feature name → bool.  Only enabled features are computed.

    Returns
    -------
    dict
        Scalar statistics for this graph.  Values may be NaN/None when
        the metric is undefined (e.g., no edges, single node, etc.).
    """
    G = _to_undirected_nx(data)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats: dict[str, float | None] = {}

    # ── Basic topology ───────────────────────────────────────────────────────
    if enabled.get("num_nodes", True):
        stats["num_nodes"] = float(n)

    if enabled.get("num_edges", True):
        stats["num_edges"] = float(m)

    if enabled.get("edge_density", True):
        stats["edge_density"] = _safe(2 * m / (n * (n - 1))) if n > 1 else 0.0

    if enabled.get("avg_degree", True):
        stats["avg_degree"] = _safe(2 * m / n) if n > 0 else 0.0

    if enabled.get("degree_assortativity", True):
        if m == 0:
            stats["degree_assortativity"] = float("nan")
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stats["degree_assortativity"] = _safe(
                        nx.degree_assortativity_coefficient(G)
                    )
            except Exception:
                stats["degree_assortativity"] = float("nan")

    # ── Distance ─────────────────────────────────────────────────────────────
    if enabled.get("pseudo_diameter", True):
        stats["pseudo_diameter"] = _pseudo_diameter(G)

    # ── Clustering ───────────────────────────────────────────────────────────
    if enabled.get("avg_clustering_coef", True):
        stats["avg_clustering_coef"] = _safe(nx.average_clustering(G))

    if enabled.get("transitivity", True):
        stats["transitivity"] = _safe(nx.transitivity(G))

    if enabled.get("degeneracy", True):
        stats["degeneracy"] = _degeneracy(G)

    # ── Degree distribution ───────────────────────────────────────────────────
    if enabled.get("gini_degree", True):
        degrees = [deg for _, deg in G.degree()]
        stats["gini_degree"] = _gini(degrees)

    # ── Spectral ──────────────────────────────────────────────────────────────
    if enabled.get("spectral_gap", True):
        stats["spectral_gap"] = _safe(_spectral_gap(G))

    if enabled.get("spectral_radius", True):
        stats["spectral_radius"] = _safe(_spectral_radius(G))

    if enabled.get("num_connected_components", True):
        stats["num_connected_components"] = float(nx.number_connected_components(G))

    # ── Attribute assortativity ───────────────────────────────────────────────
    if enabled.get("attribute_assortativity", True):
        stats["attribute_assortativity"] = _attribute_assortativity(G, data)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Split-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

_NAN_POLICIES = {"nan": None, "inf": None}

def _to_json_safe(v: float | None) -> float | None:
    """Convert NaN/Inf to None for JSON serialisation."""
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def compute_split_features(
    data_list: list,
    features_config: dict,
    split_name: str = "",
    verbose: bool = True,
) -> dict:
    """Aggregate per-graph statistics into split-level features.

    Parameters
    ----------
    data_list:
        List of ``torch_geometric.data.Data`` objects (one per graph).
    features_config:
        Dict from ``config.yaml`` ``features`` section.
    split_name:
        Label used in progress output (e.g. "train", "val", "test").
    verbose:
        Print progress every 10 % of graphs.

    Returns
    -------
    dict
        29-feature dict as described in the module docstring.
    """
    n_graphs = len(data_list)
    step = max(1, n_graphs // 10)

    per_graph: list[dict] = []
    for idx, data in enumerate(data_list):
        if verbose and (idx % step == 0 or idx == n_graphs - 1):
            pct = int(100 * (idx + 1) / n_graphs)
            print(f"    [{split_name}] {pct:3d}%  ({idx + 1}/{n_graphs})", end="\r")
        per_graph.append(compute_per_graph_stats(data, features_config))

    if verbose:
        print()  # newline after \r progress

    result: dict = {}

    # ── Global counts ─────────────────────────────────────────────────────────
    if features_config.get("num_graphs", True):
        result["num_graphs"] = n_graphs

    if features_config.get("total_num_nodes", True):
        result["total_num_nodes"] = int(
            sum(s.get("num_nodes", 0) or 0 for s in per_graph)
        )

    # ── Per-graph stats → (mean, std) ─────────────────────────────────────────
    _moment_keys = [
        "num_nodes",
        "num_edges",
        "edge_density",
        "avg_degree",
        "degree_assortativity",
        "pseudo_diameter",
        "avg_clustering_coef",
        "transitivity",
        "degeneracy",
        "gini_degree",
        "spectral_gap",
        "spectral_radius",
        "num_connected_components",
    ]

    for key in _moment_keys:
        if not features_config.get(key, True):
            continue
        values = [
            v for s in per_graph
            if (v := s.get(key)) is not None and math.isfinite(v)
        ]
        if values:
            result[f"{key}_mean"] = _to_json_safe(float(np.mean(values)))
            result[f"{key}_std"] = _to_json_safe(float(np.std(values, ddof=0)))
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"] = None

    # ── Attribute assortativity (single mean, no std) ─────────────────────────
    if features_config.get("attribute_assortativity", True):
        aa_values = [
            v for s in per_graph
            if (v := s.get("attribute_assortativity")) is not None
            and math.isfinite(v)
        ]
        result["attribute_assortativity"] = (
            _to_json_safe(float(np.mean(aa_values))) if aa_values else None
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Property shift (train → test distribution distance)
# ─────────────────────────────────────────────────────────────────────────────

_BOUNDED_PROPERTIES = [
    "edge_density",
    "avg_clustering_coef",
    "transitivity",
    "gini_degree",
    "degree_assortativity",
]  # already in ~[0,1] or [-1,1]; raw absolute difference is interpretable

_UNBOUNDED_PROPERTIES = [
    "num_nodes",
    "num_edges",
    "avg_degree",
    "pseudo_diameter",
    "spectral_gap",
    "spectral_radius",
    "degeneracy",
]  # normalised by pooled std to make scale-invariant


def compute_property_shift(train_stats: dict, test_stats: dict) -> float | None:
    """Compute a crude overall property-shift score between two split feature dicts.

    For each property the per-split dicts must contain ``{property}_mean`` (and
    ``{property}_std`` for unbounded properties).  Properties whose values are
    missing or None in either split are silently skipped.

    Bounded properties  (edge_density, avg_clustering_coef, transitivity,
    gini_degree, degree_assortativity):
        shift = |test_mean − train_mean|

    Unbounded properties  (num_nodes, num_edges, avg_degree, pseudo_diameter,
    spectral_gap, spectral_radius, degeneracy):
        pooled_std = sqrt((train_std² + test_std²) / 2)
        shift = |test_mean − train_mean| / max(pooled_std, 1e-6)

    Returns the **median** of all valid per-property shifts, or ``None`` if no
    property could be evaluated.
    """
    shifts: list[float] = []

    for p in _BOUNDED_PROPERTIES:
        tr = train_stats.get(f"{p}_mean")
        te = test_stats.get(f"{p}_mean")
        if tr is None or te is None:
            continue
        try:
            shifts.append(abs(float(te) - float(tr)))
        except (TypeError, ValueError):
            continue

    for p in _UNBOUNDED_PROPERTIES:
        tr_mean = train_stats.get(f"{p}_mean")
        te_mean = test_stats.get(f"{p}_mean")
        tr_std  = train_stats.get(f"{p}_std")
        te_std  = test_stats.get(f"{p}_std")
        if tr_mean is None or te_mean is None or tr_std is None or te_std is None:
            continue
        try:
            pooled_std = math.sqrt((float(tr_std) ** 2 + float(te_std) ** 2) / 2)
            shift = abs(float(te_mean) - float(tr_mean)) / max(pooled_std, 1e-6)
            shifts.append(shift)
        except (TypeError, ValueError):
            continue

    if not shifts:
        return None
    return _to_json_safe(float(np.median(shifts)))
