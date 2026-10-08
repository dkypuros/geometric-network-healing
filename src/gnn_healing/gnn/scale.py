"""Scale-out path on NVIDIA hardware (documented, see docs/nvidia.md).

The reference pipeline runs full-batch on one device because the synthetic twin is small.
For operator-scale graphs the same HealingGNN runs over mini-batches produced by
cugraph-pyg's GPU neighbour sampler, with graph structure and features resident in
WholeGraph (multi-GPU, NVLink-aware). This module is a thin, import-guarded adapter: it
is exercised only when those packages are present and is not part of the CPU test suite.
"""
from __future__ import annotations


def make_cugraph_loader(data, input_nodes, batch_size: int = 1024, fanout=(15, 10, 5)):
    try:
        from cugraph_pyg.data import GraphStore, TensorDictFeatureStore
        from cugraph_pyg.loader import NeighborLoader
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the [nvidia] extra: see docs/nvidia.md") from e

    graph_store = GraphStore()
    feature_store = TensorDictFeatureStore()
    graph_store[("node", "to", "node"), "coo", False, (data.num_nodes, data.num_nodes)] = data.edge_index
    feature_store["node", "x", None] = data.x
    feature_store["node", "node_type", None] = data.node_type
    # edge_type is carried as an edge attribute through the sampler so RGCNConv keeps relations
    feature_store[("node", "to", "node"), "edge_type", None] = data.edge_type
    return NeighborLoader((feature_store, graph_store), num_neighbors=list(fanout),
                          batch_size=batch_size, input_nodes=input_nodes, shuffle=True)
