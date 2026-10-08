import torch
from gnn_healing.twin import generate_twin
from gnn_healing.gnn import HealingGNN
from gnn_healing.twin.schema import K


def test_model_is_permutation_equivariant():
    """pi . f(x, G) == f(pi . x, pi . G)  (Eq. 3.1)."""
    tw = generate_twin(seed=3)
    base = tw.to_pyg()
    W = 4
    model = HealingGNN(window=W, hidden=16, layers=2).eval()
    x = torch.randn(tw.n, W * K)
    out = model(x, base.edge_index, base.edge_type, base.node_type)
    perm = torch.randperm(tw.n)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(tw.n)
    ei = inv[base.edge_index]
    out_p = model(x[perm], ei, base.edge_type, base.node_type[perm])
    for k in ("forecast", "degradation", "rca"):
        assert torch.allclose(out[k][perm], out_p[k], atol=1e-5), k


def test_intent_ordering_depends_on_revenue():
    import numpy as np
    from gnn_healing.intent import propose_branches, select_order
    tw = generate_twin(seed=4)
    cells = tw.nodes.index[tw.nodes["type"] == "cell"].tolist()[:2]
    d_hat = np.ones(tw.n)
    tw.nodes.loc[cells[0], "revenue"] = 100.0; tw.nodes.loc[cells[1], "revenue"] = 1.0
    br = propose_branches(tw, cells, d_hat, k=2)
    order = select_order(br, budget=10)
    assert order[0].target == cells[0]  # same fault type, different response
