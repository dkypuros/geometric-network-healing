import numpy as np, torch
from gnn_healing.twin import generate_twin
from gnn_healing.gnn import HealingGNN
from gnn_healing.gnn.edge import site_partition
from gnn_healing.intent import propose_branches
from gnn_healing.twin.schema import K


def test_edge_twin_has_psa_ladn_per_site():
    tw = generate_twin(seed=1, edge=True)
    assert len(tw.sites) == 6
    assert (tw.nodes["type"] == "psa").sum() == 6 and (tw.nodes["type"] == "ladn").sum() == 6
    psa = tw.nodes.index[tw.nodes["type"] == "psa"].tolist()
    assert tw.backup_psa(psa[0]) == psa[1]
    # every site's PSA reaches its LADN service, and nothing in another site
    for p in psa:
        s = int(tw.nodes.loc[p, "site"])
        reach_sites = {int(tw.nodes.loc[u, "site"]) for u in tw.downstream(p)}
        assert reach_sites <= {s, -1}


def test_partition_covers_every_node_once():
    tw = generate_twin(seed=2, edge=True)
    parts = site_partition(tw, tw.to_pyg())
    owned = np.concatenate([p["owned"] for p in parts.values()])
    assert len(owned) == len(set(owned))
    regional = set(tw.nodes.index[tw.nodes["site"] < 0])
    assert set(owned) | regional == set(range(tw.n))
    model = HealingGNN(window=4, hidden=16, layers=2).eval()
    p = parts[0]
    out = model(torch.randn(len(p["nodes"]), 4 * K), p["edge_index"], p["edge_type"], p["node_type"])
    assert out["rca"].shape[0] == len(p["nodes"])


def test_psa_relocation_cost_follows_ssc_mode():
    tw = generate_twin(seed=3, edge=True)
    psa = tw.nodes.index[tw.nodes["type"] == "psa"].tolist()[:1]
    d = np.ones(tw.n)
    b3 = propose_branches(tw, psa, d, k=1, ssc_mode=3)[0]
    b1 = propose_branches(tw, psa, d, k=1, ssc_mode=1)[0]
    assert b3.resource_units < b1.resource_units
    assert "backup" in b3.action and "re-anchor" in b3.detail
