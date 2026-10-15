"""The on-premises service is the artifact someone else would run, so it is tested."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onprem"))
from rca_service.core import Topology, RootCauseEngine, healing_intent   # noqa: E402
from rca_service.simulate import inject                                   # noqa: E402

TOPO = Path(__file__).resolve().parents[1] / "onprem" / "topology" / "ran_core_transport.json"
pytestmark = pytest.mark.skipif(not TOPO.exists(), reason="run onprem/topology/build_ran_core.py")


@pytest.fixture(scope="module")
def eng():
    return RootCauseEngine(Topology.load(TOPO))


def test_topology_is_stable(eng):
    assert eng.topo.stability_margin() < 1.0


def test_refuses_an_unstable_topology():
    d = __import__("json").loads(TOPO.read_text())
    d["coupling"]["agg_feeds_csr"] = 0.9        # would amplify, not attenuate
    with pytest.raises(ValueError, match="amplify"):
        Topology.from_dict(d)


def test_refuses_a_topology_with_a_missing_coupling():
    d = __import__("json").loads(TOPO.read_text())
    d["coupling"].pop("agg_feeds_csr")
    with pytest.raises(ValueError, match="no coupling"):
        Topology.from_dict(d)


def test_locates_the_origin_at_every_layer(eng):
    for kind in ("agg", "csr", "enodeb", "gnodeb", "upf", "mme", "sector", "hss"):
        v = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == kind][0]
        X, _ = inject(eng.topo, eng.M, steps=120, origin=v, seed=11)
        assert eng.locate(X).origin_index == v, f"missed a fault at {kind}"


def test_the_origin_stands_clear_of_the_noise_floor(eng):
    v = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == "agg"][0]
    X, _ = inject(eng.topo, eng.M, steps=120, origin=v, seed=12)
    f = eng.locate(X)
    assert f.separation_z > 5.0
    assert f.margin_over_runner_up > 0.3


def test_it_is_deterministic(eng):
    v = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == "csr"][0]
    X, _ = inject(eng.topo, eng.M, steps=100, origin=v, seed=13)
    a, b = eng.locate(X), eng.locate(X)
    assert a.origin_index == b.origin_index
    assert np.allclose(a.residual, b.residual)


def test_blast_radius_is_the_forward_closure(eng):
    v = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == "agg"][0]
    reach = eng.reach(v)
    assert v not in reach                       # the closure excludes the node itself
    assert len(reach) > 50                      # an aggregation router reaches a lot
    hss = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == "hss"][0]
    assert hss not in reach                     # transport does not reach the control plane


def test_healing_intent_carries_its_own_evidence(eng):
    v = [i for i, n in enumerate(eng.topo.nodes) if n["type"] == "upf"][0]
    X, _ = inject(eng.topo, eng.M, steps=120, origin=v, seed=14)
    intent = healing_intent(eng.topo, eng.locate(X))
    assert intent["@type"] == "Intent"
    ev = intent["expression"]["evidence"]
    assert ev["method"] == "propagation-residual"
    assert ev["residual_at_origin"] > 0
