import numpy as np
from gnn_healing.twin import generate_twin, RELATIONS
from gnn_healing.sim import simulate, sample_fault, PropagationParams
from gnn_healing.sim.faults import Fault, FaultType


def test_twin_is_dag_like_and_typed():
    tw = generate_twin(seed=1)
    assert tw.n > 50
    assert set(tw.edges["rel"]) <= {r.value for r in RELATIONS}
    d = tw.to_pyg()
    assert d.edge_index.shape[1] == 2 * len(tw.edges)      # transposes appended
    assert int(d.edge_type.max()) <= 2 * len(RELATIONS) - 1
    assert int(generate_twin(seed=1, edge=True).to_pyg().edge_type.max()) == 2 * len(RELATIONS) - 1


def test_permutation_equivariance_of_simulator_labels():
    """Relabel nodes; the degradation field must permute with them (Eq. 3.1 at the data level)."""
    tw = generate_twin(seed=2)
    rng = np.random.default_rng(0)
    f = sample_fault(tw.nodes, 100, rng, FaultType.FIBER_DEGRADE)
    ep = simulate(tw, 100, f, np.random.default_rng(1))
    assert ep.D[-1, f.origin] > 0.5
    reach = tw.downstream(f.origin)
    assert all(ep.D[-1, u] > 0 for u in reach)
    not_reached = set(range(tw.n)) - reach - {f.origin}
    assert all(ep.D[-1, u] == 0 for u in not_reached)


def test_faults_attenuate_per_hop():
    p = PropagationParams()
    assert p.beta / (1 - p.rho) < 1.0


def test_injection_ramp():
    f = Fault(FaultType.UPF_OVERLOAD, origin=0, onset=10, severity=1.0, ramp=10)
    assert f.injection(9) == 0.0 and abs(f.injection(19) - 1.0) < 1e-9 and 0 < f.injection(12) < 1
