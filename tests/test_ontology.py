"""The ontology is load-bearing, so it is tested like code, not like prose."""
import numpy as np
import pytest

from gnn_healing import ontology
from gnn_healing.operator import build_operator, relation_matrices, propagation_matrix, stability
from gnn_healing.twin import generate_twin
from gnn_healing.twin.schema import NodeType, Relation, RELATIONS

needs_rdflib = pytest.mark.skipif(not ontology.available(), reason="rdflib or the Turtle file is absent")


@needs_rdflib
def test_turtle_covers_every_python_relation():
    """The ontology and the enums cannot drift: every relation has a coupling, and nothing extra."""
    c = ontology.coupling_by_relation()
    assert set(c) == {r.value for r in RELATIONS}


@needs_rdflib
def test_turtle_covers_every_python_node_type():
    classes = ontology.node_classes()
    assert {t.value for t in NodeType} <= set(classes)


@needs_rdflib
def test_fallback_mirrors_the_turtle():
    """The hardcoded mirror used without rdflib must equal what the file actually says."""
    assert ontology.FALLBACK_COUPLING == ontology.coupling_by_relation()
    assert ontology.FALLBACK_SUPERPROPERTY == ontology.superproperty_by_relation()


@needs_rdflib
def test_every_relation_has_a_weight_tying_group():
    sup = ontology.superproperty_by_relation()
    assert set(sup) == {r.value for r in RELATIONS}
    assert set(sup.values()) <= {"feeds", "controls", "exposes"}


@needs_rdflib
def test_ontology_couplings_are_stable():
    """max beta_r / (1 - rho) < 1, or faults amplify per hop instead of attenuating."""
    assert ontology.stability_margin(rho=0.85) < 1.0


@needs_rdflib
def test_couplings_are_actually_heterogeneous():
    """If transport and the control plane coupled equally there would be nothing to fix."""
    c = ontology.coupling_by_relation()
    assert c["link->router"] > 2 * c["amf->gnb"]


def test_relation_matrices_sum_to_the_aggregate():
    """sum_r P_r == P. This is what makes the per-relation operator a strict generalisation."""
    tw = generate_twin(seed=3)
    total = sum(relation_matrices(tw).values())
    assert np.allclose(total, propagation_matrix(tw), atol=1e-6)


def test_equal_couplings_collapse_to_the_published_operator():
    """Per-relation with every beta_r equal must equal single-beta to floating point.
    This is why turning the feature on cannot silently move the published results."""
    tw = generate_twin(seed=4)
    single = build_operator(tw, rho=0.85, beta=0.13)
    per_rel = build_operator(tw, rho=0.85, beta_by_relation={r.value: 0.13 for r in RELATIONS})
    assert np.allclose(single, per_rel, atol=1e-6)


def test_operator_from_ontology_differs_from_single_beta():
    """And with the real heterogeneous values it must NOT be the same operator."""
    tw = generate_twin(seed=5)
    single = build_operator(tw, rho=0.85, beta=0.13)
    onto = build_operator(tw, rho=0.85, from_ontology=True)
    assert not np.allclose(single, onto, atol=1e-3)
    assert stability(0.85, beta_by_relation=ontology.coupling_by_relation()) < 1.0
