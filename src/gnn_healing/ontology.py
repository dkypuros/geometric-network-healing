"""The ontology, read as mathematics rather than as documentation.

`docs/ontology/network-healing.ttl` declares the node classes, the directed propagation
relations, and a `gnh:coupling` value on each relation. That coupling is beta_r in

    M  =  rho I  +  sum_r  beta_r P_r,          P_r = D_in^-1 A_r^T,   sum_r P_r = P

so the ontology is what makes per-relation coupling expressible at all. With every beta_r
equal the sum collapses to the single-beta operator of Chapter 4 exactly, which is why the
published results are unaffected.

The property hierarchy is load-bearing too: `feeds` and `controls` group relations that
should share structure, which is the natural basis for tying a relational model's weight
matrices and for giving an unseen relation an inherited prior.

rdflib is an optional dependency. Without it the hardcoded mirror below is used and
`tests/test_ontology.py` is skipped; with it, that test asserts the two cannot drift.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

TTL_PATH = Path(__file__).resolve().parents[2] / "docs" / "ontology" / "network-healing.ttl"
NS = "https://github.com/dkypuros/geometric-network-healing/ontology#"

# Mirror of the coupling values in the Turtle file, used when rdflib is unavailable.
# tests/test_ontology.py asserts this equals what the Turtle actually says.
FALLBACK_COUPLING: dict[str, float] = {
    "link->router": 0.14,
    "router(core)->router(agg)": 0.13,
    "router(agg)->gnb": 0.12,
    "router(core)->upf": 0.12,
    "gnb->cell": 0.11,
    "upf->service": 0.10,
    "cell->service": 0.06,
    "amf->gnb": 0.04,
    "smf->upf": 0.04,
}

FALLBACK_SUPERPROPERTY: dict[str, str] = {
    "link->router": "feeds", "router(core)->router(agg)": "feeds",
    "router(agg)->gnb": "feeds", "router(core)->upf": "feeds", "gnb->cell": "feeds",
    "upf->service": "exposes", "cell->service": "exposes",
    "amf->gnb": "controls", "smf->upf": "controls",
}


def available() -> bool:
    try:
        import rdflib  # noqa: F401
        return TTL_PATH.exists()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def graph():
    """Parsed RDF graph, or None when rdflib or the file is missing."""
    if not available():
        return None
    import rdflib
    g = rdflib.Graph()
    g.parse(TTL_PATH, format="turtle")
    return g


def _query(q: str) -> list[tuple]:
    g = graph()
    return [] if g is None else [tuple(r) for r in g.query(q)]


@lru_cache(maxsize=1)
def coupling_by_relation() -> dict[str, float]:
    """beta_r keyed by the Python relation value, read from the Turtle."""
    rows = _query(f"""
        PREFIX gnh: <{NS}>
        SELECT ?name ?beta WHERE {{ ?rel gnh:pythonName ?name ; gnh:coupling ?beta . }}
    """)
    return {str(n): float(b) for n, b in rows} if rows else dict(FALLBACK_COUPLING)


@lru_cache(maxsize=1)
def superproperty_by_relation() -> dict[str, str]:
    """Immediate superproperty of each relation: the weight-tying / inheritance group."""
    rows = _query(f"""
        PREFIX gnh:  <{NS}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?name ?parent WHERE {{
            ?rel gnh:pythonName ?name ; rdfs:subPropertyOf ?parent .
            ?parent rdfs:subPropertyOf gnh:propagatesTo .
        }}
    """)
    return {str(n): str(p).split("#")[-1] for n, p in rows} if rows else dict(FALLBACK_SUPERPROPERTY)


@lru_cache(maxsize=1)
def node_classes() -> dict[str, dict]:
    """Python node-type value -> {class, parents, domain}. Empty without rdflib."""
    rows = _query(f"""
        PREFIX gnh:  <{NS}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?name ?cls ?parent ?dom WHERE {{
            ?cls gnh:pythonName ?name ; rdfs:subClassOf ?parent .
            OPTIONAL {{ ?parent gnh:domainName ?dom }}
        }}
    """)
    out: dict[str, dict] = {}
    for name, cls, parent, dom in rows:
        e = out.setdefault(str(name), {"class": str(cls).split("#")[-1], "parents": [], "domain": None})
        e["parents"].append(str(parent).split("#")[-1])
        if dom is not None:
            e["domain"] = str(dom)
    return out


def coupling_vector(relations) -> list[float]:
    """beta_r in the order of the given relation enum members."""
    c = coupling_by_relation()
    return [c[r.value] for r in relations]


def stability_margin(rho: float, relations=None) -> float:
    """max_r beta_r / (1 - rho). Faults attenuate per hop iff this is below 1."""
    c = coupling_by_relation()
    vals = list(c.values()) if relations is None else [c[r.value] for r in relations]
    return max(vals) / (1.0 - rho)


def summary() -> str:
    c, sup = coupling_by_relation(), superproperty_by_relation()
    src = "docs/ontology/network-healing.ttl" if available() else "hardcoded fallback (rdflib absent)"
    lines = [f"ontology source: {src}", f"{len(c)} relations, coupling beta_r:"]
    for rel in sorted(c, key=lambda k: -c[k]):
        lines.append(f"  {c[rel]:.3f}  {rel:<28} ({sup.get(rel, '?')})")
    lines.append(f"spread {min(c.values()):.3f} to {max(c.values()):.3f}, "
                 f"ratio {max(c.values())/min(c.values()):.1f}x")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
