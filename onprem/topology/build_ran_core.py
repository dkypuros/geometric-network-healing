"""Emit a RAN + Core + Transport topology in the shape a real operator graph takes.

Node and edge types follow the hierarchy used in published autonomous-network work:
HSS and MME on the 4G control plane, AMF, SMF and UPF on the 5G side, an aggregation
router to cell-site router transport hierarchy, eNodeBs and gNodeBs beneath it, sectors
beneath those, and H3 hexbins as the geographic surface where degradation is observed.

Edges point the way a fault propagates, cause to effect, which is the orientation that makes
the adjacency a forward operator. Published graphs often name the inverse relation
("eNodeB managed_by MME"); that is the same edge read the other way.

    python onprem/topology/build_ran_core.py --out onprem/topology/ran_core_transport.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

# relation -> coupling beta_r, mirroring onprem/ontology/ran-core-transport.ttl
COUPLING = {
    "agg_feeds_csr":        0.14,   # transport, strongest: a degraded aggregation hits everything behind it
    "csr_backhauls_enodeb": 0.13,
    "csr_backhauls_gnodeb": 0.13,
    "core_feeds_upf":       0.12,
    "enodeb_serves_sector": 0.11,
    "gnodeb_serves_sector": 0.11,
    "sector_covers_hexbin": 0.09,   # geographic surface
    "hss_authenticates_mme": 0.05,  # control plane: degrades sessions, does not remove capacity
    "mme_manages_enodeb":   0.04,
    "amf_manages_gnodeb":   0.04,
    "smf_controls_upf":     0.04,
}

# how a unit of degradation shows up in each type's KPIs, per KPI in KPI_NAMES
KPI_NAMES = ["throughput", "latency", "prb_utilization", "drop_rate"]
EFFECT = {
    "hexbin": [-1.0, 0.8, 0.3, 0.9], "sector": [-1.0, 0.7, 0.8, 0.8],
    "enodeb": [-0.8, 0.6, 0.9, 0.6], "gnodeb": [-0.8, 0.6, 0.9, 0.6],
    "csr":    [-0.4, 1.0, 0.7, 0.5], "agg":    [-0.3, 1.0, 0.8, 0.4],
    "upf":    [-0.6, 0.8, 1.0, 0.5], "mme":    [0.0, 0.9, 0.7, 0.8],
    "amf":    [0.0, 0.9, 0.7, 0.8],  "smf":    [0.0, 0.8, 0.6, 0.6],
    "hss":    [0.0, 1.0, 0.4, 0.9],
}
DOMAIN = {"hss": "Core", "mme": "Core", "amf": "Core", "smf": "Core", "upf": "Core",
          "agg": "Transport", "csr": "Transport",
          "enodeb": "RAN", "gnodeb": "RAN", "sector": "RAN", "hexbin": "Coverage"}


def build(n_agg=6, csr_per_agg=4, enb_per_csr=3, gnb_per_csr=2,
          sectors_per_nb=3, n_mme=3, n_amf=2, n_upf=3, n_hss=1, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    nodes, edges = [], []

    def add(kind: str, **extra) -> int:
        i = len(nodes)
        nodes.append({"id": i, "name": f"{kind.upper()}-{i:04d}", "type": kind,
                      "domain": DOMAIN[kind], **extra})
        return i

    def link(src: int, dst: int, rel: str) -> None:
        edges.append({"src": src, "dst": dst, "relation": rel})

    hss = [add("hss") for _ in range(n_hss)]
    mme = [add("mme") for _ in range(n_mme)]
    amf = [add("amf") for _ in range(n_amf)]
    smf = add("smf")
    upf = [add("upf") for _ in range(n_upf)]
    for m in mme:
        link(hss[rng.integers(n_hss)], m, "hss_authenticates_mme")
    for u in upf:
        link(smf, u, "smf_controls_upf")

    aggs = [add("agg") for _ in range(n_agg)]
    for a in aggs:
        link(a, upf[rng.integers(n_upf)], "core_feeds_upf")

    hexbins: dict[tuple, int] = {}
    for a in aggs:
        for _ in range(csr_per_agg):
            csr = add("csr")
            link(a, csr, "agg_feeds_csr")
            for kind, count, backhaul, control in (
                    ("enodeb", enb_per_csr, "csr_backhauls_enodeb", "mme_manages_enodeb"),
                    ("gnodeb", gnb_per_csr, "csr_backhauls_gnodeb", "amf_manages_gnodeb")):
                for _ in range(count):
                    nb = add(kind)
                    link(csr, nb, backhaul)
                    link(mme[rng.integers(n_mme)] if kind == "enodeb" else amf[rng.integers(n_amf)],
                         nb, control)
                    for _ in range(sectors_per_nb):
                        # revenue density is heavy-tailed: a few sectors carry most of it
                        sec = add("sector", revenue=float(rng.lognormal(0.0, 1.0)),
                                  profile=str(rng.choice(
                                      ["Mass-Event-Venue", "Transit-Hub",
                                       "Premium-Consumer-Zone", "Enterprise-Critical-Urban",
                                       "Residential"], p=[.06, .10, .12, .12, .60])))
                        link(nb, sec, f"{kind}_serves_sector")
                        cell = (int(rng.integers(0, 24)), int(rng.integers(0, 24)))
                        if cell not in hexbins:
                            hexbins[cell] = add("hexbin", h3=f"87{cell[0]:02x}{cell[1]:02x}ffff")
                        link(sec, hexbins[cell], "sector_covers_hexbin")

    return {"kpi_names": KPI_NAMES, "coupling": COUPLING, "effect": EFFECT,
            "rho": 0.85, "nodes": nodes, "edges": edges}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="onprem/topology/ran_core_transport.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t = build(seed=a.seed)
    Path(a.out).write_text(json.dumps(t, indent=1))
    from collections import Counter
    c = Counter(n["type"] for n in t["nodes"])
    print(f"{len(t['nodes'])} nodes, {len(t['edges'])} edges -> {a.out}")
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    print(f"  stability max(beta)/(1-rho) = {max(COUPLING.values())/(1-t['rho']):.3f}")
