"""HTTP front end. Python standard library plus numpy, nothing else.

  GET  /healthz                 liveness
  GET  /readyz                  readiness, reports whether the operator is built
  GET  /v1/topology             what operator is loaded, and its stability margin
  POST /v1/root-cause           {"counters": [T][N][K]}  -> origin, onset, severity, residual
  POST /v1/healing-intent       same input                -> TMF921-shaped healing intent
  POST /v1/topology             replace the loaded topology at runtime

Deterministic: the same counters always produce the same answer. No model weights are loaded,
no training happens, and nothing is sent anywhere.

  python -m rca_service.server --topology onprem/topology/ran_core_transport.json --port 8080
"""
from __future__ import annotations
import argparse, json, logging, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .core import Topology, RootCauseEngine, healing_intent
from .ui import PAGE

log = logging.getLogger("rca")
STATE: dict = {"engine": None, "topology": None, "built_at": None, "build_seconds": None}
MAX_BODY = 256 * 1024 * 1024


def build(topo: Topology) -> None:
    t0 = time.perf_counter()
    eng = RootCauseEngine(topo)
    STATE.update(engine=eng, topology=topo, built_at=time.time(),
                 build_seconds=round(time.perf_counter() - t0, 3))
    log.info("operator built: %d nodes, %d edges, %d Riccati iterations, %.2fs, stability %.3f",
             topo.n, len(topo.edges), eng.gain_iters, STATE["build_seconds"],
             topo.stability_margin())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            raise ValueError("empty body")
        if n > MAX_BODY:
            raise ValueError(f"body too large: {n} bytes")
        return json.loads(self.rfile.read(n))

    def _counters(self, body: dict) -> np.ndarray:
        if "counters" not in body:
            raise ValueError("expected a 'counters' array of shape [steps][nodes][kpis]")
        x = np.asarray(body["counters"], dtype=np.float64)
        topo = STATE["topology"]
        if x.ndim != 3:
            raise ValueError(f"counters must be 3-D [steps][nodes][kpis], got shape {x.shape}")
        if x.shape[1] != topo.n:
            raise ValueError(f"counters has {x.shape[1]} nodes, topology has {topo.n}")
        if x.shape[2] != len(topo.kpi_names):
            raise ValueError(f"counters has {x.shape[2]} KPIs, topology declares "
                             f"{len(topo.kpi_names)}: {topo.kpi_names}")
        return x

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body); return
            if self.path.startswith("/v1/demo"):
                return self._demo()
            if self.path == "/healthz":
                return self._send(200, {"status": "ok"})
            if self.path == "/readyz":
                ready = STATE["engine"] is not None
                return self._send(200 if ready else 503, {"ready": ready})
            if self.path == "/v1/topology":
                t, e = STATE["topology"], STATE["engine"]
                if t is None:
                    return self._send(503, {"error": "no topology loaded"})
                from collections import Counter
                return self._send(200, {
                    "nodes": t.n, "edges": len(t.edges),
                    "node_types": dict(Counter(x["type"] for x in t.nodes)),
                    "relations": sorted({x["relation"] for x in t.edges}),
                    "coupling": t.coupling, "rho": t.rho,
                    "stability_margin": round(t.stability_margin(), 4),
                    "kpi_names": t.kpi_names,
                    "operator_build_seconds": STATE["build_seconds"],
                    "riccati_iterations": e.gain_iters,
                    "learned_parameters": 0,
                })
            return self._send(404, {"error": "not found", "paths": [
                "/healthz", "/readyz", "/v1/topology",
                "POST /v1/root-cause", "POST /v1/healing-intent", "POST /v1/topology"]})
        except Exception as exc:                                    # noqa: BLE001
            log.exception("GET failed")
            self._send(500, {"error": str(exc)})

    def _demo(self):
        """Inject a fault server-side and locate it. Synthetic counters, real solver."""
        from urllib.parse import urlparse, parse_qs
        from .simulate import inject
        eng, topo = STATE["engine"], STATE["topology"]
        if eng is None:
            return self._send(503, {"error": "no topology loaded"})
        kind = (parse_qs(urlparse(self.path).query).get("kind") or ["agg"])[0]
        cand = [i for i, n in enumerate(topo.nodes) if n["type"] == kind] or [0]
        v = int(np.random.default_rng().choice(cand))
        X, _ = inject(topo, eng.M, steps=120, origin=v,
                      seed=int(time.time()) % 100000)
        t0 = time.perf_counter(); f = eng.locate(X); el = time.perf_counter() - t0
        return self._send(200, {
            "injected": topo.name(v), "found": f.origin_name, "origin_index": f.origin_index,
            "confidence": f.confidence, "separation_z": f.separation_z,
            "margin": f.margin_over_runner_up, "blast_radius": len(f.blast_radius),
            "n_nodes": topo.n, "steps": int(X.shape[0]),
            "us_per_step": round(el * 1e6 / X.shape[0], 1),
            "residual": [float(x) for x in f.residual],
            "top": [{"node": topo.name(i), "value": float(x)} for i, x in f.ranked[:8]],
            "intent": healing_intent(topo, f),
        })

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == "/v1/topology":
                build(Topology.from_dict(body))
                return self._send(200, {"loaded": True, "nodes": STATE["topology"].n,
                                        "build_seconds": STATE["build_seconds"]})
            if STATE["engine"] is None:
                return self._send(503, {"error": "no topology loaded"})
            if self.path in ("/v1/root-cause", "/v1/healing-intent"):
                x = self._counters(body)
                t0 = time.perf_counter()
                f = STATE["engine"].locate(x, at_step=body.get("at_step"))
                elapsed = time.perf_counter() - t0
                if self.path == "/v1/healing-intent":
                    return self._send(200, healing_intent(STATE["topology"], f,
                                                          body.get("intent_id")))
                out = f.as_dict(STATE["topology"], top_k=int(body.get("top_k", 5)))
                out["timing"] = {"total_ms": round(elapsed * 1e3, 3),
                                 "per_step_us": round(elapsed * 1e6 / x.shape[0], 1),
                                 "steps": int(x.shape[0])}
                return self._send(200, out)
            return self._send(404, {"error": "not found"})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:                                    # noqa: BLE001
            log.exception("POST failed")
            self._send(500, {"error": str(exc)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rca-service")
    ap.add_argument("--topology", default=os.environ.get("TOPOLOGY"))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.topology:
        build(Topology.load(a.topology))
    else:
        log.warning("started with no topology; POST one to /v1/topology")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: srv.shutdown())
    log.info("listening on %s:%d", a.host, a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
