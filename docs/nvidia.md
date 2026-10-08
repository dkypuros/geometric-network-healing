# NVIDIA stack: assumptions, install, and what each piece does

This repository assumes NVIDIA hardware and the RAPIDS ecosystem for anything beyond the
laptop-sized reference twin. The reference pipeline runs on CPU so the math and the results are
checkable by anyone; this page documents the accelerated path.

## Hardware assumed

| Tier | Hardware | What it buys |
|---|---|---|
| Development | one RTX or L-series GPU, CUDA 12 driver | full-batch training on twins up to ~10^5 nodes in GPU memory |
| Regional twin | one DGX (8× H100/H200 or B200, NVLink) | WholeGraph shards structure + KPI windows across GPUs; sampling never touches host memory |
| National twin | multi-node DGX (NVSwitch / InfiniBand) | WholeGraph distributed memory; sampler sharded by seed nodes; DDP training |

## Software and what it replaces

| Reference path (this repo, CPU) | Accelerated path | Role |
|---|---|---|
| `pandas` tables in `twin/generator.py` | **cuDF** | build node/edge tables and rolling KPI windows on GPU |
| full-batch `Data` in `twin/to_pyg()` | **WholeGraph** (`pylibwholegraph`) | GPU-resident, multi-GPU graph structure and feature store |
| full-batch forward in `gnn/train.py` | **cuGraph-PyG** `NeighborLoader` over `GraphStore`/`FeatureStore` | GPU neighbour sampling; mini-batches feed the same `HealingGNN` |
| stock PyG `RGCNConv` | stock PyG `RGCNConv` (+ NVIDIA accelerated kernels where available) | relational message passing, Eq. 3.4 |
| `torch` CPU | `torch` CUDA, `torch.distributed` DDP | training |

Notes on names you may see elsewhere:
- `cugraph-ops` (the older standalone kernel package) was deprecated; use `cugraph-pyg` with the
  standard PyG convolution classes. Older examples with `CuGraphSAGEConv` / `CuGraphNeighborLoader`
  reflect that earlier API.
- `WholeGraph` is a memory and sampling layer, not a database. The durable, versioned twin still
  lives in a graph store of your choice; WholeGraph holds the working set.
- `Modulus`/`PhysicsNeMo` and `BioNeMo` are not needed here. There is no 3D geometry in a telecom
  topology graph; the symmetry is permutation only (Chapter 3 of the monograph).

## Install (Linux x86_64 or aarch64, CUDA 12.x driver, Python 3.11-3.12)

```bash
make setup                                   # CPU baseline env via uv
uv pip install -r requirements-nvidia.txt --extra-index-url https://pypi.nvidia.com
# or: make setup-nvidia
uv run gnn-healing backend                   # prints torch/cuda/cugraph_pyg/wholegraph versions
```

The RAPIDS packages follow a `YY.MM` release train; keep `cugraph-pyg-cu12`, `pylibwholegraph-cu12`
and `cudf-cu12` on the same release. Match the PyTorch CUDA build to the RAPIDS CUDA major
(cu12). The NGC PyG container (`nvcr.io/nvidia/pyg`) ships all of this preinstalled and is the
fastest route on DGX.

## Running at scale

`src/gnn_healing/gnn/scale.py::make_cugraph_loader(data, input_nodes, batch_size, fanout)` wraps
the same homogeneous `Data` (with `node_type` and `edge_type`) in a cuGraph-PyG
`GraphStore`/`FeatureStore` and returns a GPU `NeighborLoader`. Training then iterates the loader
instead of the full-batch `DataLoader` in `gnn/train.py`. Everything else (model, loss, detection,
RCA, intent, orchestration) is unchanged.

What changes mathematically: the mean over `N_r(v)` in Eq. 3.4 becomes a mean over a sampled
subset, an unbiased estimate; equivariance holds in distribution because the sampler is index-blind.

## Honesty note

The cuGraph-PyG / WholeGraph path is **documented and import-guarded, not CI-tested** in this
repository. The public release was built and tested on a machine without an NVIDIA GPU. The
reference CPU path is fully tested. If you run the accelerated path, please open an issue with
`gnn-healing backend` output and the results table so it can be recorded.
