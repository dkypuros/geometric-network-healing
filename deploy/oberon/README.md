# Running this on Oberon

Oberon is a bare-metal Single Node OpenShift cluster at the Intel Folsom Acceleration Lab.
It is the most interesting machine to run this on, because it is the configuration the
[state and coordination note](../../docs/state-and-coordination.md) argues for and the
[paper](../../docs/flatness/four_lines_one_node.pdf) depends on: a very large CPU with matrix
extensions and no GPU at all.

| | |
|---|---|
| CPU | Intel Xeon 6767P, Granite Rapids, 2 sockets by 64 cores, 256 vCPU |
| memory | 503 GiB |
| accelerators | AMX bf16 / int8 / fp16, AVX-512 VNNI, SGX, TDX |
| GPU | **none** |

## What the job does

It is deliberately self-contained and installs nothing into the cluster. One Job clones the
public repository, installs CPU-only PyTorch into a throwaway virtualenv, and runs two things:

1. **[`scripts/oberon_bench.py`](../../scripts/oberon_bench.py)**, which measures what a laptop
   cannot: the CPU model and AMX flags as the kernel reports them, thread scaling of the solver
   and the model from 1 to 64 threads, bfloat16 against float32 to see whether the matrix units
   engage, size scaling until the dense covariance becomes the wall, and a reproducibility check
   of the published numbers against different silicon.
2. **The full pipeline** at the same seed as the committed run, so the result table can be
   compared line for line with the one in the repository.

Everything goes to stdout, so `kubectl logs` is the entire result. The runner saves it locally.

## Running it

```bash
# 1. log in. Tokens on Oberon last about 24 hours, so this is usually the only manual step.
#    The combined CA is required because the OAuth redirect goes to *.apps.
oc login https://api.<your-cluster> -u <user> \
    --certificate-authority=$HOME/.kube/oberon-combined-ca.crt

# 2. deploy, follow, and save
./deploy/oberon/run.sh
```

Other modes:

```bash
NAMESPACE=telco-ci ./deploy/oberon/run.sh    # use an existing namespace instead
./deploy/oberon/run.sh --logs-only           # re-attach to a run in progress
./deploy/oberon/run.sh --clean               # delete the job
```

The runner refuses to do anything if the token has expired, and prints the login command rather
than failing obscurely. Output lands in `deploy/oberon/results/`, which is gitignored along with
the CA and any `.env`.

## What was deliberately not done

The Oberon repository's own guidance is blunt about this: *"Check what already exists before
standing something up. A third LiteLLM gateway was built without noticing two already ran."*
So this directory adds exactly one Job and nothing else.

- No new model gateway. The cluster already runs three.
- No new serving runtime. vLLM CPU and OVMS are already proven there.
- No database, broker or cache. The reasons are in the
  [state and coordination note](../../docs/state-and-coordination.md).
- No namespace-wide operators, no CRDs, no persistent volumes.

If the benchmark ever needs to serve a model rather than just measure one, the existing
`inference-proxy` in `intel-inference` is the thing to use, not a fourth gateway.

## Expected findings, written down before the run

Recording predictions first so the result can disagree with them.

| prediction | reasoning |
|---|---|
| more threads will make the model **slower** on the 74-node twin | measured on a laptop: 8 threads was 6.6x slower than 1, the graph is too small to split |
| bfloat16 **will** beat float32, unlike on the laptop | AMX exists here and that is the whole point of the comparison |
| the published numbers will reproduce exactly | fixed seeds and deterministic NumPy; a mismatch would mean a real portability bug |
| the dense covariance, not compute, will be the wall | it is O(N²) in memory and O(N³) per step |

The honest version of the last row is the one that matters. If the solver stays fast to a few
thousand nodes on this machine, the gap recorded in the state note is less urgent than I claimed.
If it falls over early, that gap is the next piece of work.

## If it fails

| symptom | likely cause |
|---|---|
| pod stuck `Pending` | namespace quota, or the image is not mirrored; try `NAMESPACE=telco-ci` |
| `git clone` hangs | no egress to github.com from the cluster; the repo would need mirroring to the in-cluster Gitea |
| `pip install` fails | no egress to pypi.org or to download.pytorch.org |
| `Unauthorized` | the token expired again; log in and re-run |
