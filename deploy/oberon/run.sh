#!/usr/bin/env bash
# Deploy the benchmark to Oberon, follow it, and keep the output locally.
#
#   ./deploy/oberon/run.sh                 # deploy, follow, save
#   NAMESPACE=telco-ci ./deploy/oberon/run.sh
#   ./deploy/oberon/run.sh --logs-only     # re-attach to a run already going
#   ./deploy/oberon/run.sh --clean         # delete the job
#
# Never writes credentials anywhere. Results land in deploy/oberon/results/ (gitignored).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-gnn-healing}"
JOB=gnn-healing-bench
RESULTS="$HERE/results"
say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---------- preflight ----------
command -v kubectl >/dev/null || die "kubectl not found"
say "context: $(kubectl config current-context 2>/dev/null || echo none)"

if ! kubectl auth whoami >/dev/null 2>&1; then
  cat >&2 <<'MSG'

  Not authenticated to Oberon. Tokens there last about 24 hours.

  Log in, then re-run this script. In Claude Code you can run it inline with a
  leading "!" so the output lands in the conversation:

    ! oc login https://api.oberon.fm2aihpcsed.com:6443 -u dkypuros \
        --certificate-authority=$HOME/.kube/oberon-combined-ca.crt

  The combined CA has already been generated at ~/.kube/oberon-combined-ca.crt
  (oberon-ca.crt + oberon-ingress-ca.crt), which is what the OAuth redirect to
  *.apps needs.

MSG
  exit 2
fi
say "authenticated as: $(kubectl auth whoami -o jsonpath='{.status.userInfo.username}' 2>/dev/null || echo unknown)"

if [[ "${1:-}" == "--clean" ]]; then
  kubectl -n "$NAMESPACE" delete job "$JOB" --ignore-not-found
  say "deleted job $JOB in $NAMESPACE"; exit 0
fi

# ---------- namespace ----------
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  say "namespace $NAMESPACE does not exist, creating"
  kubectl create namespace "$NAMESPACE" \
    || die "cannot create namespace $NAMESPACE. Set NAMESPACE=<one you can use> and re-run."
fi
kubectl auth can-i create job -n "$NAMESPACE" >/dev/null 2>&1 \
  || die "no permission to create jobs in $NAMESPACE"

# ---------- node facts, recorded before we run ----------
mkdir -p "$RESULTS"
say "node:"
kubectl get nodes -o custom-columns=\
NAME:.metadata.name,CPU:.status.capacity.cpu,MEM:.status.capacity.memory,KERNEL:.status.nodeInfo.kernelVersion \
  | tee "$RESULTS/node.txt"

if [[ "${1:-}" != "--logs-only" ]]; then
  kubectl -n "$NAMESPACE" delete job "$JOB" --ignore-not-found >/dev/null 2>&1
  say "applying job"
  kubectl -n "$NAMESPACE" apply -f "$HERE/job.yaml" || die "apply failed"

  say "waiting for the pod to start (image pull can take a minute)"
  for _ in $(seq 1 60); do
    POD=$(kubectl -n "$NAMESPACE" get pod -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -n "${POD:-}" ]] && break
    sleep 5
  done
  [[ -n "${POD:-}" ]] || die "pod never appeared; kubectl -n $NAMESPACE describe job $JOB"
  for _ in $(seq 1 120); do
    PHASE=$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
    [[ "$PHASE" == "Running" || "$PHASE" == "Succeeded" || "$PHASE" == "Failed" ]] && break
    sleep 5
  done
  say "pod $POD is $PHASE"
  if [[ "$PHASE" == "Pending" ]]; then
    kubectl -n "$NAMESPACE" describe pod "$POD" | tail -25
    die "pod stuck pending; usually quota or image pull"
  fi
fi

POD=$(kubectl -n "$NAMESPACE" get pod -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[[ -n "${POD:-}" ]] || die "no pod for job $JOB in $NAMESPACE"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$RESULTS/bench-$STAMP.log"
say "following $POD, saving to $LOG"
kubectl -n "$NAMESPACE" logs -f "$POD" 2>&1 | tee "$LOG"

# ---------- split the two reports out of the log ----------
awk '/^# Intel benchmark:/{f=1} f' "$LOG" | awk '/^=+ pipeline/{exit} {print}' > "$RESULTS/oberon_bench-$STAMP.md" 2>/dev/null
awk '/^# Results$/{f=1} f' "$LOG" | awk '/^=+ done/{exit} {print}' > "$RESULTS/results-$STAMP.md" 2>/dev/null

say ""
say "saved:"
ls -la "$RESULTS" | tail -5
say ""
say "job state: $(kubectl -n "$NAMESPACE" get job "$JOB" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo running)"
say "clean up with: ./deploy/oberon/run.sh --clean"
