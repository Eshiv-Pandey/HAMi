#!/usr/bin/env bash
# Deploy the milestone-1 workload on a HAMi-enabled cluster.
# Prereqs: a cluster with NVIDIA GPU(s), HAMi 2.10.0 installed, Prometheus
# scraping HAMi scheduler + vGPUmonitor metrics, and kubectl pointed at it.
#
# Usage:
#   ./setup.sh                 # let HAMi binpack place the four quarter pods
#   SHARED_NODE=gpu-node-a CONTROL_NODE=gpu-node-b ./setup.sh
set -euo pipefail
NS="${NS:-hami-repro}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== creating namespace $NS =="
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# Optionally pin the workloads to specific single-GPU nodes.
apply() {
  local f="$1" node_var="$2"
  local node="${!node_var:-}"
  if [ -n "$node" ]; then
    echo "== pinning $(basename "$f") to node $node =="
    kubectl -n "$NS" apply -f <(sed "s|# nodeName: .*|nodeName: $node|" "$f")
  else
    kubectl -n "$NS" apply -f "$f"
  fi
}
apply "$DIR/00-shared-4x25.yaml" SHARED_NODE
apply "$DIR/01-control-whole-gpu.yaml" CONTROL_NODE

echo "== waiting for pods to be Ready (up to 5m) =="
kubectl -n "$NS" wait --for=condition=Ready pods --all --timeout=300s || {
  echo "!! not all pods became Ready. Inspect with: kubectl -n $NS get pods -o wide"
  echo "!! if pods are Pending, the node likely lacks nvidia.com/gpu or HAMi is not installed."
  exit 1
}

echo "== placement (the four quarter pods should share ONE device_uuid) =="
kubectl -n "$NS" get pods -o wide
echo
echo "Now let it run ~1h, then: PROM_URL=http://<prometheus>:9090 ./run.sh"
