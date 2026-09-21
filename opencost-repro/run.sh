#!/usr/bin/env bash
# Query Prometheus and print the milestone-1 tables. Self-contained: expands the
# recording-rule expressions inline, so no Prometheus config change is needed.
#
# Usage:
#   PROM_URL=http://localhost:9090 ./run.sh
#   PROM_URL=http://localhost:9090 NS=hami-repro WINDOW=1h GPU_PRICE=3 ./run.sh
#
# Pricing: core and memory priced separately and summed. By default the card
# price GPU_PRICE is split evenly (core_price = mem_price = GPU_PRICE/2), so a
# fully reserved card reconciles to GPU_PRICE.
set -euo pipefail
: "${PROM_URL:?set PROM_URL, e.g. http://localhost:9090}"
NS="${NS:-hami-repro}"
WINDOW="${WINDOW:-1h}"
GPU_PRICE="${GPU_PRICE:-3}"
CORE_PRICE="${CORE_PRICE:-$(python3 -c "print($GPU_PRICE/2)")}"
MEM_PRICE="${MEM_PRICE:-$(python3 -c "print($GPU_PRICE/2)")}"

q() {
  # Run an instant query and print "value  {labels}" lines.
  local expr="$1"
  curl -sG "$PROM_URL/api/v1/query" --data-urlencode "query=$expr" \
    | python3 -c '
import json,sys
d=json.load(sys.stdin)
if d.get("status")!="success":
    print("  query error:", d.get("error","?")); sys.exit(0)
r=d["data"]["result"]
if not r: print("  (no data)"); sys.exit(0)
for s in sorted(r,key=lambda x:json.dumps(x["metric"],sort_keys=True)):
    m=s["metric"]; lbl=",".join(f"{k}={v}" for k,v in sorted(m.items()) if k!="__name__")
    print(f"  {float(s[\"value\"][1]):.4f}  {{{lbl}}}")
'
}

NSF="namespace=\"$NS\""
TOTAL="sum by (device_uuid) (hami_gpu_memory_allocated_bytes) / on (device_uuid) sum by (device_uuid) (hami_node_gpu_memory_allocated_ratio)"

echo "############ HAMi + OpenCost milestone-1 report ############"
echo "PROM_URL=$PROM_URL  NS=$NS  WINDOW=$WINDOW  GPU_PRICE=$GPU_PRICE (core=$CORE_PRICE mem=$MEM_PRICE)"
echo
echo "== [A2] shared-card total reserved core fraction (expect 1.0 on the shared card) =="
q "sum by (device_uuid) (hami_vgpu_core_allocated_ratio{$NSF}) / 100"
echo
echo "== [A1] per-pod reserved core fraction =="
q "sum by (namespace, pod, device_uuid) (hami_vgpu_core_allocated_ratio{$NSF}) / 100"
echo
echo "== [C1] per-pod reserved memory fraction =="
q "sum by (namespace, pod, device_uuid) (hami_vgpu_memory_allocated_bytes{$NSF}) / on (device_uuid) group_left () ($TOTAL)"
echo
echo "== [B1] per-pod MEASURED compute fraction over $WINDOW (runtime) =="
q "sum by (namespace, pod, device_uuid) (avg_over_time(hami_container_device_utilization_ratio{$NSF}[$WINDOW])) / 100"
echo
echo "== per-pod cost/hr (reservation basis: core_price*core_frac + mem_price*mem_frac) =="
q "$CORE_PRICE * (sum by (namespace, pod, device_uuid) (hami_vgpu_core_allocated_ratio{$NSF}) / 100) + on (namespace, pod, device_uuid) $MEM_PRICE * (sum by (namespace, pod, device_uuid) (hami_vgpu_memory_allocated_bytes{$NSF}) / on (device_uuid) group_left () ($TOTAL))"
echo
echo "== RECONCILIATION: sum(pod_cost) + idle_cost per card (expect == $GPU_PRICE) =="
q "sum by (device_uuid) ($CORE_PRICE * (sum by (namespace, pod, device_uuid) (hami_vgpu_core_allocated_ratio{$NSF}) / 100) + on (namespace, pod, device_uuid) $MEM_PRICE * (sum by (namespace, pod, device_uuid) (hami_vgpu_memory_allocated_bytes{$NSF}) / on (device_uuid) group_left () ($TOTAL))) + on (device_uuid) ($CORE_PRICE * (1 - clamp_max(sum by (device_uuid) (hami_vgpu_core_allocated_ratio{$NSF}) / 100, 1)) + on (device_uuid) $MEM_PRICE * (1 - clamp_max(sum by (device_uuid) (hami_node_gpu_memory_allocated_ratio), 1)))"
echo
echo "############ end ############"
