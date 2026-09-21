# HAMi + OpenCost fractional GPU accounting: milestone-1 reproduction

Version-pinned harness for issue
[#3073](https://github.com/Project-HAMi/HAMi/issues/3073) and OpenCost
[#3828](https://github.com/opencost/opencost/issues/3828).

Goal: show that four pods each reserving 25% of one physical GPU account for
**one** GPU-hour over an hour under a core-reservation policy, with a whole-GPU
pod as the control, and that per-pod charges plus idle reconcile to the physical
card's cost.

## Pinned versions

- HAMi chart / image: `2.10.0` (`charts/hami/Chart.yaml`, `projecthami/hami`).
- Workload image: `nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda11.7.1-ubi8`.
- Record the OpenCost and kube-state-metrics image digests you deploy with here
  when you run it.

## Layout

- `00-shared-4x25.yaml` - four quarter-GPU pods, co-located on one card.
- `01-control-whole-gpu.yaml` - one whole-GPU control pod.
- `queries.promql` - accounting and reconciliation queries, with verified units.

## Run (on a HAMi-enabled GPU cluster)

Two helper scripts wrap the whole flow:

```
# 1. deploy the workload (optionally pin to specific single-GPU nodes)
SHARED_NODE=gpu-a CONTROL_NODE=gpu-b ./setup.sh

# 2. after ~1h, print the tables (queries Prometheus directly, no rule install)
PROM_URL=http://<prometheus>:9090 ./run.sh
```

`run.sh` prints the shared-card total (expect 1.0), per-pod core and memory
fractions, per-pod measured utilization, per-pod cost, and the reconciliation
`sum(pod_cost) + idle == GPU_PRICE` per card. Prices default to an even split of
`GPU_PRICE` (default 3); override with `CORE_PRICE` / `MEM_PRICE`.

Manual alternative: `kubectl apply -f 00-shared-4x25.yaml -f 01-control-whole-gpu.yaml`
then evaluate the queries in `queries.promql` in the Prometheus UI.

## What to report back

Post both bases side by side, per the proposal:

- Reservation basis (A/C in the queries): the four pods' core fractions sum to
  1.0 and idle is 0, so `sum(pod core-hours) + idle == 1 GPU-hour`. This is the
  milestone-1 identity.
- Measured basis (B/C2): real utilization over the hour. It will not equal the
  reservation; the difference is idle inside reserved slices. Reporting both is
  what turns the proposal from on-paper into measured.
- Cost reconciliation (D): with memory and cores priced separately, per-pod cost
  plus idle cost equals the physical GPU's hourly cost.

## Validation (offline, no cluster needed)

The accounting is encoded as Prometheus recording rules in `rules.yaml` and
unit-tested in `promtool-tests.yaml` against synthetic series that mirror the
workload (four 25% pods on D1, one whole-GPU pod on D2). Run:

```
promtool test rules promtool-tests.yaml
# Unit Testing:  promtool-tests.yaml
#   SUCCESS
```

The test asserts the milestone-1 identity (four quarter pods reserve exactly
1.0 of the card, idle 0), per-pod core and memory fractions of 0.25, per-pod
cost of 0.75 under core_price=$2 / mem_price=$1, and the reconciliation
`sum(pod_cost) + idle_cost == $3/hr` full-card price on both cards. This proves
the PromQL and the math before any GPU time is spent; the cluster run then feeds
real series through the same rules.

## Accounting decisions (carried from the issue discussion)

- Memory and cores are priced separately and summed. There is no single
  universal fraction when the two limits differ.
- `gpucores: 0` or unset means **no limit**, not zero usage. Milestone 1 uses
  explicit nonzero limits only. No-limit and contended usage become later,
  explicitly measured policies, never silently billed as zero.
- Bill per pod from the vGPUmonitor runtime series
  (`hami_container_device_utilization_ratio`, `hami_vgpu_memory_used_bytes`),
  which exist only while a container runs. Keep the scheduler reservation series
  for the shared-card total, idle, and reconciliation only.

## Pod identity and the UID question

The scheduler reservation metrics and the vGPUmonitor runtime metrics both omit
`pod_uid`. Adding a `pod_uid` label was one option, but:

- OpenCost already resolves pod identity and running window through
  kube-state-metrics and cadvisor, joining on `(namespace, pod, container, node)`.
  HAMi already exposes exactly those label keys, so the UID is recovered on the
  OpenCost side without a HAMi label change.
- `pod_uid` is a very high-cardinality label. Adding it runs against the
  in-flight cardinality-reduction work on the `fix-metrics-high-cardinality`
  branch, which is removing high-cardinality labels from these same collectors.

Decision: do the join on `(namespace, pod, container, node)` plus KSM. Keep
`pod_uid` only as a documented fallback if the reproduction shows same-name pod
recreation inside a scrape window actually corrupts attribution.
