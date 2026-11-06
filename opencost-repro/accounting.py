# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed reference accounting for the explicitly core-based milestone.

This is an evidence checker, not an OpenCost allocation-source implementation.
Runtime observations corroborate a bound Pod's independently recorded running
window. Missing observations remain unattributed; they do not become idle.
"""

from decimal import Decimal


class EvidenceError(ValueError):
    """The evidence cannot support an unambiguous accounting result."""


def number(value):
    """Reject non-finite values rather than silently converting them to zero."""
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise EvidenceError(f"invalid numeric value: {value!r}") from exc
    if not result.is_finite():
        raise EvidenceError("non-finite accounting value")
    return result


def physical_price(device):
    """Normalize an explicitly identified replica quote exactly once."""
    price = number(device["gpu_hourly_cost"])
    if price < 0:
        raise EvidenceError("negative GPU price")
    basis = device["price_basis"]
    if basis == "physical":
        return price
    if basis != "replica":
        raise EvidenceError("GPU price basis must be physical or replica")
    replicas = number(device["replicas"])
    if replicas < 1 or replicas != replicas.to_integral_value():
        raise EvidenceError("invalid physical-to-virtual replica count")
    return price * replicas


def analyze(data):
    """Integrate only confirmed, sampled running windows and reconcile the card.

    Each observation supports at most one configured sampling interval. A
    missing interval leaves the corresponding running reservation unknown.
    The strict first milestone rejects Pod-name reuse, rather than pretending
    that HAMi's UID-less runtime metrics resolve historical identity.
    """
    if data.get("schema_version") != 1 or data.get("policy") != "core-reservation":
        raise EvidenceError("unsupported evidence schema or accounting policy")
    start, end = map(number, (data["window"]["start"], data["window"]["end"]))
    step = number(data["window"]["step_seconds"])
    if start >= end or step <= 0:
        raise EvidenceError("invalid accounting window")
    devices, uuids = {}, set()
    for device in data["devices"]:
        key = (device["node"], device["device_uuid"])
        if not all(key) or key in devices or device["device_uuid"] in uuids:
            raise EvidenceError("missing or duplicate physical device identity")
        if number(device["memory_limit_bytes"]) <= 0:
            raise EvidenceError("missing physical memory capacity")
        devices[key] = device
        uuids.add(device["device_uuid"])
        physical_price(device)
    if not devices:
        raise EvidenceError("no physical GPU capacity evidence")

    pods, names, intervals = {}, {}, {}
    boundaries = {start, end}
    for pod in data["pods"]:
        uid = pod["uid"]
        name = (pod["namespace"], pod["pod"])
        if not uid or uid in pods:
            raise EvidenceError("missing or duplicate Pod UID")
        if name in names and names[name] != uid:
            raise EvidenceError("Pod-name reuse requires a stronger historical identity source")
        names[name], pods[uid] = uid, pod
        key = (pod["node"], pod["device_uuid"])
        if key not in devices:
            raise EvidenceError("Pod refers to an unknown physical GPU")
        core, memory = number(pod["core_percent"]), number(pod["memory_bytes"])
        if not 0 < core <= 100 or not 0 < memory <= number(devices[key]["memory_limit_bytes"]):
            raise EvidenceError("milestone requires explicit positive, bounded allocations")
        windows = []
        for a, b in pod["running"]:
            a, b = number(a), number(b)
            if a >= b or (windows and a < windows[-1][1]):
                raise EvidenceError("invalid or overlapping container running windows")
            if pod["bound"] is not True:
                raise EvidenceError("running GPU workload lacks a successful HAMi bind")
            windows.append((a, b))
            boundaries.update((max(start, min(end, a)), max(start, min(end, b))))
        intervals[uid] = windows

    observations = {}
    for sample in data["samples"]:
        uid = sample.get("uid")
        if uid not in pods:
            raise EvidenceError("observation has missing or unrecognized Pod identity")
        pod = pods[uid]
        for label in ("namespace", "pod", "node", "container", "device_uuid"):
            if sample.get(label) != pod[label]:
                raise EvidenceError("observation does not match the bound Pod/device identity")
        at, source = number(sample["time"]), number(sample["source_timestamp"])
        if at < start or at > end or source > at:
            raise EvidenceError("observation timestamp is outside its evidence window")
        core, memory = number(sample["core_percent"]), number(sample["memory_bytes"])
        if core != number(pod["core_percent"]) or memory != number(pod["memory_bytes"]):
            raise EvidenceError("observed reservation disagrees with the binding annotation")
        utilization = number(sample["utilization_percent"])
        if not 0 <= utilization <= 100 or not isinstance(sample["runtime"], bool):
            raise EvidenceError("invalid runtime observation")
        identity = (at, uid)
        signature = (source, core, memory, utilization, sample["runtime"])
        if identity in observations and observations[identity] != signature:
            raise EvidenceError("conflicting duplicate scrapes")
        observations[identity] = signature
        boundaries.update((at, min(end, at + step)))

    # Accumulate seconds before division; repeated 15/3600 additions otherwise
    # introduce Decimal rounding drift even for an exactly one-hour fixture.
    totals = {key: {"reserved_gpu_seconds": Decimal(0), "memory_gpu_seconds": Decimal(0),
                    "idle_gpu_seconds": Decimal(0), "unattributed_gpu_seconds": Decimal(0)}
              for key in devices}
    pod_seconds = {uid: Decimal(0) for uid in pods}
    pod_unknown = {uid: Decimal(0) for uid in pods}
    sample_times = {uid: sorted(at for at, identity in observations if identity == uid)
                    for uid in pods}
    points = sorted(boundaries)
    for left, right in zip(points, points[1:]):
        seconds = right - left
        reserved = {key: Decimal(0) for key in devices}
        memory_reserved = {key: Decimal(0) for key in devices}
        for uid, pod in pods.items():
            if not any(a <= left and right <= b for a, b in intervals[uid]):
                continue
            key = (pod["node"], pod["device_uuid"])
            core_fraction = number(pod["core_percent"]) / 100
            memory_fraction = number(pod["memory_bytes"]) / number(devices[key]["memory_limit_bytes"])
            reserved[key] += core_fraction
            memory_reserved[key] += number(pod["memory_bytes"])
            eligible = [at for at in sample_times[uid] if at <= left]
            observed = False
            if eligible:
                at = eligible[-1]
                source, _, _, _, runtime = observations[(at, uid)]
                observed = runtime and right <= at + step and at - source <= step * 2
                # A sample before a new container running episode cannot prove it ran.
                observed = observed and any(a <= source <= at < b and a <= left < b for a, b in intervals[uid])
            if observed:
                totals[key]["reserved_gpu_seconds"] += core_fraction * seconds
                totals[key]["memory_gpu_seconds"] += memory_fraction * seconds
                pod_seconds[uid] += core_fraction * seconds
            else:
                totals[key]["unattributed_gpu_seconds"] += core_fraction * seconds
                pod_unknown[uid] += core_fraction * seconds
        for key in devices:
            if reserved[key] > 1 or memory_reserved[key] > number(devices[key]["memory_limit_bytes"]):
                raise EvidenceError("physical GPU reservation exceeds capacity")
            totals[key]["idle_gpu_seconds"] += (1 - reserved[key]) * seconds

    cards, allocations = [], []
    for key, device in devices.items():
        price, seconds = physical_price(device), totals[key]
        budget = (end - start) / 3600
        if seconds["reserved_gpu_seconds"] + seconds["idle_gpu_seconds"] + seconds["unattributed_gpu_seconds"] != end - start:
            raise EvidenceError("GPU-hour reconciliation failed")
        total = {field.replace("seconds", "hours"): value / 3600 for field, value in seconds.items()}
        cards.append({"node": key[0], "device_uuid": key[1],
                      **{field: str(value) for field, value in total.items()},
                      "physical_gpu_hours": str(budget), "physical_gpu_hourly_cost": str(price),
                      "reserved_cost": str(total["reserved_gpu_hours"] * price),
                      "idle_cost": str(total["idle_gpu_hours"] * price),
                      "unattributed_cost": str(total["unattributed_gpu_hours"] * price)})
    for uid, pod in pods.items():
        price = physical_price(devices[(pod["node"], pod["device_uuid"])])
        allocations.append({"uid": uid, "namespace": pod["namespace"], "pod": pod["pod"],
                            "reserved_gpu_hours": str(pod_seconds[uid] / 3600),
                            "reserved_cost": str(pod_seconds[uid] / 3600 * price),
                            "unattributed_gpu_hours": str(pod_unknown[uid] / 3600)})
    return {"policy": "core-reservation", "implementation": "offline-reference",
            "status": "complete" if all(t["unattributed_gpu_seconds"] == 0 for t in totals.values()) else "incomplete",
            "cards": cards, "pods": allocations}
