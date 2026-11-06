# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare, capture, and report a pinned native-versus-reference experiment."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

from accounting import EvidenceError, analyze, number

ROOT = Path(__file__).resolve().parent
ALLOCATION = "hami.io/vgpu-devices-allocated"
PART_OF = "app.kubernetes.io/part-of"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def harness_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("accounting.py", "runner.py", "workload.py")}


def kubectl(*args):
    result = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise EvidenceError(f"kubectl failed: {result.stderr.strip()}")
    return result.stdout


def kube_json(*args):
    return json.loads(kubectl(*args, "-o", "json"))


def http_json(base, path, params=None):
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise EvidenceError("expected a plain HTTP(S) service URL")
    url = base.rstrip("/") + path
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def query(prom_url, expression, at):
    response = http_json(prom_url, "/api/v1/query", {"query": expression, "time": at})
    if response.get("status") != "success" or response.get("data", {}).get("resultType") != "vector":
        raise EvidenceError("Prometheus query failed or returned an unexpected result type")
    if response.get("warnings"):
        raise EvidenceError("Prometheus query returned warnings; evidence may be partial")
    return response


def vectors(response):
    return response["data"]["result"]


def selector(name, **labels):
    return name + "{" + ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items()) + "}"


def unique_value(response, **labels):
    matches = [series for series in vectors(response)
               if all(series["metric"].get(key) == value for key, value in labels.items())]
    if not matches:
        raise EvidenceError(f"missing metric evidence for {labels}")
    values = {number(series["value"][1]) for series in matches}
    if len(values) != 1:
        raise EvidenceError("conflicting duplicate metric values")
    return values.pop()


def latest_sample(response, timestamps, **labels):
    """Deduplicate HA observations by source time, without summing GPU shares."""
    def identity(metric):
        return tuple(sorted((key, value) for key, value in metric.items() if key != "__name__"))
    sources = {identity(series["metric"]): number(series["value"][1]) for series in vectors(timestamps)}
    samples = []
    for series in vectors(response):
        if all(series["metric"].get(key) == value for key, value in labels.items()):
            key = identity(series["metric"])
            if key not in sources:
                raise EvidenceError("metric observation has no source timestamp")
            samples.append((sources[key], number(series["value"][1])))
    if not samples:
        raise EvidenceError(f"missing timestamped metric evidence for {labels}")
    latest = max(source for source, _ in samples)
    values = {value for source, value in samples if source == latest}
    if len(values) != 1:
        raise EvidenceError("conflicting duplicate observations at the same source time")
    return latest, values.pop()


def check_targets(prom_url, responses, step):
    """Verify the scrape interval and health for the targets actually queried."""
    target_data = http_json(prom_url, "/api/v1/targets", {"state": "active"})
    if target_data.get("status") != "success":
        raise EvidenceError("Prometheus target inspection failed")
    targets = {(target["labels"].get("job"), target["labels"].get("instance")): target
               for target in target_data["data"]["activeTargets"]}
    selected = {}
    for response in responses:
        for series in vectors(response):
            key = (series["metric"].get("job"), series["metric"].get("instance"))
            target = targets.get(key)
            if not all(key) or not target or target.get("health") != "up":
                raise EvidenceError("a queried metric lacks a healthy scrape target")
            if target.get("scrapeInterval") != f"{step}s" or target.get("lastError"):
                raise EvidenceError("actual scrape settings differ from the pinned interval")
            selected["/".join(key)] = {"job": key[0], "instance": key[1], "scrape_interval": target["scrapeInterval"]}
    return selected


def epoch(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise EvidenceError("running window timestamp lacks a timezone")
    return value.timestamp()


def manifest(namespace, node, uuid, profile, image):
    """Target a physical UUID while leaving placement to the scheduler."""
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace) or len(namespace) > 63:
        raise EvidenceError("invalid experiment namespace")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]|[a-z0-9]", node) or not re.fullmatch(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", uuid):
        raise EvidenceError("a node and a physical NVIDIA GPU UUID are required")
    if profile not in ("shared", "whole") or not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image):
        raise EvidenceError("a supported profile and immutable workload image are required")
    count, percent = (4, 25) if profile == "shared" else (1, 100)
    labels = {PART_OF: "hami-finops", "finops.hami.io/profile": profile}
    name = "hami-finops-" + profile
    resources = {"nvidia.com/gpu": "1", "nvidia.com/gpucores": str(percent),
                 "nvidia.com/gpumem-percentage": str(percent)}
    deployment = {"apiVersion": "apps/v1", "kind": "Deployment",
                  "metadata": {"namespace": namespace, "name": name, "labels": labels},
                  "spec": {"replicas": count, "selector": {"matchLabels": labels},
                  "template": {"metadata": {"labels": labels, "annotations": {"nvidia.com/use-gpuuuid": uuid}},
                  "spec": {"affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution":
                      {"nodeSelectorTerms": [{"matchFields": [{"key": "metadata.name", "operator": "In", "values": [node]}]}]}}},
                  "terminationGracePeriodSeconds": 10,
                  "containers": [{"name": "burn", "image": image,
                      "command": ["python", "-u", "/hami-finops/workload.py"],
                      "env": [{"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"}],
                      "resources": {"requests": {"cpu": "100m", "memory": "256Mi", **resources},
                                    "limits": {"memory": "768Mi", **resources}},
                      "readinessProbe": {"exec": {"command": ["python", "-c", "from pathlib import Path; raise SystemExit(not Path('/tmp/gpu-ready').exists())"]},
                                         "periodSeconds": 5, "timeoutSeconds": 3},
                      "volumeMounts": [{"name": "workload", "mountPath": "/hami-finops", "readOnly": True}]}],
                  "volumes": [{"name": "workload", "configMap": {"name": "hami-finops-workload"}}]}}}}
    return {"apiVersion": "v1", "kind": "List", "items": [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace, "labels": {PART_OF: "hami-finops"}}},
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "hami-finops-workload", "namespace": namespace},
         "data": {"workload.py": (ROOT / "workload.py").read_text(encoding="utf-8")}}, deployment]}


def matches_declared(actual, desired):
    """Compare declared fields while accepting Kubernetes-defaulted fields."""
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(key in actual and matches_declared(actual[key], value)
                                              for key, value in desired.items())
    if isinstance(desired, list):
        return isinstance(actual, list) and len(actual) == len(desired) and all(
            matches_declared(a, b) for a, b in zip(actual, desired))
    return actual == desired


def bound_pod(pod, node, uuid, profile, end):
    """Validate the actual allocation and running container, not Filter intent."""
    spec, status, metadata = pod["spec"], pod["status"], pod["metadata"]
    if spec.get("nodeName") != node or status.get("phase") != "Running":
        raise EvidenceError("workload is not Running on the requested node")
    if spec.get("initContainers") or len(spec["containers"]) != 1:
        raise EvidenceError("first milestone requires one container and no init containers")
    if metadata.get("deletionTimestamp") or metadata.get("annotations", {}).get("hami.io/bind-phase") != "success":
        raise EvidenceError("workload has no successful current HAMi bind")
    entries = [entry for entry in metadata["annotations"].get(ALLOCATION, "").split(";") if entry]
    if len(entries) != 1:
        raise EvidenceError("missing or ambiguous HAMi allocation annotation")
    devices = [part for part in entries[0].split(":") if part]
    fields = devices[0].split(",") if len(devices) == 1 else []
    if len(fields) != 4 or fields[0] != uuid or fields[1] != "NVIDIA":
        raise EvidenceError("HAMi did not allocate the requested physical GPU")
    percent = 25 if profile == "shared" else 100
    if number(fields[3]) != percent:
        raise EvidenceError("actual HAMi core allocation differs from the experiment")
    memory_mib = number(fields[2])
    if memory_mib <= 0 or memory_mib != memory_mib.to_integral_value():
        raise EvidenceError("allocation annotation must contain positive integer MiB")
    states = status.get("containerStatuses", [])
    if len(states) != 1 or not states[0].get("ready") or states[0].get("restartCount") != 0:
        raise EvidenceError("GPU workload is unready or restarted")
    running = states[0].get("state", {}).get("running")
    if not running:
        raise EvidenceError("no independently confirmed container running window")
    return {"namespace": metadata["namespace"], "pod": metadata["name"], "uid": metadata["uid"],
            "node": node, "container": states[0]["name"], "device_uuid": uuid,
            "core_percent": percent, "memory_bytes": int(memory_mib * 1048576),
            "bound": True, "running": [[epoch(running["startedAt"]), end]]}


def component_pins(pods, lock):
    """Capture only component identities; do not archive other Pods' secrets."""
    found = {name: [] for name in lock["components"] if name != "workload"}
    for pod in pods["items"]:
        statuses = {state["name"]: state for state in pod.get("status", {}).get("containerStatuses", [])}
        for container in pod["spec"]["containers"]:
            image = container["image"].removeprefix("docker.io/")
            for name in found:
                pin = lock["components"][name]
                if image.split("@")[0].split(":")[0] != pin["repository"]:
                    continue
                state = statuses.get(container["name"], {})
                digest = pin["image"].split("@", 1)[1]
                allowed = (digest, pin["config_digest"])
                if not state.get("imageID", "").endswith(allowed) or "running" not in state.get("state", {}):
                    raise EvidenceError(f"{name} does not match the immutable version pin")
                identity = {"namespace": pod["metadata"]["namespace"], "pod": pod["metadata"]["name"],
                            "image": container["image"], "image_id": state["imageID"]}
                if name == "opencost":
                    environment = {entry["name"]: entry.get("value") for entry in container.get("env", [])}
                    expected = {"PROMETHEUS_QUERY_RESOLUTION_SECONDS": str(lock["opencost_query_resolution_seconds"]),
                                "KUBECOST_SCRAPE_INTERVAL": f'{lock["scrape_interval_seconds"]}s'}
                    if any(environment.get(key) != value for key, value in expected.items()):
                        raise EvidenceError("OpenCost process resolution or scrape configuration differs from the lock")
                    identity["query_configuration"] = expected
                found[name].append(identity)
    if not all(found.values()):
        raise EvidenceError("not all pinned HAMi/OpenCost/Prometheus/KSM components were found")
    return found


def replica_basis(node, physical_count):
    """Mirror the count inputs to OpenCost's existing replica-price adjustment."""
    labels, capacity = node["metadata"].get("labels", {}), node["status"]["capacity"]
    if "nvidia.com/gpu.replicas" not in labels:
        return "physical", 1
    if number(labels.get("nvidia.com/gpu.count")) != physical_count:
        raise EvidenceError("GFD physical count disagrees with HAMi physical capacity")
    virtual = number(capacity.get("nvidia.com/gpu.shared", capacity.get("nvidia.com/gpu")))
    ratio = virtual / physical_count
    if ratio < 1 or ratio != ratio.to_integral_value():
        raise EvidenceError("invalid OpenCost physical-to-virtual price ratio")
    return "replica", int(ratio)


def hardware_inventory(output, uuid):
    """Retain driver/device evidence from the actual GPU workload environment."""
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    matches = [row for row in rows if row and row[0] == uuid]
    if len(matches) != 1 or len(matches[0]) != 4:
        raise EvidenceError("GPU hardware inventory does not identify the allocated physical UUID")
    identifier, model, memory, driver = matches[0]
    if number(memory) <= 0 or not re.fullmatch(r"\d+(?:\.\d+)+", driver):
        raise EvidenceError("invalid GPU memory or driver inventory")
    return {"device_uuid": identifier, "model": model, "reported_memory_mib": memory,
            "driver_version": driver, "source": "nvidia-smi in the allocated workload environment"}


def native_allocations(response, namespace, pod_names):
    """Read actual OpenCost allocations, including the version-pinned GPU shape."""
    if response.get("code") != 200 or not isinstance(response.get("data"), list):
        raise EvidenceError("OpenCost allocation API returned an error or unsupported schema")
    allocations, intervals = [], {}
    for allocation_set in response["data"]:
        if not isinstance(allocation_set, dict):
            raise EvidenceError("invalid OpenCost allocation set")
        for allocation in allocation_set.values():
            properties = allocation.get("properties") or {}
            if properties.get("namespace") != namespace or properties.get("pod") not in pod_names:
                continue
            if properties.get("container") != "burn":
                raise EvidenceError("unexpected container aggregation in OpenCost results")
            gpu = allocation.get("gpuAllocation") or {}
            requested = gpu.get("gpuRequestAverage", allocation.get("gpuRequestAverage"))
            if requested is None:
                raise EvidenceError("OpenCost omitted GPU request accounting")
            hours, cost, request = map(number, (allocation["gpuHours"], allocation["gpuCost"], requested))
            adjustment = number(allocation.get("gpuCostAdjustment", 0))
            if hours <= 0 or request <= 0 or cost < 0:
                raise EvidenceError("OpenCost returned missing or invalid GPU allocation data")
            start, end = epoch(allocation["start"]), epoch(allocation["end"])
            key = (properties["pod"], properties.get("node"))
            if start >= end or any(start < previous_end and previous_start < end for previous_start, previous_end in intervals.get(key, [])):
                raise EvidenceError("OpenCost allocation windows overlap or are invalid")
            intervals.setdefault(key, []).append((start, end))
            allocations.append({"pod": properties["pod"], "node": properties.get("node"),
                                "start": start, "end": end, "gpu_hours": str(hours), "gpu_cost": str(cost),
                                "gpu_cost_adjustment": str(adjustment), "gpu_cost_adjusted": str(cost + adjustment),
                                "gpu_request_average": str(request)})
    if {allocation["pod"] for allocation in allocations} != set(pod_names):
        raise EvidenceError("OpenCost is missing one or more experiment Pods")
    return allocations


def report(data):
    if data.get("evidence_source") not in ("synthetic", "live-cluster") or not data["pods"]:
        raise EvidenceError("native comparison requires declared evidence provenance and experiment Pods")
    if data["evidence_source"] == "live-cluster" and data.get("capture_status") != "complete":
        raise EvidenceError("partial or failed capture is not complete billing evidence")
    reference = analyze(data)
    native = native_allocations(data["opencost_allocation"], data["pods"][0]["namespace"],
                                {pod["pod"] for pod in data["pods"]})
    native_hours = sum((number(row["gpu_hours"]) for row in native), Decimal(0))
    native_cost = sum((number(row["gpu_cost"]) for row in native), Decimal(0))
    adjusted_cost = sum((number(row["gpu_cost_adjusted"]) for row in native), Decimal(0))
    reference_hours = sum((number(card["reserved_gpu_hours"]) for card in reference["cards"]), Decimal(0))
    reference_cost = sum((number(card["reserved_cost"]) for card in reference["cards"]), Decimal(0))
    nodes = {pod["pod"]: pod["node"] for pod in data["pods"]}
    if any(row["node"] != nodes[row["pod"]] for row in native):
        raise EvidenceError("OpenCost node identity differs from the captured placement")
    if any(number(row["start"]) < number(data["window"]["start"]) or number(row["end"]) > number(data["window"]["end"]) for row in native):
        raise EvidenceError("OpenCost allocated time outside the measured window")
    for pod in data["pods"]:
        expected = [(max(a, data["window"]["start"]), min(b, data["window"]["end"]))
                    for a, b in pod["running"] if a < data["window"]["end"] and b > data["window"]["start"]]
        actual = sorted((row["start"], row["end"]) for row in native if row["pod"] == pod["pod"])
        merged = []
        for a, b in actual:
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        if merged != expected:
            raise EvidenceError("OpenCost allocation does not cover the confirmed running window")
    return {"evidence_source": data["evidence_source"], "currency": data["currency"],
            "reference": reference, "native_opencost": native,
            "comparison": {"native_gpu_hours": str(native_hours), "reference_gpu_hours": str(reference_hours),
                           "native_gpu_cost": str(native_cost), "reference_gpu_cost": str(reference_cost),
                           "native_gpu_cost_with_adjustment": str(adjusted_cost),
                           "gpu_hour_difference": str(native_hours - reference_hours),
                           "gpu_cost_difference": str(native_cost - reference_cost),
                           "gpu_cost_with_adjustment_difference": str(adjusted_cost - reference_cost)},
            "validation": "native API evidence compared with a reference policy; no integration implementation is implied"}


def prepare(args):
    lock = load(ROOT / "versions.json")
    out = Path(args.out)
    if (out / "request.json").exists():
        raise EvidenceError("experiment directory already contains a request; preserve prior evidence")
    deployment = manifest(args.namespace, args.node, args.uuid, args.profile, lock["components"]["workload"]["image"])
    save(out / "workload.json", deployment)
    request = {"namespace": args.namespace, "node": args.node, "uuid": args.uuid, "profile": args.profile,
               "lock_sha256": hashlib.sha256((ROOT / "versions.json").read_bytes()).hexdigest(),
               "harness_sha256": harness_hashes()}
    if args.apply:
        request["context"] = kubectl("config", "current-context").strip()
        namespaces = kube_json("get", "namespaces")
        if any(ns["metadata"]["name"] == args.namespace and ns["metadata"].get("labels", {}).get(PART_OF) != "hami-finops" for ns in namespaces["items"]):
            raise EvidenceError("existing namespace is not owned by the experiment")
        existing = kube_json("get", "pods", "-A")
        if any(p["metadata"].get("namespace") == args.namespace for p in existing["items"]):
            raise EvidenceError("experiment namespace already contains Pods; use a fresh namespace")
        kubectl("apply", "-f", str(out / "workload.json"))
    save(out / "request.json", request)
    print(json.dumps({"manifest": str(out / "workload.json"), "request": request}, indent=2))


def capture(args):
    """Capture a stable window; preserve partial evidence when any check fails."""
    out = Path(args.out)
    request, lock = load(out / "request.json"), load(ROOT / "versions.json")
    if (out / "dataset.json").exists() or (out / "failure.json").exists():
        raise EvidenceError("preserve the existing experiment; use a new output directory")
    digest = hashlib.sha256((ROOT / "versions.json").read_bytes()).hexdigest()
    if digest != request["lock_sha256"]:
        raise EvidenceError("version lock changed after workload preparation")
    if request.get("harness_sha256") != harness_hashes():
        raise EvidenceError("harness changed after workload preparation; render a new experiment")
    context = kubectl("config", "current-context").strip()
    if request.get("context") and context != request["context"]:
        raise EvidenceError("Kubernetes context changed after workload preparation")
    step = lock["scrape_interval_seconds"]
    resolution = lock["opencost_query_resolution_seconds"]
    if args.duration < resolution or args.duration % resolution or args.duration % step:
        raise EvidenceError("duration must be a positive multiple of the OpenCost resolution and scrape interval")
    if not re.fullmatch(r"[A-Z]{3}", args.currency):
        raise EvidenceError("specify the actual pricing currency as a three-letter code")
    namespace, node, uuid, profile = (request[key] for key in ("namespace", "node", "uuid", "profile"))
    pod_selector = f"{PART_OF}=hami-finops,finops.hami.io/profile={profile}"
    data, cleanup_uid, capture_failed = None, None, False
    try:
        deployment = kube_json("get", "deployment", "hami-finops-" + profile, "-n", namespace)
        template = deployment["spec"]["template"]
        wanted = manifest(namespace, node, uuid, profile, lock["components"]["workload"]["image"])["items"][-1]
        labels = deployment["metadata"].get("labels", {})
        if any(labels.get(key) != value for key, value in wanted["metadata"]["labels"].items()) or template["metadata"].get("annotations", {}).get("nvidia.com/use-gpuuuid") != uuid:
            raise EvidenceError("deployment is not the prepared experiment")
        if not matches_declared(deployment["spec"], wanted["spec"]):
            raise EvidenceError("deployment spec differs from the prepared experiment")
        workload_config = kube_json("get", "configmap", "hami-finops-workload", "-n", namespace)
        if workload_config.get("data", {}).get("workload.py") != (ROOT / "workload.py").read_text(encoding="utf-8"):
            raise EvidenceError("cluster workload code differs from the prepared experiment")
        cleanup_uid = deployment["metadata"]["uid"]
        pins = component_pins(kube_json("get", "pods", "-A"), lock)
        node_data = kube_json("get", "node", node)
        if node_data["metadata"].get("labels", {}).get("kubernetes.io/arch") != "amd64":
            raise EvidenceError("runner architecture differs from the image lock")
        server = kube_json("version")
        build = http_json(args.prom_url, "/api/v1/status/buildinfo")
        if build.get("data", {}).get("version") != lock["components"]["prometheus"]["version"]:
            raise EvidenceError("Prometheus API version differs from the image lock")
        kubectl("wait", "--for=condition=Ready", "pods", "-n", namespace, "-l", pod_selector, "--timeout=120s")
        # Fresh observations must follow readiness, rather than a stale lookback
        # from an earlier Pod or the opposite experiment profile.
        time.sleep(step * 2)
        start = int(time.time())
        end = start + args.duration
        first = kube_json("get", "pods", "-n", namespace, "-l", pod_selector)
        expected_count = 4 if profile == "shared" else 1
        if len(first["items"]) != expected_count:
            raise EvidenceError("unexpected experiment Pod count")
        pods = [bound_pod(pod, node, uuid, profile, end) for pod in first["items"]]
        if len({pod["uid"] for pod in pods}) != expected_count:
            raise EvidenceError("duplicate Pod identities")
        capacity_response = query(args.prom_url, selector("hami_gpu_memory_limit_bytes", node=node), start)
        physical_uuids = {series["metric"].get("device_uuid") for series in vectors(capacity_response)}
        if physical_uuids != {uuid}:
            raise EvidenceError("first milestone needs one dedicated physical GPU on the selected node")
        capacity = unique_value(capacity_response, node=node, device_uuid=uuid)
        price_response = query(args.prom_url, selector("node_gpu_hourly_cost", node=node), start)
        price = unique_value(price_response, node=node)
        basis, replicas = replica_basis(node_data, 1)
        data = {"schema_version": 1, "policy": "core-reservation", "evidence_source": "live-cluster", "capture_status": "collecting",
                "currency": args.currency, "window": {"start": start, "end": end, "step_seconds": step},
                "devices": [{"node": node, "device_uuid": uuid, "memory_limit_bytes": str(capacity),
                             "gpu_hourly_cost": str(price), "price_basis": basis, "replicas": replicas}],
                "pods": pods, "samples": [], "versions": {"images": pins, "kubernetes": server,
                    "prometheus": build, "lock": lock, "context": context, "lock_sha256": digest,
                    "harness_sha256": request["harness_sha256"], "python": sys.version}}
        smi = kubectl("exec", pods[0]["pod"], "-n", namespace, "-c", "burn", "--", "nvidia-smi",
                      "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits")
        data["hardware"] = hardware_inventory(smi, uuid)
        save(out / "node.json", {"metadata": {"name": node, "labels": node_data["metadata"].get("labels", {})},
                                 "status": {"capacity": node_data["status"]["capacity"]}})
        metric_names = ("hami_vgpu_core_allocated_ratio", "hami_vgpu_memory_allocated_bytes",
                        "hami_container_device_utilization_ratio", "hami_vgpu_memory_used_bytes", "kube_pod_info")
        expected = {pod["uid"]: pod for pod in pods}
        for at in range(start, end + 1, step):
            time.sleep(max(0, at - time.time()))
            snapshot = kube_json("get", "pods", "-n", namespace, "-l", pod_selector)
            current = [bound_pod(pod, node, uuid, profile, end) for pod in snapshot["items"]]
            if {pod["uid"] for pod in current} != set(expected):
                raise EvidenceError("Pod identity changed inside the measured window")
            if any(pod != expected[pod["uid"]] for pod in current):
                raise EvidenceError("binding or running episode changed inside the window")
            save(out / f"pods-{at-start:06d}.json", snapshot)
            responses, timestamps = {}, {}
            for name in metric_names:
                expression = selector(name, namespace=namespace)
                responses[name] = query(args.prom_url, expression, at)
                timestamps[name] = query(args.prom_url, f"timestamp({expression})", at)
            global_core = query(args.prom_url, selector("hami_vgpu_core_allocated_ratio", device_uuid=uuid), at)
            if any(series["metric"].get("namespace") != namespace and number(series["value"][1]) > 0
                   for series in vectors(global_core)):
                raise EvidenceError("another namespace holds a reservation on the experiment GPU")
            observed_names = {series["metric"].get("pod") for series in vectors(global_core)}
            if observed_names != {pod["pod"] for pod in pods}:
                raise EvidenceError("another profile or Pod occupies the experiment GPU")
            fresh_capacity = query(args.prom_url, selector("hami_gpu_memory_limit_bytes", node=node), at)
            fresh_price = query(args.prom_url, selector("node_gpu_hourly_cost", node=node), at)
            capacity_times = query(args.prom_url, f'timestamp({selector("hami_gpu_memory_limit_bytes", node=node)})', at)
            price_times = query(args.prom_url, f'timestamp({selector("node_gpu_hourly_cost", node=node)})', at)
            capacity_time, _ = latest_sample(fresh_capacity, capacity_times, device_uuid=uuid, node=node)
            price_time, _ = latest_sample(fresh_price, price_times, node=node)
            if unique_value(fresh_capacity, device_uuid=uuid, node=node) != capacity or unique_value(fresh_price, node=node) != price:
                raise EvidenceError("capacity or GPU pricing changed inside the measured window")
            data["scrape_targets"] = check_targets(args.prom_url, [*responses.values(), fresh_capacity, fresh_price], step)
            save(out / f"metrics-{at-start:06d}.json", {"values": responses, "timestamps": timestamps,
                                                      "global_core": global_core, "capacity": fresh_capacity, "price": fresh_price,
                                                      "capacity_timestamps": capacity_times, "price_timestamps": price_times})
            for pod in pods:
                labels = {"namespace": namespace, "pod": pod["pod"], "device_uuid": uuid}
                core = unique_value(responses[metric_names[0]], node=node, **labels)
                memory = unique_value(responses[metric_names[1]], node=node, **labels)
                runtime = {**labels, "container": "burn"}
                util_time, utilization = latest_sample(responses[metric_names[2]], timestamps[metric_names[2]], **runtime)
                used_time, used = latest_sample(responses[metric_names[3]], timestamps[metric_names[3]], **runtime)
                core_time, _ = latest_sample(responses[metric_names[0]], timestamps[metric_names[0]], node=node, **labels)
                memory_time, _ = latest_sample(responses[metric_names[1]], timestamps[metric_names[1]], node=node, **labels)
                ksm_labels = {"namespace": namespace, "pod": pod["pod"], "uid": pod["uid"], "node": node}
                ksm_time, _ = latest_sample(responses[metric_names[4]], timestamps[metric_names[4]], **ksm_labels)
                source_times = (util_time, used_time, core_time, memory_time, ksm_time, capacity_time, price_time)
                if any(source > at for source in source_times):
                    raise EvidenceError("metric source timestamp is later than the observation")
                if core != pod["core_percent"] or memory != pod["memory_bytes"] or used < 0:
                    raise EvidenceError("metrics disagree with the confirmed allocation")
                workload_digest = lock["components"]["workload"]["image"].split("@", 1)[1]
                for raw_pod in snapshot["items"]:
                    if raw_pod["metadata"]["uid"] == pod["uid"]:
                        image_id = raw_pod["status"]["containerStatuses"][0]["imageID"]
                        if not image_id.endswith((workload_digest, lock["components"]["workload"]["config_digest"])):
                            raise EvidenceError("GPU workload image does not match its immutable pin")
                data["samples"].append({**{key: pod[key] for key in
                    ("namespace", "pod", "uid", "node", "container", "device_uuid", "core_percent", "memory_bytes")},
                    "time": at, "source_timestamp": str(min(source_times)),
                    "utilization_percent": str(utilization), "used_memory_bytes": str(used), "runtime": True,
                    "uid_source": "live-kubernetes-snapshot-and-kube-pod-info"})
            save(out / "partial-dataset.json", data)
        if analyze(data)["status"] != "complete":
            raise EvidenceError("missing or stale samples leave an unattributed billing window")
        time.sleep(step * 2)
        window = ",".join(datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z") for value in (start, end))
        allocation = http_json(args.opencost_url, "/allocation/compute", {"window": window,
            "step": f"{args.duration}s", "aggregate": "namespace,pod,container,node",
            "shareIdle": "false", "includeIdle": "false"})
        data["opencost_allocation"] = allocation
        save(out / "opencost-allocation.json", allocation)
        for pod in pods:
            log = kubectl("logs", pod["pod"], "-n", namespace, "-c", "burn")
            if '"event": "gpu-ready"' not in log or '"event": "gpu-alive"' not in log:
                raise EvidenceError("workload logs do not corroborate a sustained CUDA run")
            (out / f"{pod['uid']}-workload.log").write_text(log, encoding="utf-8")
        data["versions"]["images_at_end"] = component_pins(kube_json("get", "pods", "-A"), lock)
        final_node = kube_json("get", "node", node)
        if replica_basis(final_node, 1) != (basis, replicas):
            raise EvidenceError("replica pricing metadata changed inside the window")
        data["capture_status"] = "complete"
        result = report(data)
        save(out / "dataset.json", data)
        save(out / "report.json", result)
        print(json.dumps(result["comparison"], indent=2))
    except Exception as exc:
        capture_failed = True
        if data is not None:
            data["capture_status"] = "failed"
            save(out / "partial-dataset.json", data)
        save(out / "failure.json", {"message": str(exc), "evidence_complete": False})
        raise
    finally:
        if args.cleanup and cleanup_uid:
            try:
                current_deployment = kube_json("get", "deployment", "hami-finops-" + profile, "-n", namespace)
                if current_deployment["metadata"]["uid"] != cleanup_uid:
                    raise EvidenceError("deployment identity changed; cleanup requires manual inspection")
                # Enforce identity at the API server, including a replacement
                # racing the read above. Raw DELETE accepts DeleteOptions.
                with tempfile.TemporaryDirectory(prefix="hami-finops-cleanup-") as directory:
                    options = Path(directory) / "delete-options.json"
                    save(options, {"apiVersion": "v1", "kind": "DeleteOptions", "propagationPolicy": "Foreground",
                                   "preconditions": {"uid": cleanup_uid}})
                    path = f"/apis/apps/v1/namespaces/{namespace}/deployments/hami-finops-{profile}"
                    kubectl("delete", "--raw", path, "-f", str(options))
                kubectl("wait", "--for=delete", "pods", "-n", namespace, "-l", pod_selector, "--timeout=120s")
            except Exception as exc:
                save(out / "cleanup-failure.json", {"message": str(exc), "deployment_uid": cleanup_uid})
                if not capture_failed:
                    raise
                print(f"cleanup also failed: {exc}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="render a pinned workload; --apply targets the configured cluster")
    for field in ("namespace", "node", "uuid", "out"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--profile", choices=("shared", "whole"), required=True)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(function=prepare)
    p = sub.add_parser("capture", help="capture a steady running window and actual OpenCost allocations")
    p.add_argument("--out", required=True)
    p.add_argument("--prom-url", required=True)
    p.add_argument("--opencost-url", required=True)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--currency", required=True, help="actual currency used by the OpenCost GPU price")
    p.add_argument("--cleanup", action="store_true")
    p.set_defaults(function=capture)
    p = sub.add_parser("report", help="validate and compare an existing captured dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--require-corrected", action="store_true")
    def print_report(args):
        result = report(load(args.dataset))
        print(json.dumps(result, indent=2))
        if result["reference"]["status"] != "complete":
            raise EvidenceError("incomplete evidence cannot support a billing conclusion")
        if args.require_corrected and any(abs(number(result["comparison"][field])) > Decimal("0.000001")
                for field in ("gpu_hour_difference", "gpu_cost_difference", "gpu_cost_with_adjustment_difference")):
            raise EvidenceError("native OpenCost does not match the reference policy")
    p.set_defaults(function=print_report)
    args = parser.parse_args(argv)
    try:
        args.function(args)
        return 0
    except (EvidenceError, KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
