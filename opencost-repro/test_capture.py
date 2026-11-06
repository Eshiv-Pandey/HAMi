# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise the entire capture without a cluster, GPU, or wall-clock waits."""

from argparse import Namespace
import contextlib
import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from accounting import EvidenceError
import runner
from test_runner import UUID, pod_fixture, vector


def rfc3339(at):
    return datetime.fromtimestamp(at, timezone.utc).isoformat().replace("+00:00", "Z")


class Clock:
    def __init__(self):
        self.now = 970

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.lock = runner.load(runner.ROOT / "versions.json")
        self.args = Namespace(out=str(self.out), prom_url="http://prometheus:9090",
                              opencost_url="http://opencost:9003", duration=60, currency="USD", cleanup=False)
        runner.save(self.out / "request.json", {"namespace": "finops", "node": "gpu-a", "uuid": UUID,
                    "profile": "shared", "context": "fixture", "lock_sha256":
                    hashlib.sha256((runner.ROOT / "versions.json").read_bytes()).hexdigest(),
                    "harness_sha256": runner.harness_hashes()})
        self.deployment = runner.manifest("finops", "gpu-a", UUID, "shared",
                              self.lock["components"]["workload"]["image"])["items"][-1]
        self.deployment["metadata"]["uid"] = "deployment-uid"
        self.deployment["spec"]["template"]["spec"]["volumes"][0]["configMap"]["defaultMode"] = 420
        self.deployment["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
        self.workload_code = (runner.ROOT / "workload.py").read_text(encoding="utf-8")
        self.pods = []
        for index in range(4):
            pod = pod_fixture()
            pod["metadata"].update(name=f"quarter-{index}", uid=f"uid-{index}")
            pod["status"]["containerStatuses"][0]["imageID"] = self.lock["components"]["workload"]["image"]
            self.pods.append(pod)
        self.components = []
        for name, pin in self.lock["components"].items():
            if name == "workload":
                continue
            self.components.append({"metadata": {"namespace": "monitoring", "name": name},
                "spec": {"containers": [{"name": name, "image": pin["image"], "env": [
                    {"name": "PROMETHEUS_QUERY_RESOLUTION_SECONDS", "value": "60"},
                    {"name": "KUBECOST_SCRAPE_INTERVAL", "value": "15s"}]}]},
                "status": {"containerStatuses": [{"name": name, "imageID": pin["image"],
                                                     "state": {"running": {}}}]}})
        self.clock = Clock()
        self.commands, self.http_calls, self.delete_options = [], [], []
        self.uid_changes = False
        self.missing_runtime = False
        self.stale_price = False
        self.future_runtime = False
        self.short_native = False
        self.unhealthy = False
        self.wrong_interval = False
        self.replaced = False
        self.delete_fails = False
        for target, replacement in (("kubectl", self.kubectl), ("kube_json", self.kube_json),
                                    ("http_json", self.http_json)):
            self.enterContext(patch.object(runner, target, side_effect=replacement))
        self.enterContext(patch.object(runner.time, "time", side_effect=self.clock.time))
        self.enterContext(patch.object(runner.time, "sleep", side_effect=self.clock.sleep))

    def kubectl(self, *args):
        self.commands.append(args)
        if args[:2] == ("config", "current-context"):
            return "fixture\n"
        if args[0] == "exec":
            return f"{UUID}, NVIDIA RTX 3050, 4096, 555.99\n"
        if args[0] == "logs":
            return '{"event": "gpu-ready"}\n{"event": "gpu-alive"}\n'
        if args[0] == "delete":
            self.delete_options.append(runner.load(args[args.index("-f") + 1]))
            if self.delete_fails:
                raise EvidenceError("API precondition conflict")
        return ""

    def kube_json(self, *args):
        if args[:2] == ("get", "configmap"):
            return {"data": {"workload.py": self.workload_code}}
        if args[:2] == ("get", "deployment"):
            result = copy.deepcopy(self.deployment)
            if self.replaced and self.clock.now >= 1015:
                result["metadata"]["uid"] = "replacement-uid"
            return result
        if args[:2] == ("get", "node"):
            return {"metadata": {"labels": {"kubernetes.io/arch": "amd64"}},
                    "status": {"capacity": {"nvidia.com/gpu": "10"}}}
        if args[:2] == ("get", "pods"):
            result = copy.deepcopy(self.components if "-A" in args else self.pods)
            if self.uid_changes and "-A" not in args and self.clock.now >= 1015:
                result[0]["metadata"]["uid"] = "recreated-uid"
            return {"items": result}
        if args == ("version",):
            return {"serverVersion": {"gitVersion": "v1.32.0"}}
        raise AssertionError(f"unexpected Kubernetes request: {args}")

    def http_json(self, base, path, params=None):
        self.http_calls.append((base, path, copy.deepcopy(params)))
        if path == "/api/v1/status/buildinfo":
            return {"status": "success", "data": {"version": "3.2.1"}}
        if path == "/api/v1/targets":
            return {"status": "success", "data": {"activeTargets": [
                {"labels": {"job": "fixture", "instance": "fixture:9090"},
                 "health": "down" if self.unhealthy else "up", "lastError": "",
                 "scrapeInterval": "30s" if self.wrong_interval else "15s"}]}}
        if path == "/allocation/compute":
            start, end = params["window"].split(",")
            hours = Decimal(60) / 3600
            return {"code": 200, "data": [{pod["metadata"]["name"]: {
                "properties": {"namespace": "finops", "pod": pod["metadata"]["name"],
                               "container": "burn", "node": "gpu-a"},
                "start": rfc3339(1015) if self.short_native else start, "end": end,
                "gpuHours": str(hours), "gpuCost": str(hours * 3), "gpuCostAdjustment": 0,
                "gpuAllocation": {"gpuRequestAverage": 1}} for pod in self.pods}]}
        if path != "/api/v1/query":
            raise AssertionError(f"unexpected HTTP request: {path}")
        expression, at = params["query"], params["time"]
        is_timestamp = expression.startswith("timestamp(")
        name = re.search(r'(?:timestamp\()?([a-z_]+)\{', expression)[1]
        metric = {"job": "fixture", "instance": "fixture:9090"}
        values = {"hami_vgpu_core_allocated_ratio": 25, "hami_vgpu_memory_allocated_bytes": 1073741824,
                  "hami_container_device_utilization_ratio": 10, "hami_vgpu_memory_used_bytes": 1048576,
                  "kube_pod_info": 1, "hami_gpu_memory_limit_bytes": 4294967296, "node_gpu_hourly_cost": 3}
        if name in ("hami_gpu_memory_limit_bytes", "node_gpu_hourly_cost"):
            labels = {**metric, "node": "gpu-a"}
            if name == "hami_gpu_memory_limit_bytes":
                labels["device_uuid"] = UUID
            source = at - 100 if self.stale_price and name == "node_gpu_hourly_cost" else at
            return vector([{"metric": labels, "value": [at, str(source if is_timestamp else values[name])]}])
        if self.missing_runtime and name == "hami_vgpu_memory_used_bytes":
            return vector([])
        series = []
        for pod in self.pods:
            labels = {**metric, "namespace": "finops", "pod": pod["metadata"]["name"]}
            if name.startswith("hami_"):
                labels["device_uuid"] = UUID
            if "allocated" in name or name == "kube_pod_info":
                labels["node"] = "gpu-a"
            else:
                labels["container"] = "burn"
            if name == "kube_pod_info":
                labels["uid"] = pod["metadata"]["uid"]
            source = at + 1 if self.future_runtime and name == "hami_vgpu_memory_used_bytes" else at
            series.append({"metric": labels, "value": [at, str(source if is_timestamp else values[name])]})
        return vector(series)

    def capture(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runner.capture(self.args)

    def test_successful_capture_queries_actual_allocation_api(self):
        self.capture()
        data = runner.load(self.out / "dataset.json")
        self.assertEqual(data["capture_status"], "complete")
        self.assertEqual(data["hardware"]["driver_version"], "555.99")
        self.assertEqual(len(data["samples"]), 20)
        calls = [params for _, path, params in self.http_calls if path == "/allocation/compute"]
        self.assertEqual(calls, [{"window": f"{rfc3339(1000)},{rfc3339(1060)}",
                                 "step": "60s", "aggregate": "namespace,pod,container,node",
                                 "shareIdle": "false", "includeIdle": "false"}])
        self.assertEqual(runner.load(self.out / "report.json")["reference"]["status"], "complete")
        self.assertTrue((self.out / "opencost-allocation.json").exists())
        self.assertFalse(any(command[0] == "delete" for command in self.commands))

    def test_uid_change_preserves_partial_and_cannot_report_success(self):
        self.uid_changes = True
        with self.assertRaisesRegex(EvidenceError, "Pod identity changed"):
            self.capture()
        self.assertFalse((self.out / "dataset.json").exists())
        partial = runner.load(self.out / "partial-dataset.json")
        self.assertEqual(partial["capture_status"], "failed")
        with self.assertRaisesRegex(EvidenceError, "partial or failed"):
            runner.report(partial)
        self.assertFalse(runner.load(self.out / "failure.json")["evidence_complete"])

    def test_missing_runtime_is_not_billable_evidence(self):
        self.missing_runtime = True
        with self.assertRaisesRegex(EvidenceError, "missing timestamped"):
            self.capture()
        self.assertFalse((self.out / "dataset.json").exists())

    def test_stale_price_leaves_unattributed_window(self):
        self.stale_price = True
        with self.assertRaisesRegex(EvidenceError, "unattributed"):
            self.capture()

    def test_future_timestamp_cannot_be_hidden_by_minimum(self):
        self.future_runtime = True
        with self.assertRaisesRegex(EvidenceError, "later than"):
            self.capture()

    def test_partial_native_window_is_rejected_and_preserved(self):
        self.short_native = True
        with self.assertRaisesRegex(EvidenceError, "does not cover"):
            self.capture()
        self.assertTrue((self.out / "opencost-allocation.json").exists())
        self.assertFalse((self.out / "dataset.json").exists())

    def test_installed_component_pin_mismatch(self):
        self.components[0]["status"]["containerStatuses"][0]["imageID"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(EvidenceError, "immutable version pin"):
            self.capture()

    def test_opencost_process_resolution_is_checked(self):
        container = next(pod for pod in self.components if pod["metadata"]["name"] == "opencost")["spec"]["containers"][0]
        container["env"][0]["value"] = "300"
        with self.assertRaisesRegex(EvidenceError, "process resolution"):
            self.capture()

    def test_different_workload_command_is_rejected(self):
        self.deployment["spec"]["template"]["spec"]["containers"][0]["command"] = ["sleep", "infinity"]
        with self.assertRaisesRegex(EvidenceError, "deployment spec differs"):
            self.capture()

    def test_cluster_workload_code_must_match_prepared_source(self):
        self.workload_code = "print('cpu-only')"
        with self.assertRaisesRegex(EvidenceError, "cluster workload code differs"):
            self.capture()

    def test_opencost_process_scrape_configuration_is_checked(self):
        container = next(pod for pod in self.components if pod["metadata"]["name"] == "opencost")["spec"]["containers"][0]
        container["env"][1]["value"] = "30s"
        with self.assertRaisesRegex(EvidenceError, "scrape configuration"):
            self.capture()

    def test_unhealthy_actual_scrape_target(self):
        self.unhealthy = True
        with self.assertRaisesRegex(EvidenceError, "healthy scrape"):
            self.capture()

    def test_wrong_actual_scrape_interval(self):
        self.wrong_interval = True
        with self.assertRaisesRegex(EvidenceError, "pinned interval"):
            self.capture()

    def test_cleanup_requires_owned_deployment(self):
        self.args.cleanup = True
        self.deployment["metadata"]["labels"][runner.PART_OF] = "other-work"
        with self.assertRaisesRegex(EvidenceError, "not the prepared"):
            self.capture()
        self.assertFalse(any(command[0] == "delete" for command in self.commands))

    def test_cleanup_does_not_delete_replacement(self):
        self.args.cleanup = True
        self.replaced = True
        with self.assertRaisesRegex(EvidenceError, "deployment identity changed"):
            self.capture()
        self.assertFalse(any(command[0] == "delete" for command in self.commands))
        self.assertTrue((self.out / "cleanup-failure.json").exists())

    def test_cleanup_uses_atomic_api_uid_precondition(self):
        self.args.cleanup = True
        self.capture()
        self.assertEqual(self.delete_options, [{"apiVersion": "v1", "kind": "DeleteOptions",
                         "propagationPolicy": "Foreground", "preconditions": {"uid": "deployment-uid"}}])
        delete = next(command for command in self.commands if command[0] == "delete")
        self.assertEqual(delete[1:3], ("--raw", "/apis/apps/v1/namespaces/finops/deployments/hami-finops-shared"))

    def test_cleanup_error_does_not_mask_capture_failure(self):
        self.args.cleanup = True
        self.uid_changes = True
        self.delete_fails = True
        with self.assertRaisesRegex(EvidenceError, "Pod identity changed"):
            self.capture()
        self.assertIn("precondition conflict", runner.load(self.out / "cleanup-failure.json")["message"])

    def test_changed_preparation_pin_fails_before_cluster_actions(self):
        request = runner.load(self.out / "request.json")
        request["lock_sha256"] = "changed"
        runner.save(self.out / "request.json", request)
        with self.assertRaisesRegex(EvidenceError, "version lock changed"):
            self.capture()
        self.assertEqual(self.commands, [])

    def test_changed_harness_fails_before_cluster_actions(self):
        request = runner.load(self.out / "request.json")
        request["harness_sha256"]["workload.py"] = "changed"
        runner.save(self.out / "request.json", request)
        with self.assertRaisesRegex(EvidenceError, "harness changed"):
            self.capture()
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
