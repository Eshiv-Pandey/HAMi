# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""API schema, workload placement, identity, pinning, and fail-closed checks."""

import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from accounting import EvidenceError
import runner
from test_accounting import fixture

UUID = "GPU-00000000-0000-0000-0000-000000000001"
PIN = "pytorch/pytorch@sha256:" + "a" * 64


def native_fixture(data, corrected=False):
    result = {}
    for pod in data["pods"]:
        fraction = pod["core_percent"] / 100 if corrected else 1
        result[pod["pod"]] = {"properties": {"namespace": pod["namespace"], "pod": pod["pod"],
                                "node": pod["node"], "container": "burn"},
            "start": "1970-01-01T00:00:00Z", "end": "1970-01-01T01:00:00Z",
            "gpuHours": fraction, "gpuCost": 3 * fraction, "gpuCostAdjustment": 0,
            "gpuAllocation": {"gpuRequestAverage": fraction}}
    return {"code": 200, "data": [result]}


def pod_fixture():
    return {"metadata": {"namespace": "finops", "name": "quarter", "uid": "uid-1",
                "annotations": {"hami.io/bind-phase": "success", runner.ALLOCATION: f"{UUID},NVIDIA,1024,25:;"}},
            "spec": {"nodeName": "gpu-a", "containers": [{"name": "burn"}]},
            "status": {"phase": "Running", "containerStatuses": [{"name": "burn", "ready": True,
                "restartCount": 0, "state": {"running": {"startedAt": "1970-01-01T00:00:00Z"}}}]}}


def vector(series):
    return {"status": "success", "data": {"resultType": "vector", "result": series}}


class RunnerTests(unittest.TestCase):
    def test_pinned_custom_serializer_uses_lowercase_gpu_allocation(self):
        response = json.loads('''{"code":200,"data":[{"finops/quarter-0/burn/gpu-a":{
          "properties":{"namespace":"finops","pod":"quarter-0","container":"burn","node":"gpu-a"},
          "start":"1970-01-01T00:00:00Z","end":"1970-01-01T01:00:00Z",
          "gpuHours":0.25,"gpuCost":0.75,"gpuCostAdjustment":0,
          "gpuAllocation":{"gpuRequestAverage":0.25,"gpuUsageAverage":0.1,"isGPUShared":true}
        }}]}''')
        rows = runner.native_allocations(response, "finops", {"quarter-0"})
        self.assertEqual(rows[0]["gpu_request_average"], "0.25")

    def test_required_affinity_does_not_set_node_name(self):
        rendered = runner.manifest("finops", "gpu-a", UUID, "shared", PIN)
        deployment = rendered["items"][-1]
        spec = deployment["spec"]["template"]["spec"]
        self.assertNotIn("nodeName", spec)
        self.assertEqual(deployment["spec"]["replicas"], 4)
        field = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchFields"][0]
        self.assertEqual(field, {"key": "metadata.name", "operator": "In", "values": ["gpu-a"]})
        self.assertEqual(deployment["spec"]["template"]["metadata"]["annotations"]["nvidia.com/use-gpuuuid"], UUID)

    def test_control_is_sequentially_usable_on_same_gpu(self):
        deployment = runner.manifest("finops", "gpu-a", UUID, "whole", PIN)["items"][-1]
        self.assertEqual(deployment["spec"]["replicas"], 1)
        limits = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
        self.assertEqual(limits["nvidia.com/gpucores"], "100")
        self.assertEqual(limits["nvidia.com/gpumem-percentage"], "100")

    def test_mutable_workload_image_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "immutable"):
            runner.manifest("finops", "gpu-a", UUID, "shared", "pytorch/pytorch:latest")

    def test_invalid_profile_is_not_silently_whole_gpu(self):
        with self.assertRaises(EvidenceError):
            runner.manifest("finops", "gpu-a", UUID, "invalid", PIN)

    def test_annotation_confirms_actual_uuid_and_mib_units(self):
        normalized = runner.bound_pod(pod_fixture(), "gpu-a", UUID, "shared", 3600)
        self.assertEqual(normalized["uid"], "uid-1")
        self.assertEqual(normalized["memory_bytes"], 1024 * 1048576)
        self.assertEqual(normalized["core_percent"], 25)

    def test_filter_reservation_without_successful_bind_fails(self):
        pod = pod_fixture()
        pod["metadata"]["annotations"]["hami.io/bind-phase"] = "allocating"
        with self.assertRaisesRegex(EvidenceError, "successful"):
            runner.bound_pod(pod, "gpu-a", UUID, "shared", 3600)

    def test_running_pod_on_wrong_gpu_fails(self):
        pod = pod_fixture()
        pod["metadata"]["annotations"][runner.ALLOCATION] = "GPU-other,NVIDIA,1024,25:;"
        with self.assertRaisesRegex(EvidenceError, "physical GPU"):
            runner.bound_pod(pod, "gpu-a", UUID, "shared", 3600)

    def test_restarted_container_does_not_count_as_continuous_run(self):
        pod = pod_fixture()
        pod["status"]["containerStatuses"][0]["restartCount"] = 1
        with self.assertRaisesRegex(EvidenceError, "restarted"):
            runner.bound_pod(pod, "gpu-a", UUID, "shared", 3600)

    def test_init_or_injected_container_cannot_hide_aggregation(self):
        pod = pod_fixture()
        pod["spec"]["containers"].append({"name": "sidecar"})
        with self.assertRaisesRegex(EvidenceError, "one container"):
            runner.bound_pod(pod, "gpu-a", UUID, "shared", 3600)

    def test_replica_ratio_uses_capacity_not_label_value_twice(self):
        node = {"metadata": {"labels": {"nvidia.com/gpu.replicas": "4", "nvidia.com/gpu.count": "1"}},
                "status": {"capacity": {"nvidia.com/gpu": "10"}}}
        self.assertEqual(runner.replica_basis(node, 1), ("replica", 10))

    def test_no_replica_label_retains_physical_quote(self):
        node = {"metadata": {"labels": {}}, "status": {"capacity": {"nvidia.com/gpu": "10"}}}
        self.assertEqual(runner.replica_basis(node, 1), ("physical", 1))

    def test_real_versioned_opencost_gpu_allocation_shape(self):
        data = fixture()
        rows = runner.native_allocations(native_fixture(data), "finops", {p["pod"] for p in data["pods"]})
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["gpu_request_average"] == "1" for row in rows))

    def test_missing_native_gpu_request_is_not_zero(self):
        data = fixture(1)
        response = native_fixture(data)
        del response["data"][0]["quarter-0"]["gpuAllocation"]
        with self.assertRaisesRegex(EvidenceError, "omitted GPU"):
            runner.native_allocations(response, "finops", {"quarter-0"})

    def test_missing_native_pod_is_not_a_successful_report(self):
        data = fixture()
        response = native_fixture(data)
        del response["data"][0]["quarter-0"]
        with self.assertRaisesRegex(EvidenceError, "missing one"):
            runner.native_allocations(response, "finops", {p["pod"] for p in data["pods"]})

    def test_duplicate_api_windows_do_not_double_charge(self):
        data = fixture(1)
        response = native_fixture(data)
        response["data"].append(copy.deepcopy(response["data"][0]))
        with self.assertRaisesRegex(EvidenceError, "overlap"):
            runner.native_allocations(response, "finops", {"quarter-0"})

    def test_native_comparison_reports_raw_and_adjusted_cost(self):
        data = fixture()
        data.update(evidence_source="synthetic", currency="USD", opencost_allocation=native_fixture(data))
        for row in data["opencost_allocation"]["data"][0].values():
            row["gpuCostAdjustment"] = -2.25
        result = runner.report(data)
        self.assertEqual(result["evidence_source"], "synthetic")
        self.assertEqual(result["comparison"]["native_gpu_hours"], "4")
        self.assertEqual(result["comparison"]["reference_gpu_hours"], "1.00")
        self.assertEqual(result["comparison"]["native_gpu_cost"], "12")
        self.assertEqual(result["comparison"]["native_gpu_cost_with_adjustment"], "3.00")

    def test_opencost_node_identity_must_match_capture(self):
        data = fixture(1)
        response = native_fixture(data)
        response["data"][0]["quarter-0"]["properties"]["node"] = "other-node"
        data.update(evidence_source="synthetic", currency="USD", opencost_allocation=response)
        with self.assertRaisesRegex(EvidenceError, "node identity"):
            runner.report(data)

    def test_old_style_no_data_query_returns_nonzero(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"error","error":"bad query"}')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(EvidenceError, "query failed"):
                runner.query(f"http://127.0.0.1:{server.server_port}", "query", 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_ha_runtime_observation_uses_latest_source_time(self):
        values = vector([{"metric": {"pod": "a", "job": "ha-a"}, "value": [15, "5"]},
                         {"metric": {"pod": "a", "job": "ha-b"}, "value": [15, "10"]}])
        timestamps = vector([{"metric": {"pod": "a", "job": "ha-a"}, "value": [15, "10"]},
                             {"metric": {"pod": "a", "job": "ha-b"}, "value": [15, "12"]}])
        self.assertEqual(runner.latest_sample(values, timestamps, pod="a"), (12, 10))

    def test_conflicting_same_time_runtime_observations_fail(self):
        values = vector([{"metric": {"pod": "a", "job": "ha-a"}, "value": [15, "5"]},
                         {"metric": {"pod": "a", "job": "ha-b"}, "value": [15, "10"]}])
        timestamps = vector([{"metric": {"pod": "a", "job": job}, "value": [15, "12"]} for job in ("ha-a", "ha-b")])
        with self.assertRaisesRegex(EvidenceError, "conflicting duplicate"):
            runner.latest_sample(values, timestamps, pod="a")

    def test_report_cli_fails_on_incomplete_evidence(self):
        data = fixture(1)
        data["samples"] = []
        data.update(evidence_source="synthetic", currency="USD", opencost_allocation=native_fixture(data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            runner.save(path, data)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["report", "--dataset", str(path)]), 1)

    def test_corrected_native_results_are_explicitly_checkable(self):
        data = fixture()
        data.update(evidence_source="synthetic", currency="USD", opencost_allocation=native_fixture(data, corrected=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            runner.save(path, data)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["report", "--dataset", str(path), "--require-corrected"]), 0)

    def test_cost_adjustment_cannot_hide_corrected_cost_failure(self):
        data = fixture()
        data.update(evidence_source="synthetic", currency="USD", opencost_allocation=native_fixture(data, corrected=True))
        data["opencost_allocation"]["data"][0]["quarter-0"]["gpuCostAdjustment"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            runner.save(path, data)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["report", "--dataset", str(path), "--require-corrected"]), 1)

    def test_service_urls_cannot_contain_credentials(self):
        with self.assertRaisesRegex(EvidenceError, "plain HTTP"):
            runner.http_json("http://user:secret@localhost:9090", "/api/v1/query")

    def test_all_component_images_have_verified_manifests_and_configurations(self):
        lock = runner.load(runner.ROOT / "versions.json")
        for component in lock["components"].values():
            self.assertRegex(component["image"], r"@sha256:[a-f0-9]{64}$")
            self.assertRegex(component["config_digest"], r"^sha256:[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
