# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accounting invariants, including incomplete and ambiguous evidence."""

import copy
from decimal import Decimal
import unittest

from accounting import EvidenceError, analyze


def fixture(count=4, percent=25, step=900):
    data = {"schema_version": 1, "policy": "core-reservation",
            "window": {"start": 0, "end": 3600, "step_seconds": step},
            "devices": [{"node": "gpu-a", "device_uuid": "GPU-physical",
                         "memory_limit_bytes": 4096, "gpu_hourly_cost": "3.00",
                         "price_basis": "physical", "replicas": 1}],
            "pods": [], "samples": []}
    for index in range(count):
        pod = {"namespace": "finops", "pod": f"quarter-{index}", "uid": f"uid-{index}",
               "node": "gpu-a", "container": "burn", "device_uuid": "GPU-physical",
               "bound": True, "core_percent": percent,
               "memory_bytes": 4096 * percent // 100, "running": [[0, 3600]]}
        data["pods"].append(pod)
        for at in range(0, 3600, step):
            data["samples"].append({**{k: pod[k] for k in
                ("namespace", "pod", "uid", "node", "container", "device_uuid", "core_percent", "memory_bytes")},
                "time": at, "source_timestamp": at, "runtime": True, "utilization_percent": 10})
    return data


class AccountingTests(unittest.TestCase):
    def test_four_quarters_equal_one_reserved_gpu_hour(self):
        result = analyze(fixture())
        card = result["cards"][0]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(Decimal(card["reserved_gpu_hours"]), 1)
        self.assertEqual(Decimal(card["reserved_cost"]), 3)
        self.assertEqual(Decimal(card["idle_gpu_hours"]), 0)
        self.assertTrue(all(Decimal(p["reserved_gpu_hours"]) == Decimal("0.25") for p in result["pods"]))

    def test_whole_gpu_control(self):
        result = analyze(fixture(1, 100))
        self.assertEqual(Decimal(result["cards"][0]["reserved_gpu_hours"]), 1)
        self.assertEqual(Decimal(result["cards"][0]["reserved_cost"]), 3)

    def test_fifteen_second_sampling_does_not_accumulate_rounding_error(self):
        result = analyze(fixture(step=15))
        self.assertEqual(Decimal(result["cards"][0]["reserved_gpu_hours"]), 1)

    def test_late_start_and_early_exit(self):
        data = fixture(1, 25)
        data["pods"][0]["running"] = [[900, 2700]]
        result = analyze(data)["cards"][0]
        self.assertEqual(Decimal(result["reserved_gpu_hours"]), Decimal("0.125"))
        self.assertEqual(Decimal(result["idle_gpu_hours"]), Decimal("0.875"))

    def test_failed_bind_does_not_charge_filter_reservation(self):
        data = fixture(1)
        data["pods"][0].update(bound=False, running=[])
        card = analyze(data)["cards"][0]
        self.assertEqual(Decimal(card["reserved_gpu_hours"]), 0)
        self.assertEqual(Decimal(card["idle_gpu_hours"]), 1)

    def test_idle_gpu_uses_direct_capacity(self):
        data = fixture(0)
        card = analyze(data)["cards"][0]
        self.assertEqual(Decimal(card["idle_gpu_hours"]), 1)
        self.assertEqual(Decimal(card["idle_cost"]), 3)

    def test_missing_sample_is_unattributed_not_idle(self):
        data = fixture()
        data["samples"] = [s for s in data["samples"] if not (s["uid"] == "uid-0" and s["time"] == 900)]
        result = analyze(data)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(Decimal(result["cards"][0]["unattributed_gpu_hours"]), Decimal("0.0625"))
        self.assertEqual(Decimal(result["cards"][0]["idle_gpu_hours"]), 0)

    def test_stale_sample_is_not_a_running_window(self):
        data = fixture(1)
        data["samples"][0]["source_timestamp"] = -1801
        self.assertEqual(analyze(data)["status"], "incomplete")

    def test_sample_before_new_running_episode_does_not_prove_it(self):
        data = fixture(1)
        data["pods"][0]["running"] = [[900, 3600]]
        data["samples"][1]["source_timestamp"] = 890
        result = analyze(data)
        self.assertEqual(result["status"], "incomplete")
        self.assertGreater(Decimal(result["cards"][0]["unattributed_gpu_hours"]), 0)

    def test_duplicate_ha_scrapes_do_not_double_charge(self):
        data = fixture()
        data["samples"] += copy.deepcopy(data["samples"])
        self.assertEqual(Decimal(analyze(data)["cards"][0]["reserved_cost"]), 3)

    def test_conflicting_duplicates_fail(self):
        data = fixture()
        conflicting = copy.deepcopy(data["samples"][0])
        conflicting["utilization_percent"] = 90
        data["samples"].append(conflicting)
        with self.assertRaisesRegex(EvidenceError, "conflicting duplicate"):
            analyze(data)

    def test_pod_name_reuse_does_not_alias_uids(self):
        data = fixture(1)
        old = data["pods"][0]
        old["running"] = [[0, 1800]]
        new = {**old, "uid": "uid-new", "running": [[1800, 3600]]}
        data["pods"].append(new)
        with self.assertRaisesRegex(EvidenceError, "Pod-name reuse"):
            analyze(data)

    def test_zero_utilization_still_has_a_reserved_cost(self):
        data = fixture()
        for sample in data["samples"]:
            sample["utilization_percent"] = 0
        self.assertEqual(Decimal(analyze(data)["cards"][0]["reserved_cost"]), 3)

    def test_replica_quote_is_normalized_once(self):
        data = fixture()
        data["devices"][0].update(gpu_hourly_cost="0.75", price_basis="replica", replicas=4)
        card = analyze(data)["cards"][0]
        self.assertEqual(Decimal(card["physical_gpu_hourly_cost"]), 3)
        self.assertEqual(Decimal(card["reserved_cost"]), 3)

    def test_physical_quote_is_not_discounted_by_replica_labels(self):
        data = fixture()
        data["devices"][0]["replicas"] = 4
        self.assertEqual(Decimal(analyze(data)["cards"][0]["reserved_cost"]), 3)

    def test_zero_core_limit_is_unsupported_not_free(self):
        data = fixture(1)
        data["pods"][0]["core_percent"] = 0
        with self.assertRaisesRegex(EvidenceError, "explicit positive"):
            analyze(data)

    def test_overcommit_fails_instead_of_clamping_idle(self):
        with self.assertRaisesRegex(EvidenceError, "exceeds capacity"):
            analyze(fixture(5))

    def test_non_finite_price_is_rejected(self):
        data = fixture()
        data["devices"][0]["gpu_hourly_cost"] = "NaN"
        with self.assertRaisesRegex(EvidenceError, "non-finite"):
            analyze(data)

    def test_unknown_price_basis_is_rejected(self):
        data = fixture()
        data["devices"][0]["price_basis"] = "guess"
        with self.assertRaisesRegex(EvidenceError, "price basis"):
            analyze(data)

    def test_zero_capacity_is_not_reconstructed_from_allocations(self):
        data = fixture(0)
        data["devices"][0]["memory_limit_bytes"] = 0
        with self.assertRaisesRegex(EvidenceError, "capacity"):
            analyze(data)

    def test_missing_metric_uid_fails_identity_check(self):
        data = fixture(1)
        del data["samples"][0]["uid"]
        with self.assertRaisesRegex(EvidenceError, "identity"):
            analyze(data)

    def test_duplicate_uuid_on_different_nodes_is_not_two_gpus(self):
        data = fixture(0)
        data["devices"].append({**data["devices"][0], "node": "gpu-b"})
        with self.assertRaisesRegex(EvidenceError, "device identity"):
            analyze(data)

    def test_overlapping_running_episodes_are_rejected(self):
        data = fixture(1)
        data["pods"][0]["running"] = [[0, 1800], [900, 3600]]
        with self.assertRaisesRegex(EvidenceError, "overlapping"):
            analyze(data)


if __name__ == "__main__":
    unittest.main()
