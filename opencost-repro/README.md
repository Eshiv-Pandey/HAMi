# HAMi/OpenCost experimental reproduction

Harness for [HAMi #3073](https://github.com/Project-HAMi/HAMi/issues/3073) and
[OpenCost #3828](https://github.com/opencost/opencost/issues/3828). It compares
actual native OpenCost allocations with an explicitly core-based reservation
reference. The reference is not an OpenCost integration implementation.

The two profiles use one physical NVIDIA GPU sequentially. Workloads require
explicit positive limits and a stable single-container running window. Missing
or ambiguous evidence fails the experiment. Image manifests are pinned in
`versions.json`; currency must be supplied from the runner's pricing setup.
Use Python 3.11 or newer. The lock distinguishes 15-second Prometheus scrapes
from OpenCost's 60-second process query resolution. The capture checks both
settings; the allocation API does not accept a resolution override in this
pinned release.

CPU validation and command reference:

```sh
python3 -m unittest discover -s opencost-repro -p 'test_*.py' -v
python3 opencost-repro/runner.py --help
```

GPU Kubernetes and live OpenCost validation remain pending. Operational and
design documentation will be submitted separately to the HAMi website, as
requested in the proposal discussion.

The implementation, regression tests, and this artifact were prepared primarily
with Codex assistance. CPU fixtures do not establish working GPU integration.
