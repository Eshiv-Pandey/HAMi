# Copyright 2026 The HAMi Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep a CUDA context alive; readiness follows a successful GPU operation."""

import json
import os
from pathlib import Path
import signal
import time

import torch

running = True


def stop(_signal, _frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; this is not a GPU reproduction")
a = torch.ones((256, 256), device="cuda")
b = a.clone()
result = a @ b
torch.cuda.synchronize()
print(json.dumps({"event": "gpu-ready", "torch": torch.__version__,
                  "cuda": torch.version.cuda,
                  "device": torch.cuda.get_device_name(0),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
Path("/tmp/gpu-ready").write_text("ready\n")
last_heartbeat = time.monotonic()
while running:
    result = a @ b
    torch.cuda.synchronize()
    if time.monotonic() - last_heartbeat >= 15:
        print(json.dumps({"event": "gpu-alive", "at": time.time()}), flush=True)
        last_heartbeat = time.monotonic()
    time.sleep(0.05)
