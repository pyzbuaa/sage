# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
SERVER_ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SERVER_ROOT_DIR, "results")

ROBOMIMIC_ROOT_DIR = os.path.join(SERVER_ROOT_DIR, "../robomimic")
M2T2_ROOT_DIR = os.path.join(SERVER_ROOT_DIR, "../M2T2")
MATFUSE_ROOT_DIR = os.path.join(SERVER_ROOT_DIR, "../matfuse-sd/src")

# Room floor/wall textures: "flux" (remote HTTP) or "matfuse" (local GPU).
MATERIAL_BACKEND = os.environ.get("MATERIAL_BACKEND", "flux").lower()
if MATERIAL_BACKEND not in ("flux", "matfuse"):
    raise ValueError(f"Invalid MATERIAL_BACKEND={MATERIAL_BACKEND!r}; use 'flux' or 'matfuse'.")

PHYSICS_CRITIC_ENABLED = os.environ.get("PHYSICS_CRITIC_ENABLED", "true").lower() == "true"
SEMANTIC_CRITIC_ENABLED = os.environ.get("SEMANTIC_CRITIC_ENABLED", "true").lower() == "true"

# Concurrent TRELLIS 3D generation requests from place_objects_in_room.
# Should match the number of TRELLIS worker GPUs (one in-flight job per GPU).
TRELLIS_GENERATION_MAX_WORKERS = int(os.environ.get("TRELLIS_GENERATION_MAX_WORKERS", "4"))

print(f"MATERIAL_BACKEND: {MATERIAL_BACKEND}", file=sys.stderr)
print(f"PHYSICS_CRITIC_ENABLED: {PHYSICS_CRITIC_ENABLED}", file=sys.stderr)
print(f"SEMANTIC_CRITIC_ENABLED: {SEMANTIC_CRITIC_ENABLED}", file=sys.stderr)
print(f"TRELLIS_GENERATION_MAX_WORKERS: {TRELLIS_GENERATION_MAX_WORKERS}", file=sys.stderr)