# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hot-path profiling markers and (future) gated kernel replacements.

This package is deliberately self-contained: it may only import from the
standard library and torch — never from ``verl`` or ``vllm_omni`` internals —
so that weekly drift of the pinned ``verl`` / ``vllm-omni`` commits cannot
break it. See ``docs/perf/kernel_replacement_rfc.md`` for the plan this
implements.
"""

from .site_markers import site_marker

__all__ = ["site_marker"]
