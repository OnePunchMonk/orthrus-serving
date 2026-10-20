# SPDX-License-Identifier: Apache-2.0
"""Ensure the plugin's registrations are active for tests run standalone
(outside a real vLLM engine, which would otherwise call this via its
plugin-loading mechanism). Skipped when vLLM isn't installed, so the
SGLang tests can run on their own."""

import importlib.util

if importlib.util.find_spec("vllm") is not None:
    import orthrus_vllm

    orthrus_vllm.register()
