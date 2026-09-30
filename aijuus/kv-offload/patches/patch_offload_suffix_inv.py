#!/usr/bin/env python3
"""RADIANCE offload suffix invalidation (E1, Stage 1) — machinery only, gated/inert.

Background (WORKLOG cont.42): draft rejection currently invalidates nothing (counter rollback
only); the volatile MTP/draft trailing chunk is handled by *withholding* it from store/load, which
caps every group's external hit by one chunk per turn. E1 introduces a rejection hook so the
connector can invalidate only the stale suffix, enabling MTP groups to be kept (Stage 2, separate
gate), without ever touching the prefix.

Stage 1 (this patch, gate RADIANCE_OFFLOAD_SUFFIX_INV, default off) adds the plumbing end to end:
  core scheduler rejection -> connector.on_draft_rejected -> connector scheduler
  -> manager.invalidate (CPU primary conservative removal + tiering/fs cascade).
It is INERT while the existing store-drop is in place (the volatile tail is not stored, so there is
nothing to invalidate). Stage 2 (remove the drop) is a separate gate to be added after Stage 1 is
validated as a no-op.

Conservative by design: only ready, unreferenced CPU-primary blocks are removed; in-flight/in-use
entries are left to normal eviction (no tombstones needed at this stage).

Idempotent (marker patch_offload_suffix_inv); every edited file must ast.parse.
"""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_offload_suffix_inv"
SP = Path(sysconfig.get_paths()["purelib"])
VLLM = SP / "vllm"
CORE_SCHED = VLLM / "v1/core/sched/scheduler.py"
CONN = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading_connector.py"
OSCHED = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
BASE = VLLM / "v1/kv_offload/base.py"
CPU = VLLM / "v1/kv_offload/cpu/manager.py"
TIER = VLLM / "v1/kv_offload/tiering/manager.py"
FS = VLLM / "v1/kv_offload/tiering/fs/manager.py"

# --- 1. core scheduler: notify the connector on rejection -------------------------------------
CORE_OLD = (
    "                if not output_is_stale:\n"
    "                    if request.num_computed_tokens > 0:\n"
    "                        request.num_computed_tokens -= num_rejected\n"
    "                    if request.num_output_placeholders > 0:\n"
    "                        request.num_output_placeholders -= num_rejected\n"
)
CORE_NEW = (
    "                if not output_is_stale:\n"
    "                    if request.num_computed_tokens > 0:\n"
    "                        request.num_computed_tokens -= num_rejected\n"
    "                    if request.num_output_placeholders > 0:\n"
    "                        request.num_output_placeholders -= num_rejected\n"
    "                    # patch_offload_suffix_inv: let the offload connector drop only the stale\n"
    "                    # suffix of the volatile MTP/draft group (boundary = accepted tokens).\n"
    "                    if _RAD_SUFFIX_INV and num_rejected > 0:\n"
    "                        _c = getattr(self, \"connector\", None)\n"
    "                        _hook = getattr(_c, \"on_draft_rejected\", None)\n"
    "                        if _hook is not None:\n"
    "                            try:\n"
    "                                _hook(request, request.num_computed_tokens)\n"
    "                            except Exception:\n"
    "                                logger.exception(\"[radiance] suffix-inv hook failed\")\n"
)

# --- 2. connector delegate ---------------------------------------------------------------------
CONN_OLD = (
    "    def update_state_after_alloc(\n"
    "        self, request: \"Request\", blocks: \"KVCacheBlocks\", num_external_tokens: int\n"
    "    ):\n"
    "        assert self.connector_scheduler is not None\n"
    "        return self.connector_scheduler.update_state_after_alloc(\n"
    "            request, blocks, num_external_tokens\n"
    "        )\n"
)
CONN_NEW = CONN_OLD + (
    "\n"
    "    def on_draft_rejected(self, request, accepted_boundary: int) -> None:\n"
    "        # patch_offload_suffix_inv: forward the rejected-suffix boundary to the connector\n"
    "        # scheduler (no-op unless RADIANCE_OFFLOAD_SUFFIX_INV).\n"
    "        if self.connector_scheduler is not None:\n"
    "            self.connector_scheduler.on_draft_rejected(request, accepted_boundary)\n"
)

# --- 3. connector scheduler: suffix-only invalidation ------------------------------------------
OSCHED_OLD = (
    "    def request_finished(\n"
    "        self,\n"
    "        request: Request,\n"
    "    ) -> tuple[bool, dict[str, Any] | None]:\n"
)
OSCHED_NEW = (
    "    def on_draft_rejected(self, request, accepted_boundary: int) -> None:\n"
    "        \"\"\"patch_offload_suffix_inv: invalidate only the volatile suffix of eagle/MTP groups.\"\"\"\n"
    "        if not _RAD_SUFFIX_INV:\n"
    "            return\n"
    "        rs = self._req_status.get(request.request_id)\n"
    "        if rs is None:\n"
    "            return\n"
    "        for gcfg, gst in zip(self.config.kv_group_configs, rs.group_states):\n"
    "            if not gcfg.is_eagle_group:\n"
    "                continue\n"
    "            c = max(1, int(gcfg.tokens_per_chunk))\n"
    "            b = min(int(accepted_boundary), int(request.num_tokens))\n"
    "            first_stale = max(0, b // c - 1)\n"
    "            stale = list(gst.offload_keys[first_stale:])\n"
    "            if stale:\n"
    "                try:\n"
    "                    self.manager.invalidate(stale, rs.req_context)\n"
    "                except Exception:\n"
    "                    logger.exception(\"[radiance] suffix-inv invalidate failed\")\n"
    "                del gst.offload_keys[first_stale:]\n"
    "            if gst.next_stored_chunk_idx > first_stale:\n"
    "                gst.next_stored_chunk_idx = first_stale\n"
    "\n"
    + OSCHED_OLD
)

# --- 4. base OffloadingManager: default no-op --------------------------------------------------
BASE_OLD = (
    "    def prepare_store(\n"
)
BASE_NEW = (
    "    def invalidate(\n"
    "        self, keys: Collection[OffloadKey], req_context: ReqContext\n"
    "    ) -> None:\n"
    "        \"\"\"patch_offload_suffix_inv: drop stored entries for `keys` (default no-op).\"\"\"\n"
    "        return\n"
    "\n"
    + BASE_OLD
)

# --- 5. CPU primary tier: conservative removal -------------------------------------------------
CPU_OLD = (
    "    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:\n"
)
CPU_NEW = (
    "    def invalidate(\n"
    "        self, keys: Collection[OffloadKey], req_context: ReqContext\n"
    "    ) -> None:\n"
    "        # patch_offload_suffix_inv: remove ready, unreferenced entries; leave in-flight/in-use\n"
    "        # entries to normal eviction (conservative, no tombstone needed at this stage).\n"
    "        for key in keys:\n"
    "            block = self._policy.get(key)\n"
    "            if block is None:\n"
    "                continue\n"
    "            if not getattr(block, \"is_ready\", True):\n"
    "                continue\n"
    "            if getattr(block, \"ref_cnt\", 0) != 0:\n"
    "                continue\n"
    "            self._policy.remove(key)\n"
    "            self._free_block(block)\n"
    "            if self._num_evictable_cache_blocks > 0:\n"
    "                self._num_evictable_cache_blocks -= 1\n"
    "\n"
    + CPU_OLD
)

# --- 6. tiering manager: cascade --------------------------------------------------------------
TIER_OLD = (
    "    @override\n"
    "    def complete_store(\n"
)
TIER_NEW = (
    "    @override\n"
    "    def invalidate(\n"
    "        self, keys: Collection[OffloadKey], req_context: ReqContext\n"
    "    ) -> None:\n"
    "        # patch_offload_suffix_inv: primary first, then cascade to every secondary tier.\n"
    "        self.primary_tier.invalidate(keys, req_context)\n"
    "        for tier in self.secondary_tiers:\n"
    "            fn = getattr(tier, \"invalidate\", None)\n"
    "            if fn is not None:\n"
    "                try:\n"
    "                    fn(keys, req_context)\n"
    "                except Exception:\n"
    "                    logger.exception(\"[radiance] secondary tier invalidate failed\")\n"
    "\n"
    + TIER_OLD
)

# --- 7. fs tier: remove file + forget lookup ---------------------------------------------------
FS_OLD = (
    "    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:\n"
)
FS_NEW = (
    "    def invalidate(self, keys, req_context) -> None:\n"
    "        # patch_offload_suffix_inv: drop the on-disk blocks and the async-lookup cache so a\n"
    "        # cached `present` verdict cannot resurrect a stale entry.\n"
    "        for key in keys:\n"
    "            try:\n"
    "                p = self.file_mapper.get_file_name(key)\n"
    "                if os.path.exists(p):\n"
    "                    os.remove(p)\n"
    "            except OSError:\n"
    "                pass\n"
    "        try:\n"
    "            self._lookup_manager.invalidate(keys)\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    + FS_OLD
)

CORE_TAIL = '''

# ---- RADIANCE offload suffix invalidation (patch_offload_suffix_inv) -------------------------
import os as _rsi_os

_RAD_SUFFIX_INV = _rsi_os.environ.get("RADIANCE_OFFLOAD_SUFFIX_INV", "0") == "1"
'''
OSCHED_TAIL = '''

# ---- RADIANCE offload suffix invalidation (patch_offload_suffix_inv) -------------------------
import os as _rsi2_os

_RAD_SUFFIX_INV = _rsi2_os.environ.get("RADIANCE_OFFLOAD_SUFFIX_INV", "0") == "1"
'''


def edit(path, old, new, tail=None):
    src = path.read_text()
    if MARK in src:
        print(f"[suffix-inv] {path.name} already applied")
        return
    n = src.count(old)
    if n != 1:
        print(f"[suffix-inv] {path.name}: anchor matched {n}x, NOT applied", file=sys.stderr)
        raise SystemExit(1)
    src = src.replace(old, new, 1)
    if tail:
        src += tail
    ast.parse(src)
    path.write_text(src)
    print(f"[suffix-inv] applied: {path.name}")


edit(CORE_SCHED, CORE_OLD, CORE_NEW, tail=CORE_TAIL)
edit(CONN, CONN_OLD, CONN_NEW)
edit(OSCHED, OSCHED_OLD, OSCHED_NEW, tail=OSCHED_TAIL)
edit(BASE, BASE_OLD, BASE_NEW)
edit(CPU, CPU_OLD, CPU_NEW)
edit(TIER, TIER_OLD, TIER_NEW)
edit(FS, FS_OLD, FS_NEW)
