#!/usr/bin/env python3
"""RADIANCE offload lazy-commit (E2): take the KV-offload store D2H off the step path.

Deep research (WORKLOG cont.41-42) corrected the earlier anchor: the ~18% cold-prefill stall is
NOT `wait_for_save()` (that is a no-op in `offloading_connector.py`) but
`pre_forward -> handle_preemptions -> worker.wait(jobs_to_flush)` (`offloading/worker.py:232`,
`cpu/gpu_worker.py:708`), caused by (a) the self-flush of a finishing request's own just-allocated
blocks (`offloading/scheduler.py:1706-1712`) and (b) one-step store deferral
(`offloading/worker.py:248-260`).

Two gated hunks, both preserving every E2 invariant (the existing `_block_id_to_pending_jobs`
fence, FIFO completion, content-hash keys, primary `ref_cnt`):
  A. submit the store at creation in `post_forward` (submit is non-blocking; the D2H only enqueues
     on a stream ordered after compute), so its completion overlaps the next engine step;
  B. stop adding a finishing request's own just-allocated blocks to `jobs_to_flush` — the existing
     flush-on-reallocation path still fences the job the first time a block is actually reused.

Gate: RADIANCE_OFFLOAD_LAZY_COMMIT=1 (read at vLLM runtime; patch always applied, inert when off).
The fence is re-timed, not removed: rare under-pressure reuse still blocks on the job's event.

Idempotent (marker patch_offload_lazy_commit); every edited file must ast.parse.
"""
import ast
import sys
import sysconfig
from pathlib import Path

MARK = "patch_offload_lazy_commit"
SP = Path(sysconfig.get_paths()["purelib"])
OFF = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading"
WORKER = OFF / "worker.py"
SCHED = OFF / "scheduler.py"

WORKER_OLD = (
    "    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):\n"
    "        for job_id, entry in metadata.store_jobs.items():\n"
    "            if not self._is_store_writer:\n"
    "                # Gate before queueing: no _unsubmitted_store_jobs entry.\n"
    "                self._connector_worker_meta.mark_completed(job_id)\n"
    "                continue\n"
    "            # NOTE(orozery): defer the store to the beginning of the next\n"
    "            # engine step, so that offloading starts AFTER transfers related\n"
    "            # to token sampling, thereby avoiding delays to token generation.\n"
    "            assert isinstance(entry.src_spec, GPULoadStoreSpec)\n"
    "            self._unsubmitted_store_jobs.append(\n"
    "                (job_id, entry.src_spec, entry.dst_spec)\n"
    "            )\n"
)
WORKER_NEW = (
    "    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):\n"
    "        for job_id, entry in metadata.store_jobs.items():\n"
    "            if not self._is_store_writer:\n"
    "                # Gate before queueing: no _unsubmitted_store_jobs entry.\n"
    "                self._connector_worker_meta.mark_completed(job_id)\n"
    "                continue\n"
    "            assert isinstance(entry.src_spec, GPULoadStoreSpec)\n"
    "            if _LAZY_COMMIT:\n"
    "                # patch_offload_lazy_commit: start the D2H now so it overlaps the next engine\n"
    "                # step; submit_store only enqueues (non-blocking), ordered after compute.\n"
    "                success = self.worker.submit_store(job_id, entry.src_spec, entry.dst_spec)\n"
    "                assert success\n"
    "                continue\n"
    "            # NOTE(orozery): defer the store to the beginning of the next\n"
    "            # engine step, so that offloading starts AFTER transfers related\n"
    "            # to token sampling, thereby avoiding delays to token generation.\n"
    "            self._unsubmitted_store_jobs.append(\n"
    "                (job_id, entry.src_spec, entry.dst_spec)\n"
    "            )\n"
)

SCHED_OLD = (
    "            if req.is_finished():\n"
    "                # Register non-sliding-window blocks for flush detection.\n"
    "                for bid in deferred_fence_block_ids:\n"
    "                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)\n"
    "                    if bid in self._current_batch_allocated_block_ids:\n"
    "                        self._current_batch_jobs_to_flush.add(job_id)\n"
)
SCHED_NEW = (
    "            if req.is_finished():\n"
    "                # Register non-sliding-window blocks for flush detection.\n"
    "                for bid in deferred_fence_block_ids:\n"
    "                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)\n"
    "                    # patch_offload_lazy_commit: under lazy commit, do not self-flush a\n"
    "                    # finished request's own blocks; the flush-on-reallocation path fences\n"
    "                    # the job the first time one of these blocks is actually reused.\n"
    "                    if (not _LAZY_COMMIT) and bid in self._current_batch_allocated_block_ids:\n"
    "                        self._current_batch_jobs_to_flush.add(job_id)\n"
)

TAIL = '''

# ---- RADIANCE offload lazy-commit (patch_offload_lazy_commit) --------------------------------
import os as _lc_os

_LAZY_COMMIT = _lc_os.environ.get("RADIANCE_OFFLOAD_LAZY_COMMIT", "0") == "1"
'''


def edit(path, old, new, tail=False):
    src = path.read_text()
    if MARK in src:
        print(f"[offload-lazy] {path.name} already applied")
        return
    n = src.count(old)
    if n != 1:
        print(f"[offload-lazy] {path.name}: anchor matched {n}x, NOT applied", file=sys.stderr)
        raise SystemExit(1)
    src = src.replace(old, new, 1)
    if tail:
        src += TAIL
    ast.parse(src)
    path.write_text(src)
    print(f"[offload-lazy] applied: {path.name}")


edit(WORKER, WORKER_OLD, WORKER_NEW, tail=True)
edit(SCHED, SCHED_OLD, SCHED_NEW, tail=True)
