#!/usr/bin/env python3
"""Make the filesystem secondary tier production-safe: fan the I/O out, keep the
async-lookup verdict honest, and forget a failed load instead of re-promoting it forever.

A port of three sibling patches from zzpanic/qwen3.6-vllm-gfx1201-launchers
(`kv-cache/patches/patch_offload_fs_fanout.py`, `patch_lookup_invalidate.py`,
`patch_fs_failed_load.py`), re-anchored to this tree, applied in dependency order.

WHY THE fs TIER NEEDS THIS

The fs tier is the only way to get cross-instance or restart-surviving KV reuse
here (both instances can share one `root_dir`; block filenames are content-hashed
with a deterministic seed). But it ships with three defects that only show up under
real load:

  1. NO FAN-OUT. Both submit paths end in `enqueue_*(job_id, 1, [task])`; one job
     is one task, and `batch_load_block`/`batch_store_block` are serial loops. The
     8 read / 4 write threads are never used, because the C extension only releases
     the GIL (no io_uring/aio). A multi-chunk promotion reads ~200 x 27 MB files at
     queue depth 1 -- the 64 s promotion. Fix: split the job into partials across
     the pool (upstream PR #49225; the pool already supports n_tasks per job).
  2. A STALE `absent` VERDICT. `AsyncLookupManager` caches one existence verdict
     per key until every referencing request finishes. A request looks up K
     (cached absent), computes and STORES K, and every later lookup of K keeps
     returning absent -- a MISS that is a lie, forcing a full recompute. Fix:
     `invalidate()` drops the stale `absent` when a store LANDS (drop, do not flip
     to present: an external reaper can delete the file, so re-stat is the honest
     answer).
  3. A FAILED LOAD RE-PROMOTES FOREVER. A failed fs read fails the fs->CPU
     promotion; `cpu/manager.py complete_store(success=False)` drops the
     half-written CPU block, so `worker.py:361` is NOT reached (the request does not
     crash the engine). But nothing tells the lookup cache, so the request promotes
     the same missing file again and again until the client gives up -- a hung
     request, 240 s / 294 failed reads measured. Fix: `forget()` drops the cached
     verdict for the failed keys, so the next lookup re-states the file, sees it is
     gone, and the request recomputes that block. This is what makes an external
     reaper safe: deleting a young block costs a recompute, not a hang.

THE REAPER is still mandatory: the tier has no capacity/quota/TTL and no eviction
hook. `ops/kvcache-reap.sh` (+ .service/.timer) is the eviction policy; its
MIN_AGE floor is the safety argument, and defect 3 is what lets the floor be
crossed under real capacity pressure without hanging requests.

GATES (all default 0 = upstream behaviour, so the patch is a no-op until enabled):
  RADIANCE_FS_FANOUT_TARGET_MB (32)  byte budget for the batch count
  RADIANCE_FS_FANOUT_MAX       (0)   flat cap; 1 restores one-task-per-job exactly
  RADIANCE_LOOKUP_INVALIDATE   (0)   drop the stale `absent` when a store lands
  RADIANCE_LOOKUP_STALE_WATCH  (0)   diagnostic: count cached-absent-that-exist-lies
  RADIANCE_FS_FAILED_LOAD_FORGET (0) forget the verdict of a failed load's keys
For production with the fs tier, set INVALIDATE=1 and FAILED_LOAD_FORGET=1 (the
launcher does when KV_OFFLOAD_DISK_DIR is set). Idempotent; run once pre-serve.
"""
import os
import sys
import sysconfig
from pathlib import Path

# _patchlib.py lives at the repo root (four levels up from patches/); insert it
# explicitly since sys.path[0] is aijuus/ when run as `python3 aijuus/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from _patchlib import apply  # noqa: E402

SP = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"]))
ASYNC_LOOKUP = SP / "vllm" / "v1" / "kv_offload" / "tiering" / "async_lookup.py"
FS_MANAGER = SP / "vllm" / "v1" / "kv_offload" / "tiering" / "fs" / "manager.py"

print("[radiance] fs tier: fan-out + lookup invalidation + failed-load forget")


# =====================================================================================
# 1. FAN-OUT (upstream PR #49225): knock knobs + helpers, thread counts, splitter, paths.
# =====================================================================================

apply(
    FS_MANAGER,
    anchor="from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool",
    new='''from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# radiance R3.14 (upstream PR #49225): how far to fan a single fs job out across the
# pool. TARGET_MB is the byte budget that decides the batch count; MAX overrides it
# with a flat cap (MAX=1 restores the old one-task-per-job behaviour exactly).
_RADIANCE_FANOUT_TARGET_BYTES = int(
    float(os.environ.get("RADIANCE_FS_FANOUT_TARGET_MB", "32")) * (2**20)
)
_RADIANCE_FANOUT_MAX = int(os.environ.get("RADIANCE_FS_FANOUT_MAX", "0"))
# radiance: invalidate the per-key async-lookup cache when a fs store LANDS. A FAILED
# store leaves the verdict (the block genuinely is not on disk, so `absent` is correct).
_RADIANCE_LOOKUP_INVALIDATE = os.environ.get("RADIANCE_LOOKUP_INVALIDATE", "0") == "1"
# radiance diagnostic (off by default): a cached `absent` that is actually on disk is a
# MISS that is a lie. Costs one faccessat per cached-absent lookup; do not leave on.
_RADIANCE_LOOKUP_STALE_WATCH = os.environ.get("RADIANCE_LOOKUP_STALE_WATCH", "0") == "1"
# radiance: forget the lookup verdict of keys whose fs load failed, so the request
# recomputes instead of re-promoting a file that is gone (see the module docstring).
_RADIANCE_FAILED_LOAD_FORGET = os.environ.get("RADIANCE_FS_FAILED_LOAD_FORGET", "0") == "1"


def _radiance_fanout_degree(
    n_blocks: int,
    n_threads: int,
    total_threads: int,
    inflight_jobs: int,
    block_size: int,
) -> int:
    """How many batches to split a job of n_blocks into.

    Splitting only pays while the device is not already saturated, and once several
    jobs are in flight the threads are busy anyway -- so the budget is divided by the
    in-flight job count. Beyond that, extra batches buy queue entries and wake-ups
    without moving more bytes.
    """
    if n_blocks <= 1 or n_threads <= 1:
        return 1
    if _RADIANCE_FANOUT_MAX > 0:
        budget = _RADIANCE_FANOUT_MAX
    elif block_size > 0:
        budget = -(-_RADIANCE_FANOUT_TARGET_BYTES // block_size)
    else:
        budget = total_threads
    budget = min(budget, total_threads)
    jobs = max(1, inflight_jobs)
    return max(1, min(-(-budget // jobs), n_blocks, n_threads))


def _radiance_batches(n_items: int, n_batches: int):
    """Yield (start, stop) slices splitting n_items evenly, largest remainder first."""
    q, r = divmod(n_items, n_batches)
    start = 0
    for i in range(min(n_items, n_batches)):
        stop = start + (q + 1 if i < r else q)
        yield start, stop
        start = stop''',
    sentinel="_RADIANCE_FANOUT_TARGET_BYTES",
    label="1a fs manager: fanout knobs and helpers",
)

apply(
    FS_MANAGER,
    anchor='''        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )''',
    new='''        # radiance R3.14: kept so submit_store/submit_load can size the fanout.
        self._radiance_n_read_threads = n_read_threads
        self._radiance_n_write_threads = n_write_threads
        self._radiance_total_threads = max(1, n_read_threads + n_write_threads)
        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )''',
    sentinel="_radiance_total_threads",
    label="1b fs manager: record the pool thread counts",
)

apply(
    FS_MANAGER,
    anchor="    @override\n    def submit_store(self, job_metadata: JobMetadata) -> None:",
    new='''    def _radiance_split(self, io_fn, paths, offsets, n_threads):
        """radiance R3.14: turn one job into a list of partials, one per batch.

        Returns a single full-range partial for jobs of 0 or 1 blocks, which is
        byte-for-byte the old behaviour -- an empty list would build a JobState that
        never completes and would hang drain_jobs.
        """
        total = len(paths)
        if total <= 1:
            return [
                functools.partial(
                    io_fn,
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            ]
        n_batches = _radiance_fanout_degree(
            total,
            n_threads if n_threads > 0 else self._radiance_total_threads,
            self._radiance_total_threads,
            getattr(self._pool, "_inflight_jobs", 0),
            self._block_size,
        )
        return [
            functools.partial(
                io_fn,
                paths[a:b],
                self._primary_kv_view,
                offsets[a:b],
                self._block_size,
                self._use_o_direct,
            )
            for a, b in _radiance_batches(total, n_batches)
        ]

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:''',
    sentinel="def _radiance_split",
    label="1c fs manager: _radiance_split helper",
)

apply(
    FS_MANAGER,
    anchor='''        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])''',
    new='''        tasks = self._radiance_split(  # radiance R3.14
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._radiance_n_write_threads,
        )
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)''',
    sentinel="tasks = self._radiance_split(  # radiance R3.14\n            batch_store_block,",
    label="1d fs manager: fan the store path out",
)

apply(
    FS_MANAGER,
    anchor='''        task = functools.partial(
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )

        self._pool.enqueue_load(job_metadata.job_id, 1, [task])''',
    new='''        tasks = self._radiance_split(  # radiance R3.14
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._radiance_n_read_threads,
        )
        self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)''',
    sentinel="tasks = self._radiance_split(  # radiance R3.14\n            batch_load_block,",
    label="1e fs manager: fan the load path out",
)


# =====================================================================================
# 2. LOOKUP INVALIDATION: drop a stale `absent` when a store lands.
# =====================================================================================

apply(
    ASYNC_LOOKUP,
    anchor='''    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]''',
    new='''    def invalidate(self, keys: "Iterable[OffloadKey]") -> None:
        """Drop the cached verdict for ``keys`` so the next lookup re-states it.

        Called by a tier when a STORE for these keys completes successfully: the
        block is now on disk, so a previously-cached `absent` verdict is stale.
        Only `absent` (False) entries are dropped -- a `present` entry is left
        (re-stating it would just re-confirm True) and an in-flight (None) entry
        is already being checked. Dropping (not flipping to True) keeps the
        answer honest under external eviction: the next lookup re-states the
        file, so the verdict reflects its actual current state. Scheduler-thread only.
        """
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None and state.result is False:
                del self._lookup_state[key]

    def forget(self, keys: "Iterable[OffloadKey]") -> None:
        """Drop the cached verdict for ``keys`` whatever it is (radiance).

        Called by a tier when a LOAD of these keys FAILED: the cached `present`
        has been disproved (the file was reaped, short, or unreadable, and the
        tier deleted it). Left in place it makes the waiting request promote the
        same missing file forever. An in-flight (None) entry is left -- it is
        already being re-stated. Scheduler-thread only.
        """
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None and state.result is not None:
                del self._lookup_state[key]

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            # A key may have been dropped by invalidate()/forget() and not yet
            # re-looked-up, so its entry can be absent here. This guard is a no-op
            # when nothing was dropped (e.g. the fix gates are unset).
            state = self._lookup_state.get(key)
            if state is None:
                continue
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]''',
    sentinel="def forget(self, keys",
    label="2a async_lookup: invalidate() + forget() + guarded cleanup()",
)

apply(
    FS_MANAGER,
    anchor='''        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}''',
    new='''        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # radiance: keys of in-flight store jobs, tracked ALWAYS so
        # get_finished_jobs() can invalidate the async-lookup cache when a store
        # lands, independent of whether KV events are enabled.
        self._store_lookup_keys: dict[JobId, list[OffloadKey]] = {}
        # radiance: load job -> keys, so a FAILED load can forget its verdicts.
        self._radiance_load_keys: dict[JobId, list[OffloadKey]] = {}
        # radiance diagnostic: lookups that returned a cached `absent` for a key
        # whose file now exists (a "lie"). Gated on RADIANCE_LOOKUP_STALE_WATCH.
        self._radiance_stale_lie = 0
        if _RADIANCE_LOOKUP_INVALIDATE:
            logger.info(
                "radiance: RADIANCE_LOOKUP_INVALIDATE on -- a stale `absent` "
                "lookup verdict is dropped when a store for it lands"
            )
        else:
            logger.warning(
                "radiance: RADIANCE_LOOKUP_INVALIDATE unset -- the fs "
                "async-lookup cache keeps a stale `absent` after a store lands "
                "(upstream behaviour); a later lookup of a just-stored block can "
                "MISS and force a full recompute. Set it to 1 to drop the stale "
                "verdict on store completion."
            )''',
    sentinel="self._store_lookup_keys: dict[JobId, list[OffloadKey]] = {}",
    label="2b fs manager __init__: tracking maps, diagnostic counter, gate warning",
)

apply(
    FS_MANAGER,
    anchor='''        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    new='''        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        # radiance: always-on (independent of events) so get_finished_jobs() can
        # invalidate the async-lookup cache when this store lands.
        self._store_lookup_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    sentinel="self._store_lookup_keys[job_metadata.job_id] = list(job_metadata.keys)",
    label="2c fs manager submit_store: track store keys unconditionally",
)

apply(
    FS_MANAGER,
    anchor='''            results.append(JobResult(job_id=job_id, success=success))
        return results''',
    new='''            # radiance: a successful store put these blocks on disk, so any
            # previously-cached `absent` verdict for them is stale. Drop it (the
            # next lookup re-states the file). A FAILED store leaves the verdict
            # untouched. Also forget the verdict of a FAILED load's keys so the
            # request recomputes instead of re-promoting a missing file forever.
            store_keys = self._store_lookup_keys.pop(job_id, None)
            if store_keys and success and _RADIANCE_LOOKUP_INVALIDATE:
                self._lookup_manager.invalidate(store_keys)
            load_keys = self._radiance_load_keys.pop(job_id, None)
            if load_keys and not success and _RADIANCE_FAILED_LOAD_FORGET:
                self._lookup_manager.forget(load_keys)
            results.append(JobResult(job_id=job_id, success=success))
        return results''',
    sentinel="store_keys = self._store_lookup_keys.pop(job_id, None)",
    label="2d fs manager get_finished_jobs: invalidate on store, forget on failed load",
)

apply(
    FS_MANAGER,
    anchor='''    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS''',
    new='''    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        # radiance diagnostic (RADIANCE_LOOKUP_STALE_WATCH): a cached `absent`
        # that is actually on disk now is a MISS that is a lie. Inert when unset.
        # DO NOT leave this on in production: it fires on EVERY miss, one
        # unbatched os.path.exists each -- most expensive during the cold-cache
        # warm-up, exactly when the tier cannot serve yet.
        if (
            _RADIANCE_LOOKUP_STALE_WATCH
            and result is False
            and os.path.exists(self.file_mapper.get_file_name(key))
        ):
            self._radiance_stale_lie += 1
            if self._radiance_stale_lie <= 8 or self._radiance_stale_lie % 100 == 0:
                logger.warning(
                    "radiance stale-lookup LIE: cached `absent` for a key now "
                    "on disk (count=%d) -- a stored block is being missed",
                    self._radiance_stale_lie,
                )
        return LookupResult.HIT if result else LookupResult.MISS''',
    sentinel="self._radiance_stale_lie += 1",
    label="2e fs manager lookup: RADIANCE_LOOKUP_STALE_WATCH diagnostic",
)

# 2f. submit_load must record its keys so a failed load can forget them.
apply(
    FS_MANAGER,
    anchor='''    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = self._radiance_split(  # radiance R3.14''',
    new='''    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._radiance_load_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = self._radiance_split(  # radiance R3.14''',
    sentinel="self._radiance_load_keys[job_metadata.job_id] = list(job_metadata.keys)",
    label="2f fs manager submit_load: record keys for failed-load forget",
)

print(
    "[radiance] fs tier applied -- fanout target="
    f"{os.environ.get('RADIANCE_FS_FANOUT_TARGET_MB', '32')}MiB "
    f"max={os.environ.get('RADIANCE_FS_FANOUT_MAX', '0')} "
    f"invalidate={os.environ.get('RADIANCE_LOOKUP_INVALIDATE', '0')} "
    f"failed_load_forget={os.environ.get('RADIANCE_FS_FAILED_LOAD_FORGET', '0')}"
)
