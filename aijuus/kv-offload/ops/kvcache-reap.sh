#!/usr/bin/env bash
# Reaper for the vLLM fs secondary KV tier -- THE EVICTION POLICY.
#
# vllm/v1/kv_offload/tiering/fs/manager.py has no capacity, quota or TTL parameter,
# and SecondaryTierManager exposes no eviction hook: the tier writes and never
# deletes. Without this timer the filesystem fills and every subsequent store fails.
# This is not a tuning script; it is the missing eviction policy.
#
# SAFETY. Deleting a block a load is about to read normally just misses (lookup
# stats the filesystem), but between lookup and load there is a window. With
# patch_offload_fs_tier.py (RADIANCE_FS_FAILED_LOAD_FORGET=1) a load that hits a
# deleted file fails its fs->CPU promotion and the request recomputes that block --
# a hung request becomes a recompute. MIN_AGE_MIN is the hard floor no rule,
# capacity pressure included, may cross: recent blocks are the engine's hot set.
#
# Ordering is oldest-mtime-first. The tier writes each block once and never
# modifies it, so mtime is insertion time = FIFO, which for a prefix cache
# approximates LRU at zero I/O cost. (atime-LRU needs a relatime mount; the
# volume is noatime today.)
#
# Adapted from zzpanic/qwen3.6-vllm-gfx1201-launchers `kv-cache/ops/kvcache-reap.sh`.
set -euo pipefail

ROOT=${KVCACHE_ROOT:-/kvcache/blocks}
MIN_AGE_MIN=${KVCACHE_MIN_AGE_MIN:-90}      # HARD FLOOR: nothing younger is ever deleted
MAX_AGE_HOURS=${KVCACHE_MAX_AGE_HOURS:-0}   # 0 = age rule OFF, capacity alone governs
TARGET_PCT=${KVCACHE_TARGET_PCT:-65}        # keep %use at or below this (when KVCACHE_MAX_GIB=0)
MAX_GIB=${KVCACHE_MAX_GIB:-0}                # hard cap on the KV dir itself, GiB (0 = off, use %)
MAX_MB=${KVCACHE_MAX_MB:-0}                  # same, in MiB (for small deploys / testing)
MAX_DELETE=${KVCACHE_MAX_DELETE:-4000}      # per-run cap; next cycle continues
EMERGENCY_PCT=${KVCACHE_EMERGENCY_PCT:-90}  # above this after Stage B, Stage C crosses the floor
EMERGENCY_MIN_AGE_MIN=${KVCACHE_EMERGENCY_MIN_AGE_MIN:-1}
TMP_AGE_MIN=${KVCACHE_TMP_AGE_MIN:-60}      # orphaned *.tmp older than this go too
DRY=${DRY_RUN:-}

[ -d "$ROOT" ] || { echo "kvcache-reap: $ROOT does not exist; nothing to do" >&2; exit 0; }

# A floor at or above the ceiling would delete what the floor forbids -- refuse.
# MAX_AGE_HOURS=0 means the age rule is off, so the guard does not apply (without
# this exemption the script would do NO reaping, the dangerous failure on a filling volume).
if [ "$MAX_AGE_HOURS" -gt 0 ] && [ "$MIN_AGE_MIN" -ge $(( MAX_AGE_HOURS * 60 )) ]; then
  echo "kvcache-reap: MIN_AGE_MIN=${MIN_AGE_MIN}min >= MAX_AGE_HOURS=${MAX_AGE_HOURS}h; refusing to run" >&2
  exit 1
fi

pct() { df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9'; }

# Bytes actually used by KV blocks. Only run when the byte cap is set: over a few
# thousand hash-named files it is cheap, but it is not free at every cycle.
dir_bytes() { find "$ROOT" -type f -name '*.bin' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }

# A crashed store leaves <name>.bin.tmp<suffix> behind (io.py writes a temp then
# os.replace()s it). Those are never read and never reaped by vLLM.
if [ -z "$DRY" ]; then
  find "$ROOT" -type f -name '*.tmp*' -mmin "+$TMP_AGE_MIN" -delete 2>/dev/null || true
else
  find "$ROOT" -type f -name '*.tmp*' -mmin "+$TMP_AGE_MIN" -printf 'would remove stale tmp %p\n' 2>/dev/null || true
fi

removed=0; capped=0; floor_hit=0
# Loops read from fd 3 via process substitution so the body runs in this shell and
# `removed` survives. A pipe would not.
rm_one() {
  if [ -n "$DRY" ]; then echo "would remove $1"; else rm -f -- "$1" || return 0; fi
  removed=$((removed+1))
}

# --- Stage A: age (OFF by default). Runs every cycle when enabled.
if [ "$MAX_AGE_HOURS" -gt 0 ]; then
  while IFS= read -r -d '' -u 3 f; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    [ -f "$f" ] || continue
    rm_one "$f"
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$(( MAX_AGE_HOURS * 60 ))" -print0)
fi
aged=$removed

# --- Stage A2: hard byte cap (KVCACHE_MAX_GIB > 0). Bounds the fs tier to a fixed
# --- number of GiB on a SHARED disk -- the only in-script way to "reserve" a budget
# --- when the tier is not on a dedicated volume. Oldest-first, and it MAY cross the
# --- min-age floor when the cap is exceeded (deleting a young block costs a
# --- recompute, not a hang, thanks to patch_offload_fs_tier.py), sparing only the
# --- last EMERGENCY_MIN_AGE_MIN so nothing in flight is touched. When MAX_GIB>0 it
# --- replaces the %-of-filesystem rule below; use a dedicated volume for a HARD
# --- bound (see ops/make-kvcache-volume.sh) and keep MAX_GIB as the soft target.
if [ "$MAX_GIB" -gt 0 ] && [ "$capped" -eq 0 ]; then
  cap_bytes=$(( MAX_GIB * 1024 * 1024 * 1024 ))
elif [ "$MAX_MB" -gt 0 ] && [ "$capped" -eq 0 ]; then
  cap_bytes=$(( MAX_MB * 1024 * 1024 ))
else
  cap_bytes=0
fi
if [ "$cap_bytes" -gt 0 ]; then
  used=$(dir_bytes)
  if [ "$used" -gt "$cap_bytes" ]; then
    deficit=$(( used - cap_bytes ))
    freed=0
    while IFS= read -r -d '' -u 3 line; do
      if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
      read -r _rad_mtime size f <<< "$line"
      [ -f "$f" ] || continue
      rm_one "$f"
      freed=$(( freed + size ))
      [ "$freed" -ge "$deficit" ] && break
    done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$EMERGENCY_MIN_AGE_MIN" -printf '%T@ %s %p\0' | sort -z -n 2>/dev/null)
  fi
fi

# --- Stage B: capacity (% of filesystem). Only when the byte cap is off. Deleting
# --- oldest-first, NEVER below the MIN_AGE_MIN floor -- that restriction is the whole
# --- safety argument, so it is expressed in the find, not in a check an edit could drop.
cur=$(pct)
if [ "$cap_bytes" -eq 0 ] && [ "$cur" -gt "$TARGET_PCT" ] && [ "$capped" -eq 0 ]; then
  while IFS= read -r -d '' -u 3 line; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    f=${line#* }
    [ -f "$f" ] || continue
    rm_one "$f"
    if [ $(( (removed - aged) % 200 )) -eq 0 ]; then
      cur=$(pct)
      [ "$cur" -le "$TARGET_PCT" ] && break
    fi
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$MIN_AGE_MIN" -printf '%T@ %p\0' | sort -z -n 2>/dev/null)
  cur=$(pct)
  [ -z "$DRY" ] && [ "$cur" -gt "$TARGET_PCT" ] && floor_hit=1
fi

# --- Stage C: emergency. The floor held and the volume is nearly full, so stores
# --- are about to fail (ENOSPC) for everything. Delete below the floor, sparing
# --- only the last EMERGENCY_MIN_AGE_MIN. Safe because a deleted-block load is a
# --- recompute, not a hang (patch_offload_fs_tier.py).
emergency=0
cur=$(pct)
if [ "$EMERGENCY_PCT" -gt 0 ] && [ "$cur" -gt "$EMERGENCY_PCT" ] && [ "$capped" -eq 0 ]; then
  emergency=1
  while IFS= read -r -d '' -u 3 line; do
    if [ "$removed" -ge "$MAX_DELETE" ]; then capped=1; break; fi
    f=${line#* }
    [ -f "$f" ] || continue
    rm_one "$f"
    if [ $(( removed % 200 )) -eq 0 ]; then
      cur=$(pct)
      [ "$cur" -le "$TARGET_PCT" ] && break
    fi
  done 3< <(find "$ROOT" -type f -name '*.bin' -mmin "+$EMERGENCY_MIN_AGE_MIN" -printf '%T@ %p\0' | sort -z -n 2>/dev/null)
  cur=$(pct)
  [ -z "$DRY" ] && [ "$cur" -le "$TARGET_PCT" ] && floor_hit=0
fi

# Directory fan-out is <hhh>/<hh>_g<group>/, so emptied dirs accumulate.
if [ -z "$DRY" ] && [ "$removed" -gt 0 ]; then
  find "$ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

now=$(pct)
if [ "$removed" -eq 0 ]; then
  if [ "$cap_bytes" -gt 0 ]; then
    echo "kvcache-reap: ${now}% fs used, under the ${MAX_GIB:-$(( cap_bytes / 1024 / 1024 / 1024 ))}GiB byte cap; nothing to do"
  elif [ "$MAX_AGE_HOURS" -gt 0 ]; then
    echo "kvcache-reap: ${now}% used, nothing older than ${MAX_AGE_HOURS}h and at/below the ${TARGET_PCT}% target; nothing to do"
  else
    echo "kvcache-reap: ${now}% used, at/below the ${TARGET_PCT}% target (age rule off); nothing to do"
  fi
else
  echo "kvcache-reap: removed $removed block(s) [${aged} by age, $((removed - aged)) for capacity]; now ${now}% used"
fi
[ "$capped" -eq 1 ] && echo "kvcache-reap: stopped at the ${MAX_DELETE}-block per-run cap; the next cycle continues"
final_bytes=0
if [ "$cap_bytes" -gt 0 ]; then
  final_bytes=$(dir_bytes)
  echo "kvcache-reap: KV dir $(( final_bytes / 1024 / 1024 )) MiB / cap $(( cap_bytes / 1024 / 1024 )) MiB"
fi
[ "$emergency" -eq 1 ] && echo "kvcache-reap: EMERGENCY stage ran (volume above ${EMERGENCY_PCT}% with every block under ${MIN_AGE_MIN} min); deleted below the floor, sparing the last ${EMERGENCY_MIN_AGE_MIN} min" >&2
if [ "$floor_hit" -eq 1 ]; then
  echo "kvcache-reap: WARNING ${now}% used is still above the ${TARGET_PCT}% target, but every remaining block is" >&2
  echo "kvcache-reap:   younger than ${MIN_AGE_MIN} min. NOT deleting those -- recent blocks are the engine's hot set." >&2
  echo "kvcache-reap:   If this repeats, the volume is too small: grow it, lower KVCACHE_MIN_AGE_MIN, or set" >&2
  echo "kvcache-reap:   KVCACHE_MAX_AGE_HOURS to shed old blocks before pressure builds." >&2
fi

# Hard-cap enforcement. The byte cap is a CEILING, not a target: if a run ends still
# above it (the per-run delete budget was reached, or an emergency delete was needed),
# say so loudly and exit non-zero so systemd marks the unit failed and any OnFailure /
# alerting fires. This is what "the cap MUST hold" means operationally -- a silently
# over-cap tier is what fills the shared disk and stalls stores.
if [ "$cap_bytes" -gt 0 ] && [ "$final_bytes" -gt "$cap_bytes" ]; then
  over=$(( final_bytes - cap_bytes ))
  echo "kvcache-reap: ALERT fs KV tier is OVER its ${MAX_GIB}GiB hard cap by $(( over / 1024 / 1024 )) MiB after this run" >&2
  echo "kvcache-reap:   ($(( final_bytes / 1024 / 1024 )) MiB > $(( cap_bytes / 1024 / 1024 )) MiB). The next cycle continues; if this repeats," >&2
  echo "kvcache-reap:   raise KVCACHE_MAX_DELETE, shorten the timer interval, or lower KVCACHE_MIN_AGE_MIN." >&2
  exit 2
fi
exit 0
