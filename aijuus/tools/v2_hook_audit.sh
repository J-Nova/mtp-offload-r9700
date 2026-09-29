#!/usr/bin/env bash
# V2-hook audit (TCCLA plan, Phase 0.5 item 1) + AOT env-key gate check (item 2).
#
# Proves our runtime hooks actually FIRE on the vLLM 0.29 V2 runner (install != invocation).
# Static side: the installed image has NO V1 runner (vllm/worker/model_runner.py absent), so every
# hook/patched path must be V2; this script checks the boot log for each hook's runtime marker.
#
# Usage:  aijuus/tools/v2_hook_audit.sh [container]     (default: first running vllm-0)
#
# Item 2 nuance (aot env-key): the gate is correct if the AOT dir name carries the sorted RADIANCE_*
# env key. Run once, note the "env key XXXX" hash and the "new AOT dir" lines; after a restart with
# the SAME env it must NOT print "new AOT dir" again (cache reuse), and changing any RADIANCE_* must
# change XXXX.
set -u
C="${1:-$(docker ps --format '{{.Names}}' | grep -m1 '^vllm-0')}"
if [ -z "${C:-}" ]; then echo "no vllm-0 container found; pass a container name"; exit 1; fi
echo "container: $C"
LOG="$(docker logs "$C" 2>&1)"

chk() { # name regex
  local line
  line="$(printf '%s\n' "$LOG" | grep -E -m1 "$2" || true)"
  if [ -n "$line" ]; then
    printf 'PASS  %-38s %s\n' "$1" "$(printf '%s' "$line" | sed -E 's/^.*(\[radiance\]|\[aot-envkey\]|\[patch_[a-z]+\]) ?//')"
  else
    printf 'FAIL  %-38s (no match: %s)\n' "$1" "$2"
  fi
}

echo "--- item 2: AOT second-compile-cache env key ---"
chk "aot-envkey applied"          '\[aot-envkey\] applied'
printf '      %s\n' "$(printf '%s\n' "$LOG" | grep -oE 'env key [0-9a-f]+' | tail -1 || true)"
printf '      new AOT dir this boot: %s\n' "$(printf '%s\n' "$LOG" | grep -cE 'new AOT dir' || true)"

echo "--- runtime hooks (must fire on V2) ---"
chk "preshuffle load hook"        'preshuffle weight-shuffle-at-load hook installed'
chk "attn tuned-config override"  'attn tuned-config override installed on'
chk "draft head armed"            'int[0-9] draft head armed'
chk "draft head quantised"       'INT[0-9]_DRAFT_HEAD'
chk "draft vocab applied"         'DRAFT_VOCAB: [0-9]+ of [0-9]+ rows'
chk "draft controller ON"         'RADIANCE_DYNAMIC_DRAFT=ON'
chk "custom all-reduce"           'fast-reduce hook armed|custom all-reduce INSTALLED'
chk "R4D report (V2 Worker)"      'R4D kernel selection: libr4d'
chk "MXFP4 kernel selection"      '\[radiance\] .*mxfp4|RADIANCE_MXFP4'

echo "--- runner layout ---"
docker exec "$C" sh -c 'P=/opt/vllm/lib/python3.12/site-packages; \
  for f in vllm/worker/model_runner.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu_worker.py; do \
    if [ -f "$P/$f" ]; then echo "present  $f"; else echo "ABSENT   $f"; fi; done'
