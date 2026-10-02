#!/usr/bin/env bash
# N15 A/B throughput probe for the int2 verify head. Run once per arm, after that arm's restart.
# Prints the verify-head status line and decode tok/s for three regimes plus acceptance.
#
#   ./aijuus/n15_ab.sh armA
#
# Card/thermal caveat: always run on the same instance (default vllm-1) and report junction temp
# alongside, since GPU0/GPU1 differ. Use a single restart between arms; do not change other knobs.
set -u
BASE=${BASE:-http://172.18.0.21:8001}
MODEL=${MODEL:-mtp-27B-MXFP4-blend}
KEY=juup-123
LABEL=${1:-arm}
C=${C:-$(docker ps --format '{{.Names}}' | grep -m1 '^vllm-1')}

say(){ printf '\n===== %s =====\n' "$1"; }

say "verify-head status ($LABEL)"
docker logs "$C" 2>&1 | grep -aE "\[radiance.verifyhead\]" | tail -3 || true

run_one(){ # prompt temp maxtok -> prints tok/s
  local prompt="$1" temp="$2" mt="$3"
  local t0 t1 body
  body=$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"temperature":%s,"top_p":0.95,"top_k":20,"max_tokens":%d,"chat_template_kwargs":{"enable_thinking":false}}' \
    "$MODEL" "$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$prompt")" "$temp" "$mt")
  t0=$(date +%s.%N)
  curl -s -m 180 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$body" \
    "$BASE/v1/chat/completions" -o /tmp/kilo/n15_${LABEL}_one.json
  t1=$(date +%s.%N)
  python3 -c "import json;d=json.load(open('/tmp/kilo/n15_${LABEL}_one.json'));ct=d['usage']['completion_tokens'];dt=$t1-$t0;print(f'{ct/dt:.1f} tok/s  ({ct} tok / {dt:.2f}s)')"
}

say "bs1 greedy/predictable (400 tok)"
run_one "Count from 1 to 400, one number per line, no other text." 0 400
say "bs1 sampled/high-entropy (400 tok)"
run_one "Produce a long sequence of unrelated random words, one per line; do not repeat." 0.7 400

say "bs8 sampled (8x300 tok, aggregate)"
t0=$(date +%s.%N)
seq 8 | xargs -P8 -I{} curl -s -m 180 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of computing.\"}],\"temperature\":0.7,\"top_p\":0.95,\"top_k\":20,\"max_tokens\":300}" \
  "$BASE/v1/chat/completions" -o /tmp/kilo/n15_${LABEL}_bs8_{}.json
t1=$(date +%s.%N)
tot=$(for f in /tmp/kilo/n15_${LABEL}_bs8_*.json; do python3 -c "import json;print(json.load(open('$f'))['usage']['completion_tokens'])" 2>/dev/null; done | paste -sd+ | bc)
python3 -c "print(f'{$tot/($t1-$t0):.1f} agg tok/s  ($tot tok / {$t1-$t0:.2f}s)')"

say "thermal snapshot ($C)"
rocm-smi --showtemp --showclocks 2>/dev/null | grep -E "GPU\[|junction|sclk clock level" | head -6
