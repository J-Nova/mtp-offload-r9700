"""Drive the served target over prompts.jsonl so the capture hook records its own generations.
Server sampling defaults apply (prod: temperature 0.7, top_p 0.95, top_k 20, reasoning on).

Target override via env, so this works against any serve-mxfp4.sh target (not just PARO on :8080):
  BENCH_URL=http://localhost:8000/v1/chat/completions   BENCH_MODEL=<served model name>
"""
import json, os, sys, time, urllib.request, concurrent.futures as cf
prompts, out, conc, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
URL = os.environ.get("BENCH_URL", "http://localhost:8080/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-PARO-MXFP4")
rows = [json.loads(l) for l in open(prompts)]
done = set()
try:
    for l in open(out): done.add(json.loads(l)["id"])
except FileNotFoundError: pass
todo = [r for r in rows if r["id"] not in done]
print(f"{len(todo)} prompts to run ({len(done)} done), conc {conc}, max_tokens {max_tokens}", flush=True)
def one(r):
    body = json.dumps({"model": MODEL, "messages": r.get("messages") or [{"role": "user", "content": r["prompt"]}], "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        ch = d["choices"][0]; msg = ch["message"]
        return {"id": r["id"], "source": r["source"], "usage": d.get("usage"), "finish": ch.get("finish_reason"),
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content"), "content": msg.get("content"), "secs": round(time.time() - t0, 1)}
    except Exception as e:  # noqa
        return {"id": r["id"], "source": r["source"], "error": repr(e)}
t0 = time.time(); n = 0; toks = 0
with open(out, "a") as f, cf.ThreadPoolExecutor(conc) as ex:
    for res in ex.map(one, todo):
        f.write(json.dumps(res) + "\n"); f.flush(); n += 1
        toks += (res.get("usage") or {}).get("completion_tokens", 0)
        if n % 10 == 0 or n == len(todo):
            el = max(time.time() - t0, 1e-9)
            eta = (len(todo) - n) * el / n
            print(f"  [gen] {n}/{len(todo)} ({100*n/max(len(todo),1):.0f}%) "
                  f"{toks} tok, {toks/el:.0f} tok/s, {el/60:.1f} min, ETA {eta/60:.1f} min",
                  flush=True)
print(f"DONE {n} responses, {toks} completion tokens in {(time.time()-t0)/60:.1f} min", flush=True)
