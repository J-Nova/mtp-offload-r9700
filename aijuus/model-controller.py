#!/usr/bin/env python3
"""Model hot-swap controller for the 2-GPU Radiance deployment.

Watches /model-state/trigger.json (written by model-router when a request
arrives for a registered model no instance is ready to serve) and performs a
ONE-AT-A-TIME swap:

  1. pick an IDLE instance (no in-flight requests), so a swap does not drop
     active work; when both are busy, the less-busy one (the busier keeps
     serving). Idle picks rotate, so swaps spread across the cards.
  2. record it in state.json as {model, ready:false}
     -> model-router stops routing to it and (if another model is ready)
        serves its traffic with that model meanwhile, so the previously-loaded
        model keeps serving throughout (no error window)
  3. drain: wait up to IDLE_WAIT_SECONDS for its in-flight requests to finish
     (bounded, so a never-ending generation cannot block a swap forever)
  4. docker restart that container (its entrypoint re-reads state.json +
     model-registry.json and boots the new model)
  5. poll /health until ready, then flip ready:true
     -> model-router routes the new model to it

state.json also self-heals against /v1/models, so a crash or a manual
docker restart converges back to the real state. Only this process writes
state.json; the vllm containers mount it read-only and model-router only
writes trigger.json.

state.json schema (read by model-router):
    { "<instance>": {"model": "<name>", "ready": bool,
                     "endpoint": "http://host:port", "rank": <int>}, ... }
`endpoint`/`rank` let the router collapse tensor-parallel ranks (which share
an endpoint) into one routing target -- the path to TP=2 support later.
"""
import json
import os
import re
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STATE = "/model-state/state.json"
TRIGGER = "/model-state/trigger.json"
USAGE = "/model-state/usage.json"   # written by model-router; may not exist yet
REGISTRY = "/model-registry.json"
SOCK = "/var/run/docker.sock"
API_KEY = os.environ.get("VLLM_API_KEY", "")
DEFAULT_MODEL = os.environ.get("MODEL_DEFAULT", "")
SWAP_TIMEOUT = int(os.environ.get("SWAP_TIMEOUT", "3600"))
POLL = int(os.environ.get("POLL_SECONDS", "10"))
TICK = int(os.environ.get("TICK_SECONDS", "5"))
# Back off this long after a swap fails, so a bad registry entry cannot restart
# the target on every arriving request.
FAIL_COOLDOWN = int(os.environ.get("FAIL_COOLDOWN_SECONDS", "120"))
# Drain: before restarting an instance, wait up to IDLE_WAIT for its in-flight
# requests to finish (polled every IDLE_POLL), so a swap does not drop active
# work. 0 disables the wait (swap immediately).
IDLE_WAIT = int(os.environ.get("IDLE_WAIT_SECONDS", "300"))
IDLE_POLL = int(os.environ.get("IDLE_POLL_SECONDS", "5"))
# Admin HTTP endpoint (no auth beyond the API key):
#   POST /load  {"model": <key>, "instance": <name|"auto"|"all">, "force": bool}
#               -> load a DIFFERENT model onto instance(s) that are not already
#                  serving it (auto picks one idle such instance; "all" does
#                  both). Skips an instance already serving the model unless
#                  force:true. This CHANGES which model an instance serves.
#   POST /reload {"model": <key>, "instance": <name|"all">}
#               -> restart matching instance(s) so their entrypoint re-reads
#                  model-registry.json with FRESH settings (env/args) but keeps
#                  the SAME model -- no full redeploy.
#   GET /status / GET /health.
RELOAD_PORT = int(os.environ.get("RELOAD_PORT", "8101"))

# compose service name -> in-network base URL (+ tensor-parallel rank)
INSTANCES = {"vllm-0": "http://vllm-0:8000", "vllm-1": "http://vllm-1:8001"}
RANKS = {"vllm-0": 0, "vllm-1": 1}

_last_fail = {}   # model -> monotonic time of the last failed swap
_rr = 0           # rotating pick among equally-suitable (idle) instances
_op_lock = threading.Lock()  # serialize swaps/reloads: one restart op at a time


def entry(svc, model, ready):
    return {"model": model, "ready": ready,
            "endpoint": INSTANCES[svc], "rank": RANKS[svc]}


def served_to_key(reg):
    """Map each instance's advertised /v1/models name back to its registry key,
    so state["model"] is always the key the router matches requests against."""
    return {v.get("served_name", k): k for k, v in reg.items()}


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def load_registry():
    with open(REGISTRY) as f:
        reg = json.load(f)
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def read_usage():
    """Last-request epoch per model, written by model-router (MRU hint)."""
    try:
        with open(USAGE) as f:
            u = json.load(f)
            return u if isinstance(u, dict) else {}
    except Exception:
        return {}


def write_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, STATE)


def _dechunk(body):
    """Decode an HTTP chunked transfer-encoding body.

    Docker's unix-socket API replies with `Transfer-Encoding: chunked` for
    large responses (e.g. /containers/json), so the raw body starts with a hex
    chunk size followed by CRLF. Without this, json.loads(body) parses the
    leading size digits and fails with "Extra data"."""
    out = bytearray()
    i = 0
    while True:
        j = body.find(b"\r\n", i)
        if j == -1:
            break
        size = int(body[i:j].split(b";", 1)[0].strip(), 16)
        i = j + 2
        if size == 0:
            break
        out += body[i:i + size]
        i += size + 2
    return bytes(out)


def sock_request(method, path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    try:
        s.sendall(("%s %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
                   % (method, path, socket.gethostname())).encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1]) if head else 0
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        if b":" in line:
            k, _, v = line.partition(b":")
            headers[k.strip().lower()] = v.strip().lower()
    if headers.get(b"transfer-encoding", b"") == b"chunked":
        body = _dechunk(body)
    elif b"content-length" in headers:
        try:
            body = body[:int(headers[b"content-length"])]
        except ValueError:
            pass
    return status, body


def find_container(service):
    """Container id by compose service label (Coolify renames containers,
    so never rely on the bare service name)."""
    status, body = sock_request("GET", "/containers/json?all=1")
    if status != 200:
        return None
    for c in json.loads(body):
        if (c.get("Labels") or {}).get("com.docker.compose.service") == service:
            return c["Id"]
    return None


def _get(url):
    req = urllib.request.Request(url)
    if API_KEY:
        req.add_header("Authorization", "Bearer " + API_KEY)
    with urllib.request.urlopen(req, timeout=5) as r:
        # Read the body INSIDE the context: leaving `with` closes the response,
        # after which r.read() returns b'' (status stays readable, which is why
        # health_ok looked fine while instance_model silently saw nothing).
        return r.status, r.read()


def instance_model(base):
    """Model the instance actually serves right now (None if unreachable)."""
    try:
        data = json.loads(_get(base + "/v1/models")[1].decode())
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return models[0] if models else None
    except Exception:
        return None


_RUNNING_RE = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+(\S+)$")


def running_requests(base):
    """Sum of vllm:num_requests_running across label sets (0.0 if unavailable).

    vLLM emits the gauge with labels (e.g.
    `vllm:num_requests_running{engine="0",model_name="..."} 0.0`), so a plain
    `startswith(name + " ")` never matches -- parse tolerantly of the label
    block and sum all series.
    """
    total = 0.0
    try:
        for line in _get(base + "/metrics")[1].decode().splitlines():
            m = _RUNNING_RE.match(line)
            if m:
                total += float(m.group(1))
    except Exception:
        pass
    return total


def health_ok(base):
    try:
        return _get(base + "/health")[0] == 200
    except Exception:
        return False


def reconcile(st, reg):
    """Self-heal state against what the instances actually serve."""
    s2k = served_to_key(reg)
    changed = False
    for svc, base in INSTANCES.items():
        e = st.get(svc)
        actual = instance_model(base)          # advertised name, or None
        key = s2k.get(actual, actual) if actual else None
        if e is None:
            e = entry(svc, key or DEFAULT_MODEL, key is not None)
            st[svc] = e
            changed = True
        # keep endpoint/rank current (older state.json may lack them)
        if e.get("endpoint") != INSTANCES[svc] or e.get("rank") != RANKS[svc]:
            e["endpoint"] = INSTANCES[svc]
            e["rank"] = RANKS[svc]
            changed = True
        if key is None:
            if e.get("ready"):
                e["ready"] = False
                changed = True
            continue
        if e.get("model") != key:
            e["model"] = key
            e["ready"] = True
            changed = True
        elif not e.get("ready") and health_ok(base):
            e["ready"] = True
            changed = True
    return changed


def pick_target(st, usage=None):
    """Choose which instance to swap.

    Policy, in order:
      * do NOT evict the most-recently-used loaded model if another instance can
        host the new model without doing so (usage.json, written by
        model-router) -- avoids substituting a model people are actively using;
      * prefer an instance with NO in-flight requests (so the drain is instant);
      * rotate among equally-suitable instances so swaps spread across cards;
      * only when neither is idle, take the less-busy one (the busier keeps
        serving).
    """
    global _rr
    loads = {svc: running_requests(INSTANCES[svc]) for svc in INSTANCES}
    pool = ["vllm-0", "vllm-1"]
    if usage:
        mru = max(usage, key=usage.get)
        keep = [svc for svc in pool if st.get(svc, {}).get("model") != mru]
        if keep:
            pool = keep
    idle = [svc for svc in pool if loads[svc] <= 0]
    if idle:
        svc = idle[_rr % len(idle)]
        _rr += 1
        return svc
    return min(pool, key=lambda s: loads[s])


def wait_until_idle(svc):
    """Block until the instance has no running requests, bounded by IDLE_WAIT.

    Returns True if it went idle, False if the bound elapsed with work still
    running (the caller then proceeds and knowingly drops it).
    """
    if IDLE_WAIT <= 0:
        return running_requests(INSTANCES[svc]) <= 0
    deadline = time.time() + IDLE_WAIT
    while True:
        if running_requests(INSTANCES[svc]) <= 0:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(IDLE_POLL)


def clear_trigger(model):
    """Remove trigger.json only if it still names `model`. The router keeps
    rewriting it during a long swap and may have written a different model, so
    an unconditional remove would silently drop that request."""
    try:
        with open(TRIGGER) as f:
            cur = json.load(f).get("model")
    except Exception:
        cur = None
    if cur is None or cur == model:
        try:
            os.remove(TRIGGER)
        except FileNotFoundError:
            pass


def handle_trigger(reg):
    try:
        with open(TRIGGER) as f:
            model = json.load(f).get("model")
    except Exception:
        return
    if not model:
        clear_trigger(None)
        return
    if model not in reg:
        log("trigger for unknown model %r -- ignored (not in registry)" % model)
        clear_trigger(model)
        return
    st = read_state()
    for svc in INSTANCES:
        e = st.get(svc, {})
        if e.get("model") == model and e.get("ready"):
            # already serving it: nothing to do
            clear_trigger(model)
            return
    if time.time() - _last_fail.get(model, 0) < FAIL_COOLDOWN:
        # a recent swap to it failed; back off so a bad entry cannot loop restarts
        clear_trigger(model)
        return
    target = pick_target(st, read_usage())
    st[target] = entry(target, model, False)
    write_state(st)
    log("swap start: %s -> %s (the other instance is untouched)" % (target, model))
    n = running_requests(INSTANCES[target])
    if n > 0:
        log("draining %s: %d in-flight request(s); waiting up to %ss for idle"
            % (target, int(n), IDLE_WAIT))
    if wait_until_idle(target):
        if n > 0:
            log("drain complete: %s idle" % target)
    else:
        log("WARNING: %s still busy after %ss; proceeding (its in-flight "
            "requests will be dropped)" % (target, IDLE_WAIT))
    cid = find_container(target)
    if cid is None:
        log("ERROR: container for %s not found via docker socket; aborting swap" % target)
        _last_fail[model] = time.time()
        clear_trigger(model)
        return
    status, _ = sock_request("POST", "/containers/%s/restart" % cid)
    if status not in (200, 204, 304):
        log("ERROR: docker restart of %s failed (status %s)" % (target, status))
        _last_fail[model] = time.time()
        clear_trigger(model)
        return
    deadline = time.time() + SWAP_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL)
        if health_ok(INSTANCES[target]):
            st = read_state()
            st[target] = entry(target, model, True)
            write_state(st)
            _last_fail.pop(model, None)
            log("swap done: %s ready with %s" % (target, model))
            break
    else:
        _last_fail[model] = time.time()
        log("ERROR: %s still not healthy %ss after restart to %s; "
            "operator attention needed (fix the registry entry, then docker restart)"
            % (target, SWAP_TIMEOUT, model))
    clear_trigger(model)


def _actual_key(svc, reg):
    """Registry key an instance is actually serving right now (None if down)."""
    name = instance_model(INSTANCES[svc])
    return served_to_key(reg).get(name, name) if name else None


def _resolve_load_targets(model, instance, st, reg):
    """Which instances should receive `model` for a POST /load.

    `instance` may be an explicit service name, "all" (both cards), or
    "auto"/empty (default): one instance that is NOT already serving `model`,
    chosen idle-first with the same rotation/MRU preference as pick_target.
    """
    global _rr
    if instance and instance not in ("auto", "all"):
        return [instance] if instance in INSTANCES else []
    if instance == "all":
        return list(INSTANCES)
    pool = [s for s in INSTANCES if _actual_key(s, reg) != model]
    if not pool:
        return []
    loads = {s: running_requests(INSTANCES[s]) for s in pool}
    usage = read_usage()
    if usage:
        mru = max(usage, key=usage.get)
        keep = [s for s in pool if st.get(s, {}).get("model") != mru]
        if keep:
            pool = keep
    idle = [s for s in pool if loads[s] <= 0]
    if idle:
        svc = idle[_rr % len(idle)]
        _rr += 1
        return [svc]
    return [min(pool, key=lambda s: loads[s])]


def do_load(model=None, instance=None, force=False):
    """Load `model` onto instance(s) that are not already serving it.

    Unlike do_reload this CHANGES the served model: for each target it records
    state {model, ready:false} (so the router stops routing to it), drains
    in-flight work, restarts the container, polls /health, then flips
    ready:true. An instance already serving `model` is skipped unless
    force=true (which reloads it to pick up fresh registry settings).
    """
    reg = load_registry()
    if not model:
        return {"ok": False, "error": "model required", "registry": sorted(reg)}
    if model not in reg:
        return {"ok": False, "error": "model %r not in registry" % model,
                "registry": sorted(reg)}
    if instance and instance not in ("auto", "all") and instance not in INSTANCES:
        return {"ok": False, "error": "unknown instance %r" % instance,
                "instances": sorted(INSTANCES)}
    st = read_state()
    targets = _resolve_load_targets(model, instance, st, reg)
    if not targets:
        return {"ok": True, "model": model, "already_loaded": True,
                "targets": [], "message": "model already loaded on matching instance(s)"}
    results = []
    for svc in targets:
        if _actual_key(svc, reg) == model and health_ok(INSTANCES[svc]) and not force:
            results.append({"instance": svc, "ok": True, "already_loaded": True})
            continue
        st = read_state()
        st[svc] = entry(svc, model, False)
        write_state(st)
        log("load start: %s -> %s%s" % (svc, model, " (force)" if force else ""))
        if not wait_until_idle(svc):
            log("WARNING: %s still busy after %ss; proceeding" % (svc, IDLE_WAIT))
        cid = find_container(svc)
        if cid is None:
            log("ERROR: container for %s not found via docker socket" % svc)
            results.append({"instance": svc, "ok": False, "error": "container not found"})
            continue
        status, _ = sock_request("POST", "/containers/%s/restart" % cid)
        if status not in (200, 204, 304):
            log("ERROR: docker restart of %s failed (status %s)" % (svc, status))
            results.append({"instance": svc, "ok": False, "error": "restart status %s" % status})
            continue
        deadline = time.time() + SWAP_TIMEOUT
        ok = False
        while time.time() < deadline:
            time.sleep(POLL)
            if health_ok(INSTANCES[svc]):
                st = read_state()
                st[svc] = entry(svc, model, True)
                write_state(st)
                ok = True
                break
        log("load %s: %s" % (svc, "done" if ok else "TIMEOUT"))
        results.append({"instance": svc, "ok": ok})
    return {"ok": all(r.get("ok") for r in results), "model": model, "targets": results}


def do_reload(model=None, instance=None):
    """Restart matching instance(s) so their entrypoint re-reads the registry.

    Keeps the SAME model (state.json model unchanged) but picks up fresh settings
    (env/args) from model-registry.json -- the fast path for iterating on a model's
    knobs without a full Coolify redeploy. Drains in-flight work first (bounded by
    IDLE_WAIT), restarts via the docker socket, then polls /health.

    Targets: `instance` (a service name, or "all"); else `model`; else every
    currently-ready instance.
    """
    reg = load_registry()
    st = read_state()
    if instance and instance != "all":
        targets = [instance] if instance in INSTANCES else []
    elif model:
        targets = [s for s in INSTANCES if st.get(s, {}).get("model") == model]
    else:
        targets = [s for s in INSTANCES if st.get(s, {}).get("ready")] or list(INSTANCES)
    if not targets:
        return {"ok": False, "error": "no matching instance", "targets": []}

    results = []
    for svc in targets:
        tgt_model = st.get(svc, {}).get("model") or model or DEFAULT_MODEL
        if tgt_model not in reg:
            results.append({"instance": svc, "ok": False,
                            "error": "model %r not in registry" % tgt_model})
            continue
        st = read_state()
        st[svc] = entry(svc, tgt_model, False)
        write_state(st)
        log("reload start: %s (model %s) with fresh registry settings" % (svc, tgt_model))
        if not wait_until_idle(svc):
            log("WARNING: %s still busy after %ss; proceeding" % (svc, IDLE_WAIT))
        cid = find_container(svc)
        if cid is None:
            log("ERROR: container for %s not found via docker socket" % svc)
            results.append({"instance": svc, "ok": False, "error": "container not found"})
            continue
        status, _ = sock_request("POST", "/containers/%s/restart" % cid)
        if status not in (200, 204, 304):
            log("ERROR: docker restart of %s failed (status %s)" % (svc, status))
            results.append({"instance": svc, "ok": False, "error": "restart status %s" % status})
            continue
        deadline = time.time() + SWAP_TIMEOUT
        ok = False
        while time.time() < deadline:
            time.sleep(POLL)
            if health_ok(INSTANCES[svc]):
                st = read_state()
                st[svc] = entry(svc, tgt_model, True)
                write_state(st)
                ok = True
                break
        log("reload %s: %s" % (svc, "done" if ok else "TIMEOUT"))
        results.append({"instance": svc, "ok": ok})
    return {"ok": all(r.get("ok") for r in results), "targets": results}


class _Handler(BaseHTTPRequestHandler):
    server_version = "model-controller"

    def log_message(self, *a):
        return

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if not API_KEY:
            return True
        h = self.headers.get("Authorization", "")
        return h in ("Bearer " + API_KEY, API_KEY)

    def do_GET(self):
        if not self._auth():
            return self._send(401, {"ok": False, "error": "unauthorized"})
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True})
        if self.path.startswith("/status"):
            return self._send(200, {"ok": True, "state": read_state(),
                                    "registry": sorted(load_registry())})
        return self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self._auth():
            return self._send(401, {"ok": False, "error": "unauthorized"})
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path not in ("/reload", "/load"):
            return self._send(404, {"ok": False, "error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        params = {}
        if body:
            try:
                params = json.loads(body) or {}
            except Exception:
                params = {}
        if "?" in self.path:
            q = parse_qs(urlparse(self.path).query)
            for k in ("model", "instance", "force"):
                if params.get(k) in (None, "") and q.get(k):
                    params[k] = q[k][0]
        force = str(params.get("force", "")).lower() in ("1", "true", "yes", "on")
        with _op_lock:
            if path == "/load":
                res = do_load(model=params.get("model"),
                              instance=params.get("instance"), force=force)
            else:
                res = do_reload(model=params.get("model"), instance=params.get("instance"))
        return self._send(200 if res.get("ok") else 500, res)


def main():
    reg = load_registry()
    default = DEFAULT_MODEL if DEFAULT_MODEL in reg else sorted(reg)[0]
    st = read_state()
    if not st:
        st = {svc: entry(svc, default, False) for svc in INSTANCES}
        write_state(st)
        log("state initialised: both instances -> %s" % default)
    log("model-controller up (registry models: %s)" % ", ".join(sorted(reg)))
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", RELOAD_PORT), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log("admin endpoint on :%d  (POST /load, POST /reload, GET /status, GET /health)"
            % RELOAD_PORT)
    except Exception as e:
        log("WARNING: reload endpoint failed to start on :%d: %r" % (RELOAD_PORT, e))
    while True:
        try:
            with _op_lock:
                if reconcile(st, reg):
                    write_state(st)
                if os.path.exists(TRIGGER):
                    handle_trigger(reg)
                    st = read_state()
        except Exception as e:
            log("loop error: %r" % (e,))
        time.sleep(TICK)


if __name__ == "__main__":
    main()
