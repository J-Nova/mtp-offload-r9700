#!/usr/bin/env python3
"""Model hot-swap controller for the dual-card Radiance deployment.

Watches /model-state/trigger.json (written by model-router when a request
arrives for a registered model no engine is ready to serve) and swaps the ONE
dual-card container to it:

  1. record {model, ready:false} for every logical instance in state.json
     -> model-router stops routing to them
  2. drain: wait up to IDLE_WAIT_SECONDS for in-flight requests to finish
     (bounded, so a never-ending generation cannot block a swap forever)
  3. docker restart the `vllm` container (its entrypoint re-reads state.json +
     model-registry.json, then engine-supervisor.py brings up the topology the
     model's `topology` field asks for: two TP=1 engines for "dp", one TP=2
     engine for "tp2")
  4. poll /health on the topology's endpoints until ready, then flip ready:true
     -> model-router routes the new model to it

A swap restarts the whole container, so both logical targets go down together;
the router's HOLD path keeps requests open (SSE keep-alives) instead of erroring
while the new topology boots.

state.json also self-heals against /v1/models, so a crash or a manual
docker restart converges back to the real state. Only this process writes
state.json; the vllm container mounts it read-only and model-router only
writes trigger.json.

state.json schema (read by model-router):
    { "<logical>": {"model": "<name>", "ready": bool,
                    "endpoint": "http://host:port", "rank": <int>}, ... }
`endpoint`/`rank` let the router collapse tensor-parallel ranks (which share an
endpoint) into one routing target. In "dp" the two logical targets keep distinct
endpoints (8000/8001) for load balancing; in "tp2" both point at the primary
endpoint (8000) and collapse to one target.
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

# One physical container hosts the engines: the supervisor opens two ports in
# "dp" topology (one TP=1 engine per card -> 8000/8001) or one port in "tp2"
# (one TP=2 engine across both cards -> 8000). vllm-0/vllm-1 stay as LOGICAL
# routing targets so the router keeps its two-target load-balancing view in dp;
# in tp2 both collapse to the primary endpoint (the router dedupes them).
SERVICE = os.environ.get("VLLM_SERVICE", "vllm")
PRIMARY = os.environ.get("VLLM_PRIMARY_URL", "http://vllm:8000")
SECONDARY = os.environ.get("VLLM_SECONDARY_URL", "http://vllm:8001")
LOGICAL = ["vllm-0", "vllm-1"]


def topology_of(model, reg):
    """Serving topology of a model: 'dp' (default) or 'tp2'."""
    return (reg.get(model) or {}).get("topology", "dp")


def endpoints_for(topo):
    if topo == "tp2":
        return {s: PRIMARY for s in LOGICAL}
    return {"vllm-0": PRIMARY, "vllm-1": SECONDARY}


def ranks_for(topo):
    if topo == "tp2":
        return {s: 0 for s in LOGICAL}
    return {"vllm-0": 0, "vllm-1": 1}


def active_endpoints(topo):
    """Distinct endpoints the engine is listening on for this topology."""
    return [PRIMARY] if topo == "tp2" else [PRIMARY, SECONDARY]


_last_fail = {}   # model -> monotonic time of the last failed swap
_rr = 0           # rotating pick among equally-suitable (idle) instances
_op_lock = threading.Lock()  # serialize swaps/reloads: one restart op at a time


def entry(svc, model, ready, reg=None):
    reg = reg if reg is not None else load_registry()
    topo = topology_of(model, reg)
    return {"model": model, "ready": ready,
            "endpoint": endpoints_for(topo)[svc], "rank": ranks_for(topo)[svc]}


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
    """Container id by compose service label (Coolify renames containers, so
    never rely on the bare service name).

    Every logical instance (vllm-0/vllm-1) is backed by the ONE dual-card
    service, so the argument is accepted for API compatibility and ignored."""
    status, body = sock_request("GET", "/containers/json?all=1")
    if status != 200:
        return None
    for c in json.loads(body):
        if (c.get("Labels") or {}).get("com.docker.compose.service") == SERVICE:
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
    """Self-heal state against what the engine actually serves.

    One container backs every logical instance, so the model advertised on the
    primary endpoint decides the model AND topology for all of them; endpoints
    are then written to match the topology (tp2 collapses both to the primary
    endpoint, dp keeps 8000/8001)."""
    s2k = served_to_key(reg)
    changed = False
    actual = instance_model(PRIMARY)
    key = s2k.get(actual, actual) if actual else None
    topo = topology_of(key, reg) if key else "dp"
    eps = endpoints_for(topo)
    rks = ranks_for(topo)
    ready_now = health_ok(PRIMARY) and (topo == "tp2" or health_ok(SECONDARY))
    for svc in LOGICAL:
        e = st.get(svc)
        want_model = key or (e or {}).get("model") or DEFAULT_MODEL
        want_ready = bool(key) and ready_now
        if e is None:
            st[svc] = {"model": want_model, "ready": want_ready,
                       "endpoint": eps[svc], "rank": rks[svc]}
            changed = True
            continue
        if e.get("endpoint") != eps[svc] or e.get("rank") != rks[svc]:
            e["endpoint"] = eps[svc]
            e["rank"] = rks[svc]
            changed = True
        if key and e.get("model") != key:
            e["model"] = key
            e["ready"] = want_ready
            changed = True
        elif e.get("ready") != want_ready:
            e["ready"] = want_ready
            changed = True
    return changed


def wait_until_idle(svc=None):
    """Block until the active endpoints have no running requests, bounded by
    IDLE_WAIT. One container backs every logical instance, so a swap drains the
    whole deployment; `svc` is accepted for compatibility and ignored.

    Returns True if it went idle, False if the bound elapsed with work still
    running (the caller then proceeds and knowingly drops it)."""
    bases = [PRIMARY, SECONDARY]
    if IDLE_WAIT <= 0:
        return all(running_requests(b) <= 0 for b in bases)
    deadline = time.time() + IDLE_WAIT
    while True:
        if all(running_requests(b) <= 0 for b in bases):
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
    topo = topology_of(model, reg)
    eps = endpoints_for(topo)
    rks = ranks_for(topo)
    if all(st.get(svc, {}).get("model") == model and st.get(svc, {}).get("ready")
           for svc in LOGICAL):
        # already serving it: nothing to do
        clear_trigger(model)
        return
    if time.time() - _last_fail.get(model, 0) < FAIL_COOLDOWN:
        # a recent swap to it failed; back off so a bad entry cannot loop restarts
        clear_trigger(model)
        return
    for svc in LOGICAL:
        st[svc] = {"model": model, "ready": False,
                   "endpoint": eps[svc], "rank": rks[svc]}
    write_state(st)
    log("swap start: -> %s (topology=%s; the single container restarts)" % (model, topo))
    n = sum(running_requests(b) for b in active_endpoints(topo))
    if n > 0:
        log("draining: %d in-flight request(s); waiting up to %ss for idle"
            % (int(n), IDLE_WAIT))
    if wait_until_idle():
        if n > 0:
            log("drain complete: idle")
    else:
        log("WARNING: still busy after %ss; proceeding (in-flight requests will "
            "be dropped)" % IDLE_WAIT)
    cid = find_container(SERVICE)
    if cid is None:
        log("ERROR: container for service %s not found via docker socket; aborting swap"
            % SERVICE)
        _last_fail[model] = time.time()
        clear_trigger(model)
        return
    status, _ = sock_request("POST", "/containers/%s/restart" % cid)
    if status not in (200, 204, 304):
        log("ERROR: docker restart of %s failed (status %s)" % (SERVICE, status))
        _last_fail[model] = time.time()
        clear_trigger(model)
        return
    deadline = time.time() + SWAP_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL)
        if health_ok(PRIMARY) and (topo == "tp2" or health_ok(SECONDARY)):
            st = read_state()
            for svc in LOGICAL:
                st[svc] = {"model": model, "ready": True,
                           "endpoint": eps[svc], "rank": rks[svc]}
            write_state(st)
            _last_fail.pop(model, None)
            log("swap done: %s ready (topology=%s)" % (model, topo))
            break
    else:
        _last_fail[model] = time.time()
        log("ERROR: not healthy %ss after restart to %s; operator attention needed "
            "(fix the registry entry, then docker restart)" % (SWAP_TIMEOUT, model))
    clear_trigger(model)


def _actual_key(reg):
    """Registry key the engine is actually serving right now (None if down)."""
    name = instance_model(PRIMARY)
    return served_to_key(reg).get(name, name) if name else None


def _mark(st, model, ready, reg):
    topo = topology_of(model, reg)
    eps = endpoints_for(topo)
    rks = ranks_for(topo)
    for svc in LOGICAL:
        st[svc] = {"model": model, "ready": ready,
                   "endpoint": eps[svc], "rank": rks[svc]}


def _restart_and_wait(model, reg):
    """Restart the single container and wait for the target topology to be live."""
    topo = topology_of(model, reg)
    st = read_state()
    _mark(st, model, False, reg)
    write_state(st)
    if not wait_until_idle():
        log("WARNING: still busy after %ss; proceeding" % IDLE_WAIT)
    cid = find_container(SERVICE)
    if cid is None:
        return {"ok": False, "error": "container not found"}
    status, _ = sock_request("POST", "/containers/%s/restart" % cid)
    if status not in (200, 204, 304):
        return {"ok": False, "error": "restart status %s" % status}
    deadline = time.time() + SWAP_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL)
        if health_ok(PRIMARY) and (topo == "tp2" or health_ok(SECONDARY)):
            st = read_state()
            _mark(st, model, True, reg)
            write_state(st)
            return {"ok": True}
    return {"ok": False, "error": "timeout waiting for %s" % model}


def do_load(model=None, instance=None, force=False):
    """Load `model` onto the deployment (one dual-card container).

    `instance` is accepted for API compatibility and ignored: one container
    hosts every engine, so the whole deployment is always the target.
    """
    reg = load_registry()
    if not model:
        return {"ok": False, "error": "model required", "registry": sorted(reg)}
    if model not in reg:
        return {"ok": False, "error": "model %r not in registry" % model,
                "registry": sorted(reg)}
    if _actual_key(reg) == model and health_ok(PRIMARY) and not force:
        return {"ok": True, "model": model, "already_loaded": True,
                "targets": [], "message": "model already loaded"}
    log("load start: -> %s%s (topology=%s)"
        % (model, " (force)" if force else "", topology_of(model, reg)))
    res = _restart_and_wait(model, reg)
    log("load %s: %s" % (model, "done" if res.get("ok") else res.get("error")))
    return {"ok": res.get("ok", False), "model": model, "targets": [res]}


def do_reload(model=None, instance=None):
    """Restart the container so the entrypoint re-reads the registry with fresh
    settings (env/args), keeping the SAME model unless `model` names another.

    The fast path for iterating on a model's knobs without a full redeploy.
    """
    reg = load_registry()
    st = read_state()
    tgt_model = model or st.get("vllm-0", {}).get("model") or DEFAULT_MODEL
    if tgt_model not in reg:
        return {"ok": False, "error": "model %r not in registry" % tgt_model,
                "registry": sorted(reg)}
    log("reload start: model %s with fresh registry settings" % tgt_model)
    res = _restart_and_wait(tgt_model, reg)
    log("reload %s: %s" % (tgt_model, "done" if res.get("ok") else res.get("error")))
    return {"ok": res.get("ok", False), "targets": [res]}


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
        st = {}
        for svc in LOGICAL:
            st[svc] = entry(svc, default, False)
        write_state(st)
        log("state initialised: all logical instances -> %s" % default)
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
