#!/usr/bin/env python3
"""Model router / OpenAI-compatible front door for the 2-GPU Radiance deployment.

This replaces the HAProxy data path. The stock `haproxy:2.8` image has no
request-time lua, no `json` fetch and no way to rewrite a JSON body, so it
cannot (a) parse the `model` field, (b) serve a request for a not-yet-loaded
model with a model that IS loaded, or (c) advertise registry models in
`GET /v1/models`. This router does all three.

What it does with each /v1 request
----------------------------------
1. Parse the OpenAI `model` field from the JSON body.
2. If that model is loaded and ready -> forward to a ready instance, choosing
   the least-in-flight one (this is the load balancing across the two GPUs).
3. If the model is registered but NOT loaded anywhere yet:
     * write /model-state/trigger.json so model-controller performs a
       ONE-AT-A-TIME swap (it restarts a single instance; the other keeps
       serving -> the previously-loaded model never has an error window), and
     * if another model IS ready, transparently rewrite `model` in the body to
       that model and serve the request now, so the user never sees an error
       while the selected model loads, or
     * if NOTHING else is ready (e.g. a future TP=2 model that occupies both
       ranks, so swapping it blanks both), HOLD the request open with SSE
       keep-alives until the model is ready, then stream the real answer.
4. If the model is neither loaded nor in the registry -> OpenAI-style 404.

`GET /v1/models` advertises every registry model, so a client can select a
model that is not loaded yet (selecting it is what triggers the swap).

State
-----
/model-state/state.json is written ONLY by model-controller:
    { "<instance>": {"model": "<name>", "ready": bool,
                     "endpoint": "http://host:port", "rank": 0, "group": "<id>"}, ... }
Instances that share an `endpoint` (tensor-parallel ranks) collapse into ONE
routing target whose readiness is the AND of its ranks. This is what makes the
router TP=2-ready: a TP=2 deployment exposes a single endpoint, so the router
needs no change -- only model-controller gains multi-rank swap logic later.

Stdlib only (no pip): ThreadingHTTPServer + http.client keep the image
(python:3.13-alpine) dependency-free.
"""
import hashlib
import http.client
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8100"))
STATE = os.environ.get("STATE_FILE", "/model-state/state.json")
TRIGGER = os.environ.get("TRIGGER_FILE", "/model-state/trigger.json")
REGISTRY = os.environ.get("REGISTRY_FILE", "/model-registry.json")
API_KEY = os.environ.get("VLLM_API_KEY", "")
HOLD_TIMEOUT = int(os.environ.get("HOLD_TIMEOUT", "3600"))
KEEPALIVE = int(os.environ.get("KEEPALIVE_SECONDS", "10"))
POLL = int(os.environ.get("POLL_SECONDS", "2"))
UPSTREAM_TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "3600"))
# Hard cap on a client request body (JSON chat payloads are tiny; this only
# stops an unauthenticated memory/CPU exhaustion attempt).
MAX_BODY = int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024 * 1024)))
# Max requests held open at once (each pins a thread up to HOLD_TIMEOUT).
MAX_HOLDS = int(os.environ.get("MAX_HELD_REQUESTS", "64"))
# Last-request time per model, written for model-controller's eviction protection.
USAGE = os.environ.get("USAGE_FILE", "/model-state/usage.json")
USAGE_MIN_INTERVAL = float(os.environ.get("USAGE_MIN_INTERVAL_SECONDS", "2"))
# Independent liveness: the router probes each endpoint's /health itself, so a
# state.json that has stopped advancing (controller wedged) cannot make it keep
# routing to a dead instance.
HEALTH_INTERVAL = int(os.environ.get("HEALTH_INTERVAL_SECONDS", "5"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT_SECONDS", "2"))

# --- Load-balancing tuning (see ROUTER-LB-PLAN / WORKLOG) -------------------
def _on(name, default="0"):
    return os.environ.get(name, default) not in ("0", "false", "False", "")

# W1: rotate equally-loaded targets round-robin. Off restores the old
# lexicographic tie-break (which sent all non-overlapping traffic to vllm-0).
LB_RR = _on("RADIANCE_LB_RR", "1")
# W2: use each instance's live vLLM load (`/metrics`: running/waiting/kv) as the
# primary routing signal instead of the router's open-connection count.
LB_METRICS = _on("RADIANCE_LB_METRICS", "1")
LB_METRICS_INTERVAL = float(os.environ.get("RADIANCE_LB_METRICS_INTERVAL", "1"))
LB_METRICS_STALE = float(os.environ.get("RADIANCE_LB_METRICS_STALE", "3"))
# W3: prefix-hash session affinity (stable-prefix rendezvous), gated on whether
# the shared fs KV tier already gives cross-instance reuse (see WORKLOG W4).
LB_AFFINITY = _on("RADIANCE_LB_AFFINITY", "0")
LB_AFFINITY_SLACK = int(os.environ.get("RADIANCE_LB_AFFINITY_SLACK", "1"))
# Verbose per-request routing decision.
LB_DEBUG = _on("RADIANCE_LB_DEBUG", "0")

# Endpoints are read ONLY from state.json (written by model-controller); the
# router keeps no second copy of the instance->endpoint map so the two services
# cannot drift.

# Headers that are per-hop and must not be forwarded verbatim.
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

_lock = threading.Lock()
_inflight = {}      # endpoint -> in-flight requests (for least-connections)
_rr = 0             # rotating cursor, breaks ties between equally-loaded targets
_alive = {}         # endpoint -> bool, independent /health view (default True)
_metrics = {}       # endpoint -> {"running","waiting","kv","ts"} live vLLM load
_metrics_lock = threading.Lock()
_dispatched = {}    # endpoint -> requests this router actually committed
_holds = threading.Semaphore(MAX_HOLDS)
_reg_cache = {"mtime": object(), "reg": {}}
_usage = {}          # model -> last-request epoch (MRU, for eviction protection)
_usage_written = 0.0


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_registry():
    """Registry models, cached; re-read only when the file's mtime changes."""
    try:
        mtime = os.stat(REGISTRY).st_mtime
    except OSError:
        mtime = None
    if mtime == _reg_cache["mtime"]:
        return _reg_cache["reg"]
    reg = read_json(REGISTRY, {})
    reg = {k: v for k, v in reg.items() if not k.startswith("_")}
    _reg_cache["mtime"], _reg_cache["reg"] = mtime, reg
    return reg


def read_state():
    return read_json(STATE, {})


def _endpoint(inst, e):
    return e.get("endpoint")


def targets_for_model(model, state):
    """Routing targets {endpoint, ready} for one model. Ranks that share an
    endpoint collapse to a single target; ready is the AND over those ranks."""
    groups = {}
    for inst, e in state.items():
        if not isinstance(e, dict) or e.get("model") != model:
            continue
        ep = _endpoint(inst, e)
        if not ep:
            continue
        g = groups.setdefault(ep, {"endpoint": ep, "ready": True})
        g["ready"] = g["ready"] and bool(e.get("ready"))
    return list(groups.values())


def all_targets(state):
    """Every routing target, annotated with the model its endpoint serves."""
    groups = {}
    for inst, e in state.items():
        if not isinstance(e, dict):
            continue
        model = e.get("model")
        ep = _endpoint(inst, e)
        if not model or not ep:
            continue
        g = groups.setdefault(ep, {"endpoint": ep, "model": model, "ready": True})
        g["ready"] = g["ready"] and bool(e.get("ready"))
    return list(groups.values())


def pick(targets):
    """Least-in-flight target that is both state-ready and independently alive;
    ties rotate round-robin so sequential bursts still spread across instances."""
    global _rr
    ready = [t for t in targets
             if t["ready"] and _alive.get(t["endpoint"], True)]
    if not ready:
        return None
    ready.sort(key=lambda t: t["endpoint"])
    with _lock:
        minload = min(_inflight.get(t["endpoint"], 0) for t in ready)
        cands = [t for t in ready
                 if _inflight.get(t["endpoint"], 0) == minload]
        t = cands[_rr % len(cands)]
        _rr += 1
    return t


def inflight_add(endpoint, delta):
    with _lock:
        _inflight[endpoint] = _inflight.get(endpoint, 0) + delta


def dispatched_add(endpoint):
    with _lock:
        _dispatched[endpoint] = _dispatched.get(endpoint, 0) + 1


def load_key(endpoint):
    """Ordering key for one endpoint; lower sorts first.

    W2: when live metrics are available (and fresh) the key is the engine's own
    load -- (running, waiting, kv%) -- so a long generation counts the same as
    any other in-flight request is no longer true. Otherwise it degrades to the
    router's open-connection count (`_inflight`), the original behaviour.
    """
    infl = _inflight.get(endpoint, 0)
    if LB_METRICS:
        m = _metrics.get(endpoint)
        if m and time.time() - m.get("ts", 0.0) <= LB_METRICS_STALE:
            return (m.get("running", 0.0), m.get("waiting", 0.0),
                    m.get("kv", 0.0), infl)
    return (float(infl), 0.0, 0.0, infl)


def order_targets(targets):
    """Ready, independently-alive targets, least-loaded first.

    Retry/failover order for `_forward`. With LB_RR the equally-loaded head
    group rotates round-robin (`_rr`) so sequential / non-overlapping traffic
    spreads across cards instead of always hitting vllm-0.
    """
    ready = [t for t in targets
             if t["ready"] and _alive.get(t["endpoint"], True)]
    if not ready:
        return []
    ready.sort(key=lambda t: (load_key(t["endpoint"]), t["endpoint"]))
    if not LB_RR:
        return ready
    global _rr
    with _lock:
        head = load_key(ready[0]["endpoint"])
        n = 0
        for t in ready:
            if load_key(t["endpoint"]) == head:
                n += 1
            else:
                break
        i = _rr % n
        _rr += 1
    group = ready[:n]
    return group[i:] + group[:i] + ready[n:]


def _affinity_key(body):
    """Stable conversation prefix hash (W3): system + first user turn.

    Deliberately NOT the whole prompt and NOT the shared system prompt alone:
    the prefix must be identical across a conversation's turns but distinct
    across sessions. Returns None when there is nothing to hash.
    """
    try:
        o = json.loads(body) if body else None
    except Exception:
        return None
    msgs = o.get("messages") if isinstance(o, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return None
    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        c = m.get("content")
        if isinstance(c, list):     # multimodal: join the text parts
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif not isinstance(c, str):
            c = json.dumps(c, sort_keys=True) if c is not None else ""
        parts.append(role + "\x1f" + c)
        if role == "user":
            break
    if not parts:
        return None
    return hashlib.sha1("\x1e".join(parts).encode("utf-8")).hexdigest()


def _affinity_target(key, candidates):
    """Rendezvous (highest-random-weight) endpoint for a prefix key.

    Consistent: adding/removing an endpoint remaps only its own share of keys.
    """
    return max(candidates,
               key=lambda t: int(hashlib.sha1(
                   (key + "|" + t["endpoint"]).encode("utf-8")).hexdigest(), 16))


def apply_affinity(cands, body):
    """Promote the prefix's endpoint when it is not materially hotter than the
    least-loaded one (bounded by LB_AFFINITY_SLACK), else keep least-loaded."""
    if not LB_AFFINITY or len(cands) < 2:
        return cands
    key = _affinity_key(body)
    if not key:
        return cands
    aff = _affinity_target(key, cands)
    if aff["endpoint"] == cands[0]["endpoint"]:
        return cands
    if load_key(aff["endpoint"])[0] <= load_key(cands[0]["endpoint"])[0] + LB_AFFINITY_SLACK:
        return [aff] + [t for t in cands if t["endpoint"] != aff["endpoint"]]
    return cands


def scrape_endpoint_metrics(endpoint):
    """GET one instance's /metrics and extract the routing load signal."""
    u = urllib.parse.urlsplit(endpoint)
    conn = None
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=HEALTH_TIMEOUT)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    want = {"vllm:num_requests_running": "running",
            "vllm:num_requests_waiting": "waiting",
            "vllm:kv_cache_usage_perc": "kv"}
    out = {"running": 0.0, "waiting": 0.0, "kv": 0.0}
    seen = set()
    for line in data.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in want and name not in seen:
            try:
                out[want[name]] = float(line.rsplit(" ", 1)[1])
                seen.add(name)
            except Exception:
                pass
    if not seen:
        return None
    out["ts"] = time.time()
    return out


def metrics_loop():
    """W2: keep a fresh per-endpoint live-load snapshot for `load_key`."""
    while True:
        try:
            endpoints = {t["endpoint"] for t in all_targets(read_state())}
            for ep in endpoints:
                snap = scrape_endpoint_metrics(ep)
                if snap is not None:
                    with _metrics_lock:
                        _metrics[ep] = snap
                else:
                    log("metrics scrape failed: %s" % ep)
        except Exception as e:
            log("metrics loop error: %r" % (e,))
        time.sleep(LB_METRICS_INTERVAL)


def record_usage(models):
    """Remember when each model was last requested/served (MRU). Written at most
    every USAGE_MIN_INTERVAL seconds; model-controller reads it to avoid evicting
    the model people are actively using."""
    global _usage_written
    now = time.time()
    for m in models:
        if m:
            _usage[m] = now
    if now - _usage_written < USAGE_MIN_INTERVAL:
        return
    _usage_written = now
    try:
        tmp = USAGE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_usage, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, USAGE)
    except Exception as e:
        log("usage write failed: %r" % (e,))


def health_probe(endpoint):
    """Independent /health probe (vLLM serves it unauthenticated)."""
    u = urllib.parse.urlsplit(endpoint)
    conn = None
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=HEALTH_TIMEOUT)
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def health_loop():
    """Refresh the independent liveness view for every endpoint in state.json."""
    while True:
        try:
            endpoints = {t["endpoint"] for t in all_targets(read_state())}
            for ep in endpoints:
                _alive[ep] = health_probe(ep)
        except Exception as e:
            log("health loop error: %r" % (e,))
        time.sleep(HEALTH_INTERVAL)


def write_trigger(model):
    """Ask model-controller to swap one instance onto `model` (atomic)."""
    try:
        tmp = TRIGGER + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"model": model}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, TRIGGER)
        log("swap trigger written: %r" % model)
    except Exception as e:
        log("trigger write failed: %r" % (e,))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "radiance-model-router/1"

    # ---- helpers ---------------------------------------------------------
    def log_message(self, fmt, *args):
        log("%s - %s" % (self.address_string(), fmt % args))

    def _send(self, status, body=b"", ctype="text/plain"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if body:
            self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj), "application/json")

    def _error(self, status, message, etype="InvalidRequestError", param="model"):
        self._json(status, {"error": {"message": message, "type": etype,
                                      "param": param, "code": status}})

    def _auth_fail(self):
        if not API_KEY:
            return False
        if self.headers.get("Authorization", "") == "Bearer " + API_KEY:
            return False
        self._error(401, "Invalid API key", "AuthenticationError")
        return True

    def _read_body(self):
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = []
            total = 0
            while True:
                line = self.rfile.readline()
                if not line:          # EOF: stop instead of spinning forever
                    break
                line = line.strip()
                if not line:          # tolerate a leading CRLF
                    continue
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline()
                    break
                total += size
                if total > MAX_BODY:  # bounded accumulation
                    break
                out.append(self.rfile.read(size))
                self.rfile.read(2)    # trailing CRLF
            return b"".join(out)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY:
            return b""
        return self.rfile.read(n) if n else b""

    @staticmethod
    def _model_of(body):
        try:
            o = json.loads(body)
        except Exception:
            return None, None
        if isinstance(o, dict):
            m = o.get("model")
            if isinstance(m, str) and m:
                return m, o
        return None, o

    # ---- upstream proxying ----------------------------------------------
    def _up_headers(self, netloc, body, rewrite_model):
        headers = {}
        for k, v in self.headers.items():
            if k.lower() in HOP:
                continue
            headers[k] = v
        headers["Host"] = netloc
        if rewrite_model is not None and body:
            try:
                o = json.loads(body)
                if isinstance(o, dict) and "model" in o:
                    o["model"] = rewrite_model
                    body = json.dumps(o).encode()
            except Exception:
                pass
        if body:
            headers["Content-Length"] = str(len(body))
        return headers, body

    def _forward(self, candidates, method, path, body, rewrite_model=None,
                 use_target_model=False, already_started=False):
        """Forward to the first candidate that BEGINS responding.

        Until the first response byte (or, for a fresh response, the status line)
        is committed to the client we retry the next candidate, so a request is
        not dropped just because the instance it first landed on died or is
        restarting. After that we cannot retry, so a later break is relayed as-is.
        """
        last_err = None
        for target in candidates:
            ep = target["endpoint"]
            u = urllib.parse.urlsplit(ep)
            rm = target.get("model") if use_target_model else rewrite_model
            headers, fbody = self._up_headers(u.netloc, body, rm)
            inflight_add(ep, 1)
            conn = None
            committed = False
            try:
                conn = http.client.HTTPConnection(u.hostname, u.port,
                                                  timeout=UPSTREAM_TIMEOUT)
                conn.request(method, path,
                             body=fbody if method != "GET" else None, headers=headers)
                resp = conn.getresponse()
                # Buffer the first body chunk: a failure before it is retryable.
                try:
                    first = resp.read(65536)
                except Exception as e:
                    last_err = e
                    try:
                        conn.close()
                    except Exception:
                        pass
                    log("upstream %s failed before first byte: %r; trying next"
                        % (ep, last_err))
                    continue
                if not already_started:
                    self._begin_response(resp.status, resp.getheaders())
                committed = True
                dispatched_add(ep)
                if first:
                    self.wfile.write(first)
                    self.wfile.flush()
                self._relay(resp)
                return
            except Exception as e:
                last_err = e
                if committed:
                    break   # bytes already sent; cannot move to another target
                log("upstream %s failed before first byte: %r; trying next"
                    % (ep, last_err))
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                inflight_add(ep, -1)
        if already_started:
            try:
                self.wfile.write(b"data: {\"error\":{\"message\":\"all upstreams "
                                 b"failed\"}}\n\n")
            except Exception:
                pass
        else:
            self._error(502, "no ready instance served the request: %s" % (last_err,),
                        "UpstreamError")

    def _begin_response(self, status, headers):
        self.send_response(status)
        for k, v in headers:
            if k.lower() in HOP:
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _relay(self, resp, size=65536):
        while True:
            try:
                chunk = resp.read(size)
            except Exception:
                break
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except Exception:
                break

    # ---- GET -------------------------------------------------------------
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/health":
            return self._send(200, "ok\n")
        if path == "/metrics":
            return self._metrics()
        if path.startswith("/v1/"):
            if self._auth_fail():
                return
            if path == "/v1/models":
                return self._models()
            state = read_state()
            cands = order_targets(all_targets(state))
            if not cands:
                return self._error(503, "no instance available")
            return self._forward(cands, "GET", self.path, None)
        return self._error(404, "not found", "NotFoundError")

    # ---- POST ------------------------------------------------------------
    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if not path.startswith("/v1/"):
            return self._error(404, "not found", "NotFoundError")
        if self._auth_fail():          # auth BEFORE reading the body
            return
        cl = self.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY:
            return self._error(413, "request body too large", param="")
        body = self._read_body()
        model, obj = self._model_of(body)
        stream = bool(isinstance(obj, dict) and obj.get("stream"))
        state = read_state()
        reg = load_registry()

        if not model:
            cands = order_targets(all_targets(state))
            if not cands:
                return self._error(503, "no instance available")
            return self._forward(cands, "POST", self.path, body)

        cands = order_targets(targets_for_model(model, state))
        if cands:
            record_usage([model])
            cands = apply_affinity(cands, body)
            if LB_DEBUG:
                log("route %r -> %s (load=%s%s)"
                    % (model, cands[0]["endpoint"], load_key(cands[0]["endpoint"]),
                       " affinity" if LB_AFFINITY and _affinity_key(body) else ""))
            return self._forward(cands, "POST", self.path, body)

        # Requested model is not ready anywhere.
        if model not in reg:
            return self._error(404, "The model `%s` does not exist." % model,
                               "NotFoundError")
        write_trigger(model)
        record_usage([model])
        fb = pick(all_targets(state))
        if fb:
            log("fallback: %r not ready -> serving with %r" % (model, fb["model"]))
            record_usage([fb["model"]])
            return self._forward(order_targets(all_targets(state)), "POST", self.path, body,
                                 use_target_model=True)
        log("hold: %r not ready and nothing else is; holding request" % model)
        return self._hold(model, body, stream)

    def _hold(self, model, body, stream):
        if not _holds.acquire(blocking=False):
            return self._error(503, "The model `%s` is loading and the server "
                               "is at its hold limit; try again shortly." % model)
        try:
            self._hold_wait(model, body, stream)
        finally:
            _holds.release()

    def _hold_wait(self, model, body, stream):
        deadline = time.time() + HOLD_TIMEOUT
        if stream:
            # Commit to an SSE response now and keep it warm while we wait.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while time.time() < deadline:
                t = pick(targets_for_model(model, read_state()))
                if t:
                    break
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except Exception:
                    return
                time.sleep(KEEPALIVE)
            t = pick(targets_for_model(model, read_state()))
            if not t:
                try:
                    self.wfile.write(b"data: {\"error\":{\"message\":\"model "
                                     b"load timed out\"}}\n\n")
                except Exception:
                    pass
                return
            return self._forward(order_targets(targets_for_model(model, read_state())),
                                 "POST", self.path, body, already_started=True)
        # Non-streaming: block until ready (bounded), then forward.
        while time.time() < deadline:
            if pick(targets_for_model(model, read_state())):
                break
            time.sleep(POLL)
        t = pick(targets_for_model(model, read_state()))
        if not t:
            return self._error(503, "The model `%s` is still loading; try again "
                               "shortly." % model)
        return self._forward(order_targets(targets_for_model(model, read_state())),
                             "POST", self.path, body)

    # ---- metadata endpoints ---------------------------------------------
    def _models(self):
        state = read_state()
        reg = load_registry()
        ids = []
        for m in list(reg) + [e.get("model") for e in state.values()
                              if isinstance(e, dict)]:
            if m and m not in ids:
                ids.append(m)
        data = [{"id": m, "object": "model", "created": 0, "owned_by": "radiance"}
                for m in sorted(ids)]
        self._json(200, {"object": "list", "data": data})

    def _metrics(self):
        state = read_state()
        lines = [
            "# HELP radiance_router_up 1 while the router is serving",
            "# TYPE radiance_router_up gauge",
            "radiance_router_up 1",
            "# HELP radiance_router_instance_ready 1 when an instance reports ready",
            "# TYPE radiance_router_instance_ready gauge",
        ]
        for inst, e in state.items():
            if not isinstance(e, dict):
                continue
            ready = 1 if e.get("ready") else 0
            lines.append('radiance_router_instance_ready{instance="%s",model="%s"} %d'
                         % (inst, e.get("model", ""), ready))
        lines.append("# HELP radiance_router_inflight in-flight requests per endpoint")
        lines.append("# TYPE radiance_router_inflight gauge")
        with _lock:
            items = sorted(_inflight.items())
        for ep, n in items:
            lines.append('radiance_router_inflight{endpoint="%s"} %d' % (ep, n))
        lines.append("# HELP radiance_router_endpoint_alive independent /health view")
        lines.append("# TYPE radiance_router_endpoint_alive gauge")
        with _lock:
            alive = sorted(_alive.items())
        for ep, ok in alive:
            lines.append('radiance_router_endpoint_alive{endpoint="%s"} %d'
                         % (ep, 1 if ok else 0))
        lines.append("# HELP radiance_router_load primary load signal per endpoint")
        lines.append("# TYPE radiance_router_load gauge")
        for ep in sorted({t["endpoint"] for t in all_targets(state)}):
            lines.append('radiance_router_load{endpoint="%s"} %g'
                         % (ep, load_key(ep)[0]))
        lines.append("# HELP radiance_router_dispatched_total requests committed per endpoint")
        lines.append("# TYPE radiance_router_dispatched_total counter")
        with _lock:
            dispatched = sorted(_dispatched.items())
        for ep, n in dispatched:
            lines.append('radiance_router_dispatched_total{endpoint="%s"} %d' % (ep, n))
        self._send(200, "\n".join(lines) + "\n", "text/plain; version=0.0.4")


def main():
    reg = load_registry()
    threading.Thread(target=health_loop, daemon=True).start()
    if LB_METRICS:
        threading.Thread(target=metrics_loop, daemon=True).start()
    log("model-router up on :%d (registry: %s)"
        % (PORT, ", ".join(sorted(reg)) or "<none>"))
    log("lb: rr=%d metrics=%d affinity=%d slack=%d debug=%d"
        % (int(LB_RR), int(LB_METRICS), int(LB_AFFINITY), LB_AFFINITY_SLACK, int(LB_DEBUG)))
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    # Default backlog is 5, which drops connections under a high-concurrency burst.
    srv.request_queue_size = int(os.environ.get("LISTEN_BACKLOG", "128"))
    srv.serve_forever()


if __name__ == "__main__":
    main()
