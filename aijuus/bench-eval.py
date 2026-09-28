#!/usr/bin/env python3
"""Intelligence + time-to-correct-answer benchmark for local OpenAI-compatible endpoints.

Where `bench-async.py` measures *serving* (decode/prefill/concurrency percentiles via
BetterBench), this tool measures whether a model is *right and how long a correct answer
takes*. It runs the exact same questions against every --target, auto-grades each answer,
and reports the trade-off that matters for quantization / speculative-decoding A-Bs:

    accuracy   sec/question   sec/correct-answer   correct answers/hour

Targets use the same inline-JSON shape as bench-async.py, and --launch reuses the exact
same serve-mxfp4.sh launcher, health wait, card pick and teardown:

  # two already-running servers
  ./bench-eval.py --label mxfp4-ab \
    --target '{"name":"blend","base":"http://localhost:8000/v1","model":"Blend"}' \
    --target '{"name":"std","base":"http://localhost:8001/v1","model":"Standard"}'

  # or stand them up one at a time (one card, sequential), then evaluate
  MODELS=$HOME/models ./bench-eval.py --launch --quick \
    --target '{"name":"blend","port":8000,"model":"Blend","snap":"/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ","spec_method":"dflash"}' \
    --target '{"name":"std","port":8001,"model":"Standard","snap":"/models/Qwen3.8-27B-MXFP4-mtpfp8","spec_method":"dflash"}'

  # re-read a saved run
  ./bench-eval.py --compare bench/eval/blend.json bench/eval/std.json

Tasks (`--tasks`, default "gsm8k,aime25,mmlu_pro,ifeval,humaneval"):
  gsm8k     openai/gsm8k            grade-school math            numeric match
  aime24    math-ai/aime24          AIME 2024 (30)               numeric match
  aime25    math-ai/aime25          AIME 2025 (30)               numeric match
  aimo      AI-MO/aimo-validation-aime   AIME 2022-24 (90)       numeric match
  math500   HuggingFaceH4/MATH-500  competition math (500)       boxed/numeric match
  mmlu_pro  TIGER-Lab/MMLU-Pro      broad knowledge (A-J)        letter match
  ifeval    google/IFEval           instruction following        full verifiable checks
  humaneval openai/openai_humaneval coding, executable tests     unit tests
  gpqa      (gated)                 GPQA Diamond (A-D)           letter match

Task packs are downloaded once from the HF datasets-server into `--task-dir` (default
./bench/eval) and reused offline after that; --refresh re-downloads. GPQA is gated, so its
pack is not fetched: drop a `gpqa.jsonl` ({"id","prompt"/"question","answer","options"})
into the task dir to use it. Every pack is plain JSONL, so any task can be swapped for a
local one by writing a file with the same name.

Timing is wall-clock per request from send to last streamed token: TTFT plus generation.
`sec/correct-answer` averages latency over the answers that were actually right (the
metric that exposes "slightly smarter but much slower"), and `correct/hour` is throughput
of *correct* answers over the whole run (makespan), so it behaves under --concurrency too.

Sampling defaults: greedy (temperature 0) with reasoning_effort=low (brief thinking on), which
keeps the reasoning tasks (AIME/GPQA) discriminating without the full medium/xhigh token cost;
use --reasoning-effort medium/high for more thinking, or none/off to disable it. A target JSON may
override it per arm with "reasoning_effort". Raise --max-tokens (default 4096, env EVAL_MAX_TOKENS)
if answers truncate: the summary counts every finish_reason=length response as `trunc` and warns.
`--quick` caps every task at 10 questions for a smoke run.

Stdlib only. It shells out to `python3` only to execute HumanEval unit tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None

import bench_eval_ifeval

HERE = os.path.dirname(os.path.abspath(__file__))
DS_BASE = "https://datasets-server.huggingface.co"
DEFAULT_TASKS = "gsm8k,aime25,mmlu_pro,ifeval,humaneval"
DEFAULT_LIMIT = {"gsm8k": 100, "aime24": 30, "aime25": 30, "aimo": 90, "math500": 100,
                 "mmlu_pro": 100, "ifeval": 100, "humaneval": 50, "gpqa": 100}
# Known dataset sizes, used to cap --limit so an over-large request does not
# re-download a genuinely smaller task on every run.
TOTALS = {"aime24": 30, "aime25": 30, "aimo": 90, "humaneval": 164, "ifeval": 541,
          "gsm8k": 1319, "math500": 500, "mmlu_pro": 12032}

# ---------------------------------------------------------------------------
# Task registry + fetching.
# ---------------------------------------------------------------------------

MATH_INSTRUCTION = (
    "Solve the problem below. Think it through, then give ONLY the final answer on the "
    "last line in the form 'Answer: <value>'. If it is an integer, give the integer with "
    "no commas or units."
)
CHOICE_INSTRUCTION = (
    "Answer the multiple-choice question below. Reason if you like, then on the last line "
    "write ONLY the letter of the correct option in the form 'Answer: <letter>'."
)
CODE_INSTRUCTION = (
    "Complete the Python function below so it satisfies its docstring. Output the COMPLETE "
    "function definition (signature and body) as plain Python, with no prose and no "
    "markdown fences."
)

TASKS = {
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test",
              "grader": "numeric", "question": "question", "answer": "answer"},
    "aime24": {"dataset": "math-ai/aime24", "config": "default", "split": "test",
               "grader": "numeric", "question": "problem", "answer": "solution",
               "boxed": True},
    "aime25": {"dataset": "math-ai/aime25", "config": "default", "split": "test",
               "grader": "numeric", "question": "problem", "answer": "answer"},
    "aimo": {"dataset": "AI-MO/aimo-validation-aime", "config": "default", "split": "train",
             "grader": "numeric", "question": "problem", "answer": "answer"},
    "math500": {"dataset": "HuggingFaceH4/MATH-500", "config": "default", "split": "test",
                "grader": "math", "question": "problem", "answer": "answer"},
    "mmlu_pro": {"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test",
                 "grader": "choice", "question": "question", "answer": "answer",
                 "options": "options"},
    "ifeval": {"dataset": "google/IFEval", "config": "default", "split": "train",
               "grader": "ifeval", "question": "prompt"},
    "humaneval": {"dataset": "openai/openai_humaneval", "config": "openai_humaneval",
                  "split": "test", "grader": "code", "question": "prompt"},
    "gpqa": {"grader": "choice", "local": True},
}


def _http_json(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _fetch_rows(dataset: str, config: str, split: str, want: int) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while len(out) < want:
        length = min(100, want - len(out))
        query = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                        "split": split, "offset": offset, "length": length})
        payload = _http_json(f"{DS_BASE}/rows?{query}")
        if "error" in payload:
            raise RuntimeError(payload["error"])
        rows = payload.get("rows") or []
        if not rows:
            break
        out.extend(r["row"] for r in rows)
        offset += len(rows)
        if offset >= payload.get("num_rows_total", offset):
            break
    return out


def _boxed(text: str) -> str:
    """Last \\boxed{...} value, brace-balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    i = text.find("{", idx)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return text[i + 1:].strip()


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _normalize(raw: dict, task: str) -> dict | None:
    """Turn one dataset row into the pack record the runner and graders consume."""
    spec = TASKS[task]
    grader = spec["grader"]
    if grader == "ifeval":
        prompt = raw.get("prompt") or ""
        if not prompt:
            return None
        return {"id": str(raw.get("key", "")), "task": task, "grader": grader,
                "prompt": prompt, "answer": "", "options": None,
                "extra": {"instruction_id_list": raw.get("instruction_id_list") or [],
                          "kwargs": raw.get("kwargs") or []}}
    if grader == "code":
        code_prompt = raw.get("prompt") or ""
        if not code_prompt:
            return None
        return {"id": str(raw.get("task_id", "")), "task": task, "grader": grader,
                "prompt": CODE_INSTRUCTION + "\n\n" + code_prompt, "answer": "",
                "options": None,
                "extra": {"code_prompt": code_prompt, "test": raw.get("test") or "",
                          "entry_point": raw.get("entry_point") or ""}}
    question = str(raw.get(spec["question"], "")).strip()
    if not question:
        return None
    if task == "gsm8k":
        text = str(raw.get("answer", ""))
        gold = text.split("####")[-1].strip() if "####" in text else text.strip()
    elif spec.get("boxed"):
        gold = _boxed(str(raw.get(spec["answer"], "")))
    else:
        gold = str(raw.get(spec["answer"], "")).strip()
    options = None
    if "options" in spec:
        options = [str(o) for o in (raw.get(spec["options"]) or [])]
        labeled = "\n".join(f"{l}. {o}" for l, o in zip(_letters(len(options)), options))
        prompt = f"{CHOICE_INSTRUCTION}\n\n{question}\n\n{labeled}"
    elif grader in ("numeric", "math"):
        prompt = f"{MATH_INSTRUCTION}\n\n{question}"
    else:
        prompt = question
    return {"id": str(raw.get("id", raw.get("question_id", ""))), "task": task,
            "grader": grader, "prompt": prompt, "answer": gold, "options": options,
            "extra": {}}


def _load_local_pack(path: str, task: str) -> list[dict]:
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "prompt" in raw and "grader" in raw:
                recs.append(raw)
                continue
            question = str(raw.get("question", raw.get("prompt", ""))).strip()
            gold = str(raw.get("answer", "")).strip()
            options = raw.get("options")
            if options:
                labeled = "\n".join(f"{l}. {o}" for l, o in zip(_letters(len(options)),
                                                                 [str(o) for o in options]))
                prompt = f"{CHOICE_INSTRUCTION}\n\n{question}\n\n{labeled}"
            else:
                prompt = question
            recs.append({"id": str(raw.get("id", "")), "task": task,
                         "grader": TASKS[task]["grader"], "prompt": prompt,
                         "answer": gold, "options": options, "extra": {}})
    return recs


def load_task(task: str, args) -> list[dict]:
    """Fetch/cache one task pack and return its records (already clipped to limit)."""
    limit = args.limit if args.limit else DEFAULT_LIMIT.get(task, 100)
    if args.quick:
        limit = min(limit, 10)
    limit = min(limit, TOTALS.get(task, limit))
    os.makedirs(args.task_dir, exist_ok=True)
    path = os.path.join(args.task_dir, f"{task}.jsonl")
    spec = TASKS[task]
    if spec.get("local") and not os.path.exists(path):
        raise FileNotFoundError(f"{task}: no dataset source; expected a local pack at {path}")
    recs: list[dict] = []
    if os.path.exists(path) and not args.refresh:
        recs = _load_local_pack(path, task)
    if spec.get("local"):
        return recs[:limit]
    if len(recs) < limit:  # cache miss, or a quick run cached fewer than now wanted
        print(f"[tasks] fetching {task} ({limit}) from {spec['dataset']}/{spec['config']} ...",
              flush=True)
        rows = _fetch_rows(spec["dataset"], spec["config"], spec["split"], limit)
        recs = [r for r in (_normalize(raw, task) for raw in rows) if r]
        with open(path, "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        print(f"[tasks] cached {len(recs)} {task} questions -> {path}")
    return recs[:limit]


def resolve_tasks(spec: str) -> list[str]:
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown --tasks: {', '.join(unknown)} "
                         f"(known: {', '.join(TASKS)})")
    return tasks


# ---------------------------------------------------------------------------
# Graders.
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> str:
    boxed = _boxed(text)
    if boxed:
        return boxed
    tail = text
    for pat in (r"final\s+answer\s*(?:is|:)?", r"answer\s*(?:is|:)"):
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if matches:
            tail = text[matches[-1].end():]
            break
    num_re = r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?"
    nums = re.findall(num_re, tail) or re.findall(num_re, text)
    return nums[0].strip() if nums else ""


def _extract_answer_text(text: str) -> str:
    """Final-answer string (boxed, else the line after the last 'answer is')."""
    boxed = _boxed(text)
    if boxed:
        return boxed
    tail = text
    for pat in (r"final\s+answer\s*(?:is|:)?", r"answer\s*(?:is|:)"):
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if matches:
            tail = text[matches[-1].end():]
            break
    for line in tail.splitlines():
        cleaned = line.strip().strip(":\t *`")
        if cleaned:
            return cleaned
    return tail.strip()


def _norm_string(s: str) -> str:
    s = s.strip()
    for junk in ("$", "\\left", "\\right", "\\,", "\\!", "\\;", " ", "\u00a0"):
        s = s.replace(junk, "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = s.replace("{", "").replace("}", "").replace("^\\circ", "")
    s = s.replace("\\%", "%").replace("\\", "")
    s = s.strip(".").lower()
    return s


def _as_number(s: str) -> float | None:
    s = _norm_string(s).replace(",", "")
    frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", s)
    if frac:
        try:
            return float(frac.group(1)) / float(frac.group(2))
        except ZeroDivisionError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _grade_math(gold: str, pred: str) -> bool:
    if not pred:
        return False
    g, p = _norm_string(gold), _norm_string(pred)
    if g and g == p:
        return True
    gn, pn = _as_number(gold), _as_number(pred)
    if gn is not None and pn is not None:
        return math.isclose(gn, pn, rel_tol=1e-6, abs_tol=1e-6)
    return False


def _grade_numeric(gold: str, pred: str) -> bool:
    gn, pn = _as_number(gold), _as_number(pred)
    if gn is not None and pn is not None:
        return math.isclose(gn, pn, rel_tol=1e-6, abs_tol=1e-6)
    return _norm_string(gold) == _norm_string(pred) and bool(_norm_string(gold))


_LETTER_RE = re.compile(r"\b([A-J])\b")


def _grade_choice(gold: str, pred: str) -> bool:
    gold = gold.strip().upper()
    explicit = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\b",
                          pred, flags=re.IGNORECASE)
    letters = explicit or _LETTER_RE.findall(pred)
    if not letters:
        return False
    return letters[-1].upper() == gold


_FENCE_RE = re.compile(r"```(?:python|py)?\n?(.*?)```", re.DOTALL)
_DEF_RE_CACHE: dict[str, re.Pattern] = {}


def _def_re(entry: str) -> re.Pattern:
    if entry not in _DEF_RE_CACHE:
        _DEF_RE_CACHE[entry] = re.compile(rf"^def\s+{re.escape(entry)}\s*\(", re.MULTILINE)
    return _DEF_RE_CACHE[entry]


def _code_candidates(text: str, entry: str) -> list[str]:
    """Plausible code reconstructions of a HumanEval answer.

    Models answer in either style: the whole function, or a continuation that
    repeats the docstring tail before the body. Fenced blocks are tried too, and
    leading prose is dropped by starting at the first code-looking line.
    """
    blocks = _FENCE_RE.findall(text)
    cands: list[str] = []
    seen: set[str] = set()

    def add(block: str) -> None:
        block = block.replace("\r", "").strip("\n")
        if block and block not in seen:
            seen.add(block)
            cands.append(block)

    for block in (blocks + [text]):
        add(block)
        lines = block.replace("\r", "").split("\n")
        for i, ln in enumerate(lines):
            if re.match(r"^(def |import |from |@|#)", ln) or re.match(r"^[ \t]+\S", ln):
                if i:
                    add("\n".join(lines[i:]))  # drop a prose preamble
                break
    for block in list(cands):  # continuation that re-emitted the docstring
        idx = block.rfind('"""')
        if idx != -1:
            add(block[idx + 3:])
    return cands


def _code_programs(text: str, entry: str, raw: str) -> list[str]:
    programs: list[str] = []
    seen: set[str] = set()

    def add(program: str) -> None:
        if program and program not in seen:
            seen.add(program)
            programs.append(program)

    def_re = _def_re(entry)
    for cand in _code_candidates(text, entry):
        if def_re.search(cand):
            add(cand)                      # self-contained: signature + body
        add(raw.rstrip("\n") + "\n" + cand)  # body continues the prompt's signature
        first = next((ln for ln in cand.split("\n") if ln.strip()), "")
        if first and not first[:1].isspace():
            indented = "\n".join("    " + ln if ln.strip() else ln for ln in cand.split("\n"))
            add(raw.rstrip("\n") + "\n" + indented)  # flat body -> indent under the def
    return programs


def _grade_code(rec: dict, response: str) -> bool:
    entry = rec["extra"].get("entry_point", "")
    test = rec["extra"].get("test", "")
    raw = rec["extra"].get("code_prompt", "")
    if not entry or not test:
        return False
    programs = _code_programs(response, entry, raw)
    if not programs:
        return False
    with tempfile.TemporaryDirectory(prefix="bencheval-") as tmp:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        for index, program in enumerate(programs):
            path = os.path.join(tmp, f"candidate_{index}.py")
            with open(path, "w") as fh:
                fh.write(program + "\n\n" + test + f"\ncheck({entry})\n")
            try:
                proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                                      timeout=20, cwd=tmp, env=env)
            except (subprocess.SubprocessError, OSError):
                continue
            if proc.returncode == 0:
                return True
    return False


def grade(rec: dict, response: str, content: str) -> tuple[bool, int, int]:
    """Return (correct, instructions_checked, instructions_passed)."""
    grader = rec["grader"]
    if grader == "code":
        return _grade_code(rec, content or response), 1, 1
    if grader == "ifeval":
        checked, passed = bench_eval_ifeval.grade(rec, content or response)
        return checked > 0 and passed == checked, checked, passed
    pred = _extract_number(response) if grader in ("numeric", "math") else response
    if grader == "numeric":
        ok = _grade_numeric(rec["answer"], pred)
    elif grader == "math":
        ok = _grade_math(rec["answer"], _extract_answer_text(response))
    elif grader == "choice":
        ok = _grade_choice(rec["answer"], response)
    else:
        ok = False
    return ok, 1, 1


# ---------------------------------------------------------------------------
# Model requests.
# ---------------------------------------------------------------------------

class RequestError(Exception):
    pass


def thinking_kwargs(reasoning_effort: str) -> dict:
    """Map an effort level to the froggeric template's chat_template_kwargs.

    Template contract (qwen-fixed-v22.3.jinja:18-30): none/off disables thinking,
    minimal/low keeps thinking on but brief, high/xhigh/max expand it.
    """
    if reasoning_effort in ("none", "off"):
        return {"enable_thinking": False}
    return {"enable_thinking": True, "reasoning_effort": reasoning_effort}


def chat_stream(base: str, model: str, prompt: str, *, api_key: str, max_tokens: int,
                temperature: float, seed: int | None, reasoning_effort: str,
                timeout: float) -> dict:
    """One streaming chat completion. Returns text/content/timing/token fields."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    body["chat_template_kwargs"] = thinking_kwargs(reasoning_effort)
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    start = time.time()
    ttft = None
    pieces: list[str] = []
    content_pieces: list[str] = []
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.time()
                        pieces.append(piece)
                        if delta.get("content"):
                            content_pieces.append(piece)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RequestError(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RequestError(str(exc)) from exc
    end = time.time()
    gen = (usage or {}).get("completion_tokens", 0)
    decode_s = max(end - (ttft or start), 1e-9)
    return {"text": "".join(pieces), "content": "".join(content_pieces), "usage": usage or {},
            "latency_s": end - start, "ttft_ms": (ttft - start) * 1000 if ttft else None,
            "decode_tps": gen / decode_s if gen else None, "finish": finish}


# ---------------------------------------------------------------------------
# Launch/teardown: reuse bench-async.py wholesale.
# ---------------------------------------------------------------------------

class _LaunchArgs:
    serve_script = "./serve-mxfp4.sh"
    models = ""
    runtime = ""
    logs = True
    verbose_logs = False
    launch_timeout = 1800.0
    cleanup = True
    offload_gib = ""
    offload_head_cap = ""
    offload_policy = ""
    offload_read_threads = ""
    offload_write_threads = ""
    offload_disk_host = ""
    offload_fanout_max = ""
    offload_fanout_target_mb = ""


def _load_bench_async():
    path = os.path.join(HERE, "bench-async.py")
    if not os.path.exists(path):
        raise SystemExit("--launch needs bench-async.py beside this script")
    spec = importlib.util.spec_from_file_location("bench_async_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

def _answer_snippet(grader: str, out: dict) -> str:
    text = out["content"] or out["text"]
    return text[:8000] if grader == "code" else text[-200:]


def run_target(args, target: dict, tasks: list[str], packs: dict[str, list[dict]]) -> dict:
    effort = target.get("reasoning_effort") or args.reasoning_effort
    target["reasoning_effort"] = effort
    records = [(task, rec) for task in tasks for rec in packs[task]]
    print(f"\n[eval {target['name']}] {len(records)} questions across "
          f"{', '.join(tasks)} @ {target['base']} model={target['model']} "
          f"reasoning={effort} max_tokens={args.max_tokens} "
          f"concurrency={args.concurrency}", flush=True)
    results: list[dict] = []
    lock = threading.Lock()
    bar = _tqdm(total=len(records), desc=f"eval {target['name']}", leave=True) \
        if _tqdm is not None else None
    started = time.time()

    def work(item):
        task, rec = item
        seed = args.seed
        try:
            out = chat_stream(target["base"], target["model"], rec["prompt"],
                              api_key=args.api_key, max_tokens=args.max_tokens,
                              temperature=args.temperature, seed=seed,
                              reasoning_effort=effort, timeout=args.timeout)
            correct, checked, passed = grade(rec, out["text"], out["content"])
            return {"task": task, "id": rec["id"], "ok": bool(correct),
                    "checked": checked, "passed": passed, "error": None,
                    "latency_s": out["latency_s"], "ttft_ms": out["ttft_ms"],
                    "completion_tokens": out["usage"].get("completion_tokens", 0),
                    "prompt_tokens": out["usage"].get("prompt_tokens", 0),
                    "decode_tps": out["decode_tps"], "finish": out["finish"],
                    "answer": _answer_snippet(rec["grader"], out)}
        except RequestError as exc:
            return {"task": task, "id": rec["id"], "ok": False, "checked": 0, "passed": 0,
                    "error": str(exc), "latency_s": None, "ttft_ms": None,
                    "completion_tokens": 0, "prompt_tokens": 0, "decode_tps": None,
                    "finish": "error", "answer": ""}
        finally:
            with lock:
                if bar is not None:
                    bar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for res in pool.map(work, records):
            results.append(res)
    makespan = time.time() - started
    if bar is not None:
        bar.close()

    summary = summarize(target, tasks, results, makespan)
    detail = {"target": {k: target.get(k) for k in
                         ("name", "base", "model", "snap", "drafter", "spec_method",
                          "spec", "maxlen", "gpu", "port", "container", "reasoning_effort")},
              "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "makespan_s": makespan,
              "config": {"tasks": tasks, "concurrency": args.concurrency,
                         "max_tokens": args.max_tokens, "temperature": args.temperature,
                         "seed": args.seed, "reasoning_effort": effort},
              "summary": summary, "results": results}
    report_target(summary)
    return detail


def summarize(target: dict, tasks: list[str], results: list[dict], makespan: float) -> dict:
    def quant(values, q):
        vals = sorted(v for v in values if v is not None)
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
        return vals[idx]

    per_task = {}
    for task in tasks:
        rows = [r for r in results if r["task"] == task]
        graded = [r for r in rows if r["checked"] > 0]
        lat = [r["latency_s"] for r in graded if r["latency_s"] is not None]
        correct_rows = [r for r in graded if r["ok"]]
        instr_checked = sum(r["checked"] for r in graded) if task == "ifeval" else 0
        instr_passed = sum(r["passed"] for r in graded) if task == "ifeval" else 0
        tps = [r["decode_tps"] for r in graded if r["decode_tps"]]
        truncated = [r for r in graded if r.get("finish") == "length"]
        per_task[task] = {
            "n": len(rows), "n_graded": len(graded), "n_errors": len(rows) - len(graded),
            "correct": len(correct_rows),
            "n_truncated": len(truncated),
            "accuracy": len(correct_rows) / len(graded) if graded else None,
            "instr_checked": instr_checked, "instr_passed": instr_passed,
            "instruction_accuracy": (instr_passed / instr_checked) if instr_checked else None,
            "mean_latency_s": sum(lat) / len(lat) if lat else None,
            "p50_latency_s": quant(lat, 0.5),
            "sec_per_correct": (sum(r["latency_s"] for r in correct_rows) / len(correct_rows)
                                if correct_rows else None),
            "mean_ttft_ms": (sum(r["ttft_ms"] for r in graded if r["ttft_ms"] is not None)
                             / max(1, sum(1 for r in graded if r["ttft_ms"] is not None))),
            "mean_completion_tokens": (sum(r["completion_tokens"] for r in graded)
                                       / len(graded) if graded else None),
            "mean_decode_tps": sum(tps) / len(tps) if tps else None,
        }
    graded_all = [r for r in results if r["checked"] > 0]
    correct_all = [r for r in graded_all if r["ok"]]
    lats = [r["latency_s"] for r in graded_all if r["latency_s"] is not None]
    overall = {
        "n": len(results), "n_graded": len(graded_all), "n_errors": len(results) - len(graded_all),
        "correct": len(correct_all),
        "n_truncated": sum(1 for r in graded_all if r.get("finish") == "length"),
        "accuracy": len(correct_all) / len(graded_all) if graded_all else None,
        "mean_latency_s": sum(lats) / len(lats) if lats else None,
        "median_latency_s": quant(lats, 0.5),
        "sec_per_correct": (sum(r["latency_s"] for r in correct_all) / len(correct_all)
                            if correct_all else None),
        "correct_per_hour": (len(correct_all) / makespan * 3600.0) if makespan > 0 else None,
        "questions_per_hour": (len(graded_all) / makespan * 3600.0) if makespan > 0 else None,
        "mean_decode_tps": (sum(r["decode_tps"] for r in graded_all if r["decode_tps"])
                            / max(1, sum(1 for r in graded_all if r["decode_tps"]))),
        "total_completion_tokens": sum(r["completion_tokens"] for r in graded_all),
    }
    return {"name": target["name"], "base": target["base"], "model": target["model"],
            "makespan_s": makespan, "overall": overall, "tasks": per_task}


def _pct(v):
    return "  n/a " if v is None else f"{100 * v:5.1f}%"


def _task_acc(task: str, t: dict):
    """Prompt-level accuracy, except IFEval which is graded per instruction."""
    if t.get("instruction_accuracy") is not None:
        return t["instruction_accuracy"]
    return t.get("accuracy")


def _num(v, unit="", fmt="6.1f"):
    if v is None:
        return "   n/a"
    return f"{v:{fmt}}{unit}"


def report_target(summary: dict) -> None:
    o = summary["overall"]
    print(f"--- {summary['name']} ({summary['model']}) ---")
    print(f"  accuracy {_pct(o['accuracy'])} | {_num(o['mean_latency_s'],'s')} sec/question"
          f" | {_num(o['sec_per_correct'],'s')} sec/correct"
          f" | {_num(o['correct_per_hour'],'',fmt='7.0f')} correct/hour"
          f" | {_num(o['mean_decode_tps'],' tok/s')}")
    if o.get("n_truncated"):
        print(f"  WARNING {o['n_truncated']}/{o['n_graded']} responses hit the max_tokens cap "
              f"(finish_reason=length) -- raise --max-tokens to avoid cut-offs")
    header = f"  {'task':>10} {'acc':>7} {'n':>5} {'trunc':>6} {'mean s':>8} {'p50 s':>7} {'s/correct':>10} {'tok/s':>7}"
    print(header)
    for task, t in summary["tasks"].items():
        acc = _pct(_task_acc(task, t))
        print(f"  {task:>10} {acc:>7} {t['n_graded']:>5} {t.get('n_truncated', 0):>6} "
              f"{_num(t['mean_latency_s'],'',fmt='8.2f')} {_num(t['p50_latency_s'],'',fmt='7.2f')} "
              f"{_num(t['sec_per_correct'],'',fmt='10.2f')} "
              f"{_num(t['mean_decode_tps'],'',fmt='7.1f')}")


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    eff_a = (a.get("config") or {}).get("reasoning_effort", "?")
    eff_b = (b.get("config") or {}).get("reasoning_effort", "?")
    if eff_a != eff_b:
        print(f"[compare] WARNING reasoning_effort differs: "
              f"{a['target']['name']}={eff_a} vs {b['target']['name']}={eff_b}", file=sys.stderr)
    rows = [("name", a["target"]["name"], b["target"]["name"], ""),
            ("accuracy", _pct(a["summary"]["overall"]["accuracy"]),
             _pct(b["summary"]["overall"]["accuracy"]), ""),
            ("sec/question", _num(a["summary"]["overall"]["mean_latency_s"], "", "7.2f"),
             _num(b["summary"]["overall"]["mean_latency_s"], "", "7.2f"), "s"),
            ("sec/correct", _num(a["summary"]["overall"]["sec_per_correct"], "", "7.2f"),
             _num(b["summary"]["overall"]["sec_per_correct"], "", "7.2f"), "s"),
            ("correct/hour", _num(a["summary"]["overall"]["correct_per_hour"], "", "8.0f"),
             _num(b["summary"]["overall"]["correct_per_hour"], "", "8.0f"), ""),
            ("questions/hour", _num(a["summary"]["overall"]["questions_per_hour"], "", "8.0f"),
             _num(b["summary"]["overall"]["questions_per_hour"], "", "8.0f"), ""),
            ("decode tok/s", _num(a["summary"]["overall"]["mean_decode_tps"], "", "7.1f"),
             _num(b["summary"]["overall"]["mean_decode_tps"], "", "7.1f"), ""),
            ("reasoning", eff_a, eff_b, "")]
    print(f"{'metric':>15} | {a['target']['name']:>16} | {b['target']['name']:>16}")
    print("-" * 15 + "-+-" + "-" * 16 + "-+-" + "-" * 16)
    for label, va, vb, unit in rows:
        print(f"{label:>15} | {str(va) + unit:>16} | {str(vb) + unit:>16}")
    tasks = sorted(set(a["summary"]["tasks"]) | set(b["summary"]["tasks"]))
    print(f"\n{'per-task accuracy':>15} | {a['target']['name']:>16} | {b['target']['name']:>16}")
    for task in tasks:
        ta = a["summary"]["tasks"].get(task, {})
        tb = b["summary"]["tasks"].get(task, {})
        print(f"{task:>15} | {_pct(_task_acc(task, ta)):>16} | {_pct(_task_acc(task, tb)):>16}")
    return 0


def discover_model(base: str, api_key: str) -> str:
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data["data"][0]["id"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"could not discover a model at {base}: {exc}")


def _container_state(runtime: str, name: str) -> str:
    try:
        out = subprocess.run([runtime, "inspect", "-f", "{{.State.Status}}", name],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def wait_healthy(ba, runtime: str, launch_args, target: dict, name: str) -> None:
    """Like ba.wait_healthy, but stop waiting the moment the container dies.

    A vLLM that refuses to start (e.g. KV cache smaller than --maxlen) exits within
    seconds; without this we would poll /health for the full launch timeout.
    """
    url = f"http://localhost:{target['port']}/health"
    deadline = time.time() + launch_args.launch_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    print(f"[launch {target['name']}] healthy at {target['base']}")
                    return
        except Exception:  # noqa: BLE001
            pass
        state = _container_state(runtime, name)
        if state in ("exited", "dead", "removing"):
            raise SystemExit(
                f"[launch {target['name']}] container {name} is {state} before becoming "
                f"healthy -- see its log above. A common cause is KV cache too small for "
                f"--maxlen: lower --maxlen or free memory with --chunk / --maxseqs")
        time.sleep(5.0)
    raise SystemExit(f"[launch {target['name']}] not healthy after "
                     f"{launch_args.launch_timeout:.0f}s")


EFFORT_CHOICES = ("none", "off", "minimal", "low", "medium", "high", "xhigh")


def _valid_effort(effort: str) -> str:
    effort = str(effort).lower()
    if effort not in EFFORT_CHOICES:
        raise SystemExit(f"invalid reasoning_effort {effort!r} "
                         f"(choose from: {', '.join(EFFORT_CHOICES)})")
    return effort


def _carry_effort(parsed: dict, spec: str) -> None:
    """A target JSON may set its own reasoning_effort, overriding --reasoning-effort."""
    obj = json.loads(spec)
    if obj.get("reasoning_effort") is not None:
        parsed["reasoning_effort"] = _valid_effort(obj["reasoning_effort"])


def build_and_run(args) -> int:
    tasks = resolve_tasks(args.tasks)
    ba = _load_bench_async() if args.launch else None

    if args.target:
        if ba is not None:
            targets = [ba.parse_target(t, args.model, launch=True) for t in args.target]
            for parsed, spec in zip(targets, args.target):
                _carry_effort(parsed, spec)
        else:
            targets = []
            for spec in args.target:
                obj = json.loads(spec)
                base = obj.get("base") or (f"http://localhost:{obj['port']}/v1"
                                           if obj.get("port") else "")
                if not base:
                    raise SystemExit(f"--target needs a base URL: {spec}")
                targets.append({"name": obj.get("name", base), "base": base.rstrip("/"),
                                "model": obj.get("model", args.model)})
                _carry_effort(targets[-1], spec)
    else:
        targets = [{"name": "default", "base": args.base.rstrip("/"), "model": args.model}]

    # --maxlen and the memory knobs apply to every launch target that did not set
    # them in its JSON (a target value always wins).
    for target in targets:
        for arg, field in (("maxlen", "maxlen"), ("chunk", "chunk"),
                           ("maxseqs", "maxseqs"), ("gpu_util", "gpu_util"),
                           ("kv_mem", "kv_mem")):
            if getattr(args, arg) and not target.get(field):
                target[field] = getattr(args, arg)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    runtime = ba.detect_runtime(args.runtime) if ba is not None else "docker"
    if ba is not None:
        warn_path = args.warnings_log or (
            os.path.join(args.save_dir, "warnings.log") if args.save_dir
            else os.path.join(os.getcwd(), "warnings.log"))
        ba.warn_log_init(warn_path)
        ba._arm_cleanup(runtime)

    packs: dict[str, list[dict]] = {}
    for task in tasks:
        try:
            packs[task] = load_task(task, args)
        except Exception as exc:  # noqa: BLE001 - one bad task must not abort the run
            print(f"[tasks] {task}: {exc}", file=sys.stderr)
    tasks = [t for t in tasks if packs.get(t)]
    if not tasks:
        raise SystemExit("no tasks could be loaded")

    launch_args = None
    if ba is not None:
        launch_args = _LaunchArgs()
        launch_args.serve_script = args.serve_script
        launch_args.models = args.models
        launch_args.runtime = args.runtime
        launch_args.logs = args.logs
        launch_args.verbose_logs = args.verbose_logs
        launch_args.launch_timeout = args.launch_timeout
        launch_args.cleanup = args.cleanup

    summaries: list[dict] = []
    details: list[dict] = []
    paths: list[str | None] = []
    failures = 0

    def evaluate(target):
        if not target.get("model"):
            target["model"] = discover_model(target["base"], args.api_key)
        detail = run_target(args, target, tasks, packs)
        out_dir = args.save_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{target['name']}.eval.json")
        with open(path, "w") as fh:
            json.dump(detail, fh, indent=1)
        print(f"[eval {target['name']}] wrote {path}")
        overall = detail["summary"]["overall"]
        if overall["n"] and overall["n_graded"] == 0:
            raise RuntimeError(f"all {overall['n']} requests failed "
                               f"(first error: {detail['results'][0]['error']})")
        return detail, path

    try:
        if ba is None:
            for target in targets:
                try:
                    detail, path = evaluate(target)
                    details.append(detail)
                    summaries.append(detail["summary"])
                    paths.append(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval {target['name']}] FAILED: {exc}", file=sys.stderr)
                    paths.append(None)
                    failures += 1
        else:
            best, ranking = ba.pick_best_card(args.card or os.environ.get("BENCH_CARD") or None)
            if best is None:
                raise SystemExit("could not detect GPUs; pass --card <hip_index>")
            ba.report_card(best, ranking)
            print(f"[launch] card=GPU{best} kv_offload=off "
                  f"maxlen={args.maxlen or '(per-target/default)'} "
                  f"chunk={args.chunk or '-'} maxseqs={args.maxseqs or '-'} "
                  f"gpu_util={args.gpu_util or '-'}")
            for index, target in enumerate(targets):
                target["gpu"] = best
                target["port"] = str(target.get("port") or 8000 + index)
                if target.get("spec_method") == "dflash" and not target.get("drafter"):
                    print(f"[launch] WARNING {target['name']}: spec_method=dflash with no "
                          f"\"drafter\" -- serve-mxfp4.sh will use the stock "
                          f"$MODELS/Qwen3.8-27B-DFlash2-FP8. Set \"drafter\" if this model has "
                          f"its own fine-tuned draft head.", file=sys.stderr)
            ba.cleanup_stale(runtime, enabled=args.cleanup)
            if not ba.preflight_ports(runtime, targets):
                return 2
            width = max(len(t["name"]) for t in targets)
            for index, target in enumerate(targets):
                name = f"bench-{target['name']}"
                ba._track_container(name)
                try:
                    ba.launch_instance(launch_args, target, index)
                except subprocess.CalledProcessError as exc:
                    print(f"[launch {target['name']}] {args.serve_script} failed "
                          f"(exit {exc.returncode})", file=sys.stderr)
                    ba._do_cleanup(reason=" failed-launch:")
                    return exc.returncode or 1
                tailer = None
                if args.logs:
                    tailer = ba.ContainerLogTailer(runtime, name, target["name"],
                                                   width=width, verbose=args.verbose_logs)
                    tailer.start()
                    ba._track_tailer(tailer)
                keep_this = args.keep and index == len(targets) - 1
                try:
                    wait_healthy(ba, runtime, launch_args, target, name)
                    detail, path = evaluate(target)
                    details.append(detail)
                    summaries.append(detail["summary"])
                    paths.append(path)
                except SystemExit as exc:
                    ba.warn(str(exc), target=target["name"])
                    failures += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval {target['name']}] FAILED: {exc}", file=sys.stderr)
                    failures += 1
                finally:
                    if tailer is not None:
                        tailer.stop()
                        ba._untrack_tailer(tailer)
                    if keep_this:
                        ba._untrack_container(name)
                    else:
                        ba._stop_container(runtime, name)
                        ba._untrack_container(name)

        if details:
            out_dir = args.save_dir or "."
            overview_path = args.overview or os.path.join(out_dir, "overview.json")
            overview = {"label": args.label, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "tasks": tasks, "concurrency": args.concurrency,
                        "reasoning_effort": args.reasoning_effort, "max_tokens": args.max_tokens,
                        "reasoning_effort_by_target":
                            {d["summary"]["name"]: d["config"]["reasoning_effort"]
                             for d in details},
                        "targets": [d for d in details]}
            with open(overview_path, "w") as fh:
                json.dump(overview, fh, indent=1)
            print(f"\n[bench] wrote {overview_path}")
            if len(paths) == 2 and all(paths):
                print("\n=== target comparison ===")
                compare(paths[0], paths[1])
            elif len(summaries) > 1:
                print("\n=== all targets ===")
                print(f"  {'name':>12} {'reason':>7} {'accuracy':>9} {'sec/q':>8} "
                      f"{'sec/correct':>12} {'correct/hour':>13}")
                for s, d in zip(summaries, details):
                    o = s["overall"]
                    print(f"  {s['name']:>12} {d['config']['reasoning_effort']:>7} "
                          f"{_pct(o['accuracy']):>9} "
                          f"{_num(o['mean_latency_s'],'',fmt='8.2f')} "
                          f"{_num(o['sec_per_correct'],'',fmt='12.2f')} "
                          f"{_num(o['correct_per_hour'],'',fmt='13.0f')}")
        if failures:
            print(f"\n{failures} target(s) failed; see the messages above", file=sys.stderr)
    finally:
        if ba is not None:
            ba._do_cleanup()
            ba.warn_log_close()
    return 1 if failures else 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="print a comparison of two saved *.eval.json files and exit")
    p.add_argument("--target", action="append", default=[],
                   help="JSON target spec; repeat to evaluate several endpoints/models")
    p.add_argument("--launch", action="store_true",
                   help="start one vLLM per target via --serve-script, then tear down")
    p.add_argument("--serve-script", default="./serve-mxfp4.sh")
    p.add_argument("--models", default="", help="MODELS dir passed to --serve-script")
    p.add_argument("--maxlen", default=os.environ.get("EVAL_MAXLEN", ""),
                   help="--max-model-len for every launched target (target JSON \"maxlen\" wins); "
                        "e.g. 150000. No effect without --launch")
    p.add_argument("--chunk", default="", help="MAXLEN-chunk / --max-num-batched-tokens "
                   "passthrough for launched targets (frees KV memory when lowered)")
    p.add_argument("--maxseqs", default="", help="MAXSEQS / --max-num-seqs passthrough")
    p.add_argument("--gpu-util", default="", help="GPU_UTIL passthrough (default 0.98)")
    p.add_argument("--kv-mem", default="", help="KV_MEM pin passthrough: auto | <bytes> | 0")
    p.add_argument("--runtime", default="", help="container runtime (auto: podman, docker)")
    p.add_argument("--card", default="", help="HIP index to force as the benchmark card")
    p.add_argument("--keep", action="store_true", help="leave the last launched instance up")
    p.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True,
                   help="stop pre-existing bench-* containers before launching")
    p.add_argument("--logs", action=argparse.BooleanOptionalAction, default=True,
                   help="stream each launched container's logs")
    p.add_argument("--verbose-logs", action="store_true")
    p.add_argument("--launch-timeout", type=float, default=1800.0)

    p.add_argument("--tasks", default=os.environ.get("EVAL_TASKS", DEFAULT_TASKS),
                   help=f"comma list (default: {DEFAULT_TASKS}); known: {', '.join(TASKS)}")
    p.add_argument("--task-dir", default=os.environ.get("EVAL_TASK_DIR", "./bench/eval"),
                   help="task-pack cache directory (JSONL per task)")
    p.add_argument("--limit", type=int, default=0,
                   help="max questions per task (default: a per-task cap; 0 = use the cap)")
    p.add_argument("--quick", action="store_true", help="cap every task at 10 questions")
    p.add_argument("--refresh", action="store_true", help="re-download task packs")
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel requests (default 1: faithful single-stream latency)")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("EVAL_MAX_TOKENS", "4096")),
                   help="per-response token cap (default 4096; raise to ~8192-16384 for "
                        "medium/high effort or long-code answers). Responses that hit it are "
                        "counted as 'trunc' and flagged with finish_reason=length")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reasoning-effort", default=os.environ.get("EVAL_REASONING_EFFORT", "low"),
                   choices=["none", "off", "minimal", "low", "medium", "high", "xhigh"],
                   help="chat_template reasoning_effort (default low: thinking on but brief; "
                        "none/off disables thinking; high expands it)")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="alias: --no-thinking == --reasoning-effort off; --thinking keeps the "
                        "selected effort (useful to override EVAL_REASONING_EFFORT)")
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request seconds")
    p.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "juup-123"))
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", ""))
    p.add_argument("--label", default="")
    p.add_argument("--save-dir", default="")
    p.add_argument("--overview", default="")
    p.add_argument("--warnings-log", default="")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.thinking is False:
        args.reasoning_effort = "off"
    elif args.thinking is True and args.reasoning_effort in ("none", "off"):
        args.reasoning_effort = "medium"
    if args.compare:
        return compare(*args.compare)
    return build_and_run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
