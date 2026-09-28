"""
Row-shape adapters for calibration sources (see calibration.py).

Each adapter turns one raw dataset row into chat messages. Kept in its own
module because the set grows with every corpus you want to exercise, and
calibration.py's job is loading/tokenizing, not knowing what MMLU looks like.

A "shape" is detected from a row's keys, or forced as the 4th --source field.
Adding a dataset = one function + one SHAPES entry.

Sources that are not row-oriented (a raw .txt to chunk, a directory of images)
are handled as separate source KINDS in calibration.py, not here.
"""

import csv
import json
import random
from pathlib import Path

LETTERS = "ABCDEFGHIJ"


def _mcq(question, choices, answer_idx, preamble=""):
    """Render a multiple-choice item as an answered turn.

    The assistant turn is included on purpose: routing during an answer differs
    from routing while reading a question, and we want both in the map.
    """
    body = (preamble + question).strip() + "\n" + "\n".join(
        f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )
    return [
        {"role": "user", "content": body},
        {"role": "assistant", "content": f"The answer is {LETTERS[answer_idx]}."},
    ]


# ── Chat / instruction ───────────────────────────────────────────────────────


def chat_msgs(r):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in r["messages"]
        if m.get("content")
    ]


def sharegpt_msgs(r):
    """ShareGPT V3: {"conversations": [{"from": "human"|"gpt", "value": ...}]}."""
    role = {"human": "user", "gpt": "assistant", "system": "system"}
    out = []
    for t in r["conversations"]:
        rl = role.get(t.get("from"))
        v = (t.get("value") or "").strip()
        if rl and v:
            out.append({"role": rl, "content": v})
    # Must start with a user/system turn for most chat templates.
    while out and out[0]["role"] == "assistant":
        out.pop(0)
    return out


def ifeval_msgs(r):
    """IFEval: a bare instruction with verifiable constraints. Single turn —
    there is no reference answer, and the prompt itself is the point."""
    return [{"role": "user", "content": r["prompt"]}]


# ── Math ─────────────────────────────────────────────────────────────────────


def metamath_msgs(r):
    return [
        {"role": "user", "content": r["query"]},
        {"role": "assistant", "content": r["response"]},
    ]


def gsm8k_msgs(r):
    return [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r.get("raw_answer") or str(r.get("ground_truth", ""))},
    ]


# ── Knowledge / reasoning MCQ ────────────────────────────────────────────────


def mmlu_msgs(r):
    subj = r.get("subject", "").replace("_", " ")
    pre = f"({subj}) " if subj else ""
    return _mcq(r["question"], r["choices"], int(r["answer"]), pre)


def arc_msgs(r):
    labels = r["choices_label"]
    key = r["answerKey"]
    idx = labels.index(key) if key in labels else 0
    return _mcq(r["question"], r["choices_text"], idx)


def hellaswag_msgs(r):
    ctx = (r["ctx_a"] + " " + r.get("ctx_b", "")).strip()
    q = f"{r.get('activity_label','')}: {ctx}"
    return _mcq("Which ending is most plausible?\n" + q, r["endings"], int(r["label"]))


def winogrande_msgs(r):
    q = "Which option correctly fills the blank?\n" + r["sentence"]
    return _mcq(q, [r["option1"], r["option2"]], int(r["answer"]) - 1)


def gpqa_msgs(r):
    """GPQA CSV row. Choice order is shuffled deterministically per row so the
    correct answer is not always A (which would make routing degenerate)."""
    q = r["Question"]
    correct = r["Correct Answer"]
    choices = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    choices = [c for c in choices if c]
    rnd = random.Random(hash(q) & 0xFFFFFFFF)
    order = list(range(len(choices)))
    rnd.shuffle(order)
    shuffled = [choices[i] for i in order]
    return _mcq(q, shuffled, order.index(0))


# ── Code / tools ─────────────────────────────────────────────────────────────


def magicoder_msgs(r):
    return [
        {"role": "user", "content": r["instruction"]},
        {"role": "assistant", "content": r["response"]},
    ]


def glaive_msgs(r):
    """glaive-function-calling-v2: 'system' (tool defs) + 'chat' transcript.

    The split is crude but is the one the Step-3.7 run used; keeping it
    identical preserves comparability with that work.
    """
    msgs = []
    sys_txt = (r.get("system") or "").strip()
    if sys_txt:
        msgs.append({"role": "system", "content": sys_txt})
    chat = r.get("chat") or ""
    marked = chat.replace("ASSISTANT:", "\x00ASSISTANT:").replace("USER:", "\x00USER:")
    for chunk in marked.split("\x00"):
        chunk = chunk.strip()
        if chunk.startswith("USER:"):
            msgs.append({"role": "user", "content": chunk[5:].strip()})
        elif chunk.startswith("ASSISTANT:"):
            content = chunk[10:].strip().replace("<|endoftext|>", "")
            msgs.append({"role": "assistant", "content": content})
    return [m for m in msgs if m["content"]]


def aya_msgs(r):
    """Aya: human-written instruction/response in one of 65 languages. The
    language tag is prefixed so the routing seen is genuinely that language's,
    not English-with-foreign-content."""
    return [
        {"role": "user", "content": r["inputs"]},
        {"role": "assistant", "content": r["targets"]},
    ]


def mmmlu_msgs(r):
    """MMMLU: MMLU professionally translated into 14 languages. CSV columns
    are ['', Question, A, B, C, D, Answer, Subject]."""
    choices = [r["A"], r["B"], r["C"], r["D"]]
    idx = LETTERS.index(r["Answer"].strip().upper())
    subj = (r.get("Subject") or "").replace("_", " ")
    return _mcq(r["Question"], choices, idx, f"({subj}) " if subj else "")


def xsum_msgs(r):
    """Summarization as a TASK — reading prose and producing a summary routes
    differently from reading prose alone."""
    return [
        {"role": "user", "content": "Summarize the following article in one sentence.\n\n"
                                    + r["document"]},
        {"role": "assistant", "content": r["summary"]},
    ]


def hhrlhf_msgs(r):
    """Anthropic hh-rlhf: a "\n\nHuman: ... \n\nAssistant: ..." transcript.
    The harmless-base split carries refusals, which route distinctly from
    ordinary helpful replies."""
    text = r.get("chosen") or ""
    msgs = []
    for chunk in text.replace("\n\nAssistant:", "\x00Assistant:").replace(
        "\n\nHuman:", "\x00Human:"
    ).split("\x00"):
        chunk = chunk.strip()
        if chunk.startswith("Human:"):
            msgs.append({"role": "user", "content": chunk[6:].strip()})
        elif chunk.startswith("Assistant:"):
            msgs.append({"role": "assistant", "content": chunk[10:].strip()})
    return [m for m in msgs if m["content"]]


def writingprompts_msgs(r):
    """Creative GENERATION, as opposed to reading literary prose."""
    return [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["story"]},
    ]


def pg19_msgs(r):
    """PG19 long books — one row is a whole book, so this is normally used with
    a large SEQ_LEN to exercise long-context routing."""
    title = r.get("short_book_title") or ""
    return [{"role": "user", "content": (f"{title}\n\n" if title else "") + r["text"]}]


def rosetta_msgs(r):
    """Rosetta Code: the same task implemented across ~700 languages — the
    cheapest way to get breadth over programming languages rather than volume
    in the popular few."""
    lang = r.get("language_name") or "code"
    task = r.get("task_name") or ""
    desc = (r.get("task_description") or "")[:2000]
    return [
        {"role": "user", "content": f"Implement '{task}' in {lang}.\n\n{desc}"},
        {"role": "assistant", "content": f"```\n{r['code']}\n```"},
    ]


def stack_msgs(r):
    """the-stack-smol-xs: real source files, 87 languages."""
    lang = r.get("lang") or ""
    return [{"role": "user", "content": f"```{lang}\n{r['content']}\n```"}]


def codexglue_msgs(r):
    """CodeXGLUE code-to-text: function + docstring, go/java/js/php/python/ruby."""
    lang = r.get("language") or ""
    return [
        {"role": "user", "content": f"Explain what this {lang} function does.\n\n"
                                    f"```{lang}\n{r['code']}\n```"},
        {"role": "assistant", "content": r.get("docstring") or ""},
    ]


def promptcompletion_msgs(r):
    """{"prompt": ..., "completion": ...}, e.g. the fable5 CoT traces. The
    prompt may itself be a USER:/ASSISTANT: transcript; kept whole rather than
    re-split, since the reasoning trace is the thing being exercised."""
    return [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["completion"]},
    ]


def text_msgs(r):
    return [{"role": "user", "content": r["text"]}]


SHAPES = {
    # name: (required keys for auto-detection, adapter)
    "chat": (("messages",), chat_msgs),
    "sharegpt": (("conversations",), sharegpt_msgs),
    "metamath": (("query", "response"), metamath_msgs),
    "magicoder": (("instruction", "response"), magicoder_msgs),
    "glaive": (("system", "chat"), glaive_msgs),
    "mmlu": (("question", "choices", "answer"), mmlu_msgs),
    "gsm8k": (("question", "raw_answer"), gsm8k_msgs),
    "arc": (("question", "choices_text", "answerKey"), arc_msgs),
    "hellaswag": (("ctx_a", "endings", "label"), hellaswag_msgs),
    "winogrande": (("sentence", "option1", "option2"), winogrande_msgs),
    "gpqa": (("Question", "Correct Answer"), gpqa_msgs),
    "aya": (("inputs", "targets", "language"), aya_msgs),
    "mmmlu": (("Question", "A", "Answer", "Subject"), mmmlu_msgs),
    "xsum": (("document", "summary"), xsum_msgs),
    "hhrlhf": (("chosen", "rejected"), hhrlhf_msgs),
    "writingprompts": (("prompt", "story"), writingprompts_msgs),
    "pg19": (("short_book_title", "text"), pg19_msgs),
    "rosetta": (("language_name", "code", "task_name"), rosetta_msgs),
    "stack": (("lang", "content", "ext"), stack_msgs),
    "codexglue": (("code", "docstring", "language"), codexglue_msgs),
    "promptcompletion": (("prompt", "completion"), promptcompletion_msgs),
    "ifeval": (("prompt",), ifeval_msgs),
    "text": (("text",), text_msgs),
}

# Most specific first: mmlu's (question, choices, answer) would otherwise
# shadow arc/gpqa, and `text` matches almost anything.
SHAPE_ORDER = (
    "chat", "sharegpt", "glaive", "hhrlhf", "aya", "metamath", "magicoder",
    "arc", "hellaswag", "winogrande", "gpqa", "mmmlu", "mmlu", "gsm8k",
    "xsum", "writingprompts", "pg19", "rosetta", "stack", "codexglue",
    "promptcompletion", "ifeval", "text",
)


def detect_shape(row):
    for name in SHAPE_ORDER:
        keys, _ = SHAPES[name]
        if all(k in row for k in keys):
            return name
    return None


def adapter_for(shape):
    if shape not in SHAPES:
        raise ValueError(
            f"unknown row shape {shape!r}; expected one of {', '.join(SHAPE_ORDER)}"
        )
    return SHAPES[shape][1]


# ── Raw iteration over a source file ─────────────────────────────────────────


RECORD_SUFFIXES = (".jsonl", ".json", ".csv", ".parquet", ".gz")


def _iter_shards(path: Path):
    """A record source may be one file or a directory of shards.

    HuggingFace snapshots are directories of parquet/jsonl shards, so a source
    path that is a directory is read as the concatenation of its record files
    (sorted, so the order is stable).
    """
    if path.is_file():
        return [path]
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in RECORD_SUFFIXES
        and ".cache" not in p.parts
        and not p.name.startswith(".")
    )
    if not files:
        raise ValueError(f"{path}: no record files (.jsonl/.json/.csv/.parquet/.gz)")
    return files


def _iter_one(path: Path):
    suf = path.suffix.lower()
    if suf == ".parquet":
        # Read row-group at a time: these shards can be hundreds of MB and the
        # caller usually wants only a few hundred rows out of them.
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
        return
    if suf == ".gz":
        import gzip

        with gzip.open(path, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suf == ".csv":
        with path.open(newline="") as fh:
            yield from csv.DictReader(fh)
        return
    if suf == ".jsonl":
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    # .json — an array, a {"docs": [...]} wrapper, or (the-stack-smol-xs)
    # JSON-lines mislabelled as .json.
    with path.open() as fh:
        head = fh.read(1)
        fh.seek(0)
        if head != "[" and head != "{":
            raise ValueError(f"{path}: not JSON")
        try:
            d = json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
    if isinstance(d, dict) and isinstance(d.get("docs"), list):
        yield from d["docs"]
    elif isinstance(d, list):
        yield from d
    elif isinstance(d, dict):
        yield d
    else:
        raise ValueError(f"{path}: JSON is neither a list nor a dict with 'docs'")


def iter_rows(path: Path):
    """Rows from a record file, or from every record shard under a directory.

    Handles .jsonl, .json (array / {"docs": []} / JSON-lines mislabelled as
    .json), .csv, .parquet and gzipped .jsonl.gz.
    """
    for shard in _iter_shards(path):
        yield from _iter_one(shard)
