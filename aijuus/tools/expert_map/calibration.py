"""
Calibration corpus loading for the expert activation mapper (DESIGN.md §7.10.8).

Two ways in, both supported, nothing machine-specific baked in:

  1. a PREBUILT corpus file        -> --corpus PATH
  2. RAW SOURCES built at run time -> --source DOMAIN PATH [N] [SHAPE] (repeatable)

`build_corpus.py` turns option 2 into option 1's file, so the two share one code
path and a prebuilt corpus is reproducible from its manifest.

Every path is supplied by the caller. There are no default dataset locations —
this code runs inside the docker image, where host paths do not exist.

────────────────────────────────────────────────────────────────────────────
PREBUILT CORPUS SHAPE  (--corpus)
────────────────────────────────────────────────────────────────────────────
JSONL, one calibration sequence per line, in the order they should be run:

    {"domain": "math", "dataset": "gsm8k", "seq_len": 1024,
     "messages": [{"role": "user",      "content": "..."},
                  {"role": "assistant", "content": "..."}]}

  domain    free-form tag; the per-domain activation breakdown groups by it
  dataset   provenance label, recorded in the output; any string
  messages  chat turns, applied through the SERVED model's own chat template
            at load time
  seq_len   OPTIONAL per-row token cap, so one corpus can mix short and long
            sequences (routing is length-sensitive). Falls back to --seq-len.
  images    OPTIONAL list of image paths. A row with images is a VISION row:
            it is sent through the chat path with the images attached, not as
            bare token ids.

Messages, not token ids: token ids are tokenizer-specific, and the routing
being measured has to be the routing the served model actually does.

────────────────────────────────────────────────────────────────────────────
RAW SOURCES  (--source DOMAIN PATH [N] [SHAPE])
────────────────────────────────────────────────────────────────────────────
PATH may be a file or a directory. SHAPE is auto-detected, or forced:

  row shapes (.json / .jsonl / .csv), detected from the row's keys:
    chat metamath magicoder glaive sharegpt mmlu gsm8k arc hellaswag
    winogrande gpqa ifeval text        (see sources.py)

  non-row kinds, detected from the path:
    raw_text   a .txt file, or a directory of text/source files -> chunked
               into N sequences (prose, books, wikitext, code files)
    images     a directory of images -> vision rows, each paired with a
               rotating instruction prompt. Put a `prompts.txt` in that same
               folder (one prompt per line, # comments ignored) to supply your
               own rotation instead of the built-in one.
"""

import json
import random
from pathlib import Path

import sources

SEQ_LEN_DEFAULT = 1024
N_DEFAULT = 77
MIN_TOKENS = 16

# Non-row source kinds, forced as SHAPE or auto-detected from the path.
KIND_RAW_TEXT = "raw_text"
KIND_IMAGES = "images"
NON_ROW_KINDS = (KIND_RAW_TEXT, KIND_IMAGES)

TEXT_SUFFIXES = {".txt", ".raw", ".md", ".py", ".rs", ".ts", ".tsx", ".js", ".c",
                 ".cc", ".cpp", ".h", ".hpp", ".go", ".java", ".sh", ".toml"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Rough chars-per-token, only used to size raw-text chunks before tokenizing.
CHARS_PER_TOKEN = 3.6

# Instructions rotated over the images, so the vision domain is not one prompt
# repeated 130 times. Drop a `prompts.txt` (one prompt per line, blank lines and
# #-comments ignored) into the image folder to replace these — the folder then
# carries both halves of the vision source and nothing here is assumed.
VISION_PROMPTS_FILE = "prompts.txt"
VISION_PROMPTS = [
    "Describe this image in detail.",
    "What is happening in this picture? Be specific about objects and setting.",
    "List everything you can identify in this image.",
    "Write a caption for this image, then explain what makes it notable.",
    "What text, symbols or signage appear in this image? Transcribe what you can.",
    "Describe the composition, lighting and colour of this photograph.",
    "What can you infer about where and when this photo was taken?",
    "Summarize this image in one sentence, then in one paragraph.",
]


class Source:
    """One --source DOMAIN PATH [N] [SHAPE]."""

    def __init__(self, domain, path, n=N_DEFAULT, shape=None, seq_len=None):
        self.domain = domain
        self.path = Path(path)
        self.n = int(n)
        self.shape = shape
        self.seq_len = seq_len

    def kind(self):
        if self.shape in NON_ROW_KINDS:
            return self.shape
        if self.shape:
            return "rows"
        if self.path.is_dir():
            # Priority: images > record shards > loose text. A HuggingFace
            # snapshot is a directory of parquet/jsonl shards that also
            # contains a README.md, so "has a text file" is not sufficient
            # evidence that it is a text corpus.
            kids = list(self.path.rglob("*"))
            if any(p.suffix.lower() in IMAGE_SUFFIXES for p in kids):
                return KIND_IMAGES
            if any(
                p.suffix.lower() in sources.RECORD_SUFFIXES
                and ".cache" not in p.parts
                and not p.name.startswith(".")
                for p in kids
            ):
                return "rows"
            return KIND_RAW_TEXT
        if self.path.suffix.lower() in TEXT_SUFFIXES:
            return KIND_RAW_TEXT
        return "rows"

    def __repr__(self):
        return f"Source({self.domain}, {self.path}, n={self.n}, shape={self.shape})"


def parse_source(fields):
    """fields: DOMAIN PATH [N] [SHAPE] [SEQ_LEN]

    SEQ_LEN is the per-source token cap, so one corpus can mix short and long
    sequences — routing is length-sensitive and a map built only at 1024 tokens
    says nothing about how a 4k prompt routes.
    """
    if not 2 <= len(fields) <= 5:
        raise ValueError(
            f"--source takes DOMAIN PATH [N] [SHAPE] [SEQ_LEN], got {len(fields)}: {fields}"
        )
    domain, path = fields[0], fields[1]
    n = fields[2] if len(fields) >= 3 else N_DEFAULT
    shape = fields[3] if len(fields) >= 4 else None
    seq_len = fields[4] if len(fields) == 5 else None
    if seq_len is not None:
        try:
            seq_len = int(seq_len)
        except ValueError:
            raise ValueError(
                f"--source {domain}: SEQ_LEN must be an integer, got {seq_len!r}"
            ) from None
    if shape is not None and shape not in sources.SHAPES and shape not in NON_ROW_KINDS:
        raise ValueError(
            f"--source {domain}: unknown SHAPE {shape!r}; expected one of "
            f"{', '.join(sources.SHAPE_ORDER)} or {', '.join(NON_ROW_KINDS)}"
        )
    try:
        n = int(n)
    except ValueError:
        raise ValueError(f"--source {domain}: N must be an integer, got {n!r}") from None
    return Source(domain, path, n, shape, seq_len)


# ── Building corpus rows from raw sources ────────────────────────────────────


def _effective_max_chars(max_chars, target_tokens, log, domain):
    """Never let the per-message CHARACTER cap undercut the TOKEN target.

    The cap exists so one pathological transcript cannot dominate the corpus.
    But a flat 32768-char cap silently truncates a 32768-TOKEN request to
    ~9k tokens, and the resulting corpus quietly has no long-context rows at
    all — which is exactly the gap it was added to fill. Raise the cap to fit
    the target whenever the target needs more room.
    """
    if not max_chars:
        return 0
    need = int(target_tokens * CHARS_PER_TOKEN * 1.5)
    if need > max_chars:
        log.info(
            "  %s: raising per-message cap %d -> %d chars to fit the %d-token target",
            domain, max_chars, need, target_tokens,
        )
        return need
    return max_chars


def _rows_from_records(src, log, max_chars, sample="head", seed=0):
    """Take N usable rows from a record file.

    sample="head"   first N in file order — reproducible and cheap.
    sample="random" seeded reservoir sample over the WHOLE file, one pass.

    Use "random" for any file that is GROUPED rather than shuffled: MMLU's test
    split is ordered by subject, so head-taking 200 rows yields 3 of its 57
    subjects. Reservoir sampling costs a full read but is the difference
    between a broad corpus and an accidentally narrow one.
    """
    shape = src.shape
    rnd = random.Random(f"{seed}:{src.domain}")
    reservoir, seen = [], 0
    out, indices, skipped = [], [], 0

    for idx, row in enumerate(sources.iter_rows(src.path)):
        if shape is None:
            shape = sources.detect_shape(row)
            if shape is None:
                raise ValueError(
                    f"--source {src.domain}: cannot detect row shape of {src.path} "
                    f"(keys: {sorted(row)[:8]}). Pass SHAPE as the 4th field."
                )
            log.info("  detected row shape: %s", shape)
        to_msgs = sources.adapter_for(shape)
        try:
            msgs = to_msgs(row)
        except Exception:
            skipped += 1
            continue
        if not msgs or sum(len(m["content"]) for m in msgs) < 4 * MIN_TOKENS:
            skipped += 1
            continue
        if max_chars:
            msgs = [{"role": m["role"], "content": m["content"][:max_chars]} for m in msgs]
        item = (idx, msgs)
        if sample == "head":
            out.append(item)
            if len(out) >= src.n:
                break
        else:
            seen += 1
            if len(reservoir) < src.n:
                reservoir.append(item)
            else:
                j = rnd.randrange(seen)
                if j < src.n:
                    reservoir[j] = item

    if sample != "head":
        # Sort by source index so the manifest reads as an ordered selection;
        # the corpus-level shuffle decides run order anyway.
        out = sorted(reservoir, key=lambda t: t[0])
        log.info("  reservoir-sampled %d of %d usable rows", len(out), seen)

    rows = [
        {"domain": src.domain, "dataset": src.path.name, "messages": m}
        for _, m in out
    ]
    indices = [i for i, _ in out]
    return rows, {"shape": shape, "taken": len(rows), "skipped": skipped,
                  "sample": sample, "source_row_indices": indices}


def _text_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file()
                  and p.suffix.lower() in TEXT_SUFFIXES)


def _rows_from_raw_text(src, log, target_tokens):
    """Chunk raw text into N sequences.

    Chunks are taken with a stride across the whole corpus rather than from the
    front, so a long book contributes beginning, middle and end instead of N
    consecutive chunks of chapter one.
    """
    files = _text_files(src.path)
    if not files:
        raise ValueError(f"--source {src.domain}: no text files under {src.path}")
    chunk_chars = int(target_tokens * CHARS_PER_TOKEN)
    chunks = []
    for f in files:
        try:
            txt = f.read_text(errors="replace")
        except Exception:
            continue
        # Whole-file chunking; the stride below decides which survive.
        for i in range(0, len(txt) - chunk_chars // 2, chunk_chars):
            piece = txt[i:i + chunk_chars].strip()
            if len(piece) >= chunk_chars // 2:
                chunks.append((f.name, piece))
    if not chunks:
        raise ValueError(f"--source {src.domain}: {src.path} produced no usable chunks")
    stride = max(1, len(chunks) // src.n)
    picked = chunks[::stride][: src.n]
    out = [
        {"domain": src.domain, "dataset": name,
         "messages": [{"role": "user", "content": piece}]}
        for name, piece in picked
    ]
    log.info("  %d files -> %d chunks, strided to %d", len(files), len(chunks), len(out))
    return out, {"shape": KIND_RAW_TEXT, "taken": len(out), "skipped": 0,
                 "files": len(files), "chunk_chars": chunk_chars}


def _vision_prompts(path: Path, log):
    """Prompts from `<image folder>/prompts.txt`, else the built-in rotation."""
    pf = (path if path.is_dir() else path.parent) / VISION_PROMPTS_FILE
    if not pf.exists():
        return VISION_PROMPTS, None
    lines = [
        ln.strip()
        for ln in pf.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        log.warning("%s is empty — falling back to the built-in prompts", pf)
        return VISION_PROMPTS, None
    log.info("  using %d prompt(s) from %s", len(lines), pf)
    return lines, str(pf)


def _rows_from_images(src, log):
    imgs = sorted(p for p in src.path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not imgs:
        raise ValueError(f"--source {src.domain}: no images under {src.path}")
    prompts, prompts_file = _vision_prompts(src.path, log)
    out = []
    for i in range(src.n):
        img = imgs[i % len(imgs)]
        prompt = prompts[i % len(prompts)]
        out.append({
            "domain": src.domain,
            "dataset": src.path.name,
            "images": [str(img)],
            "messages": [{"role": "user", "content": prompt}],
        })
    log.info("  %d images x %d prompts -> %d vision rows",
             len(imgs), len(prompts), len(out))
    return out, {"shape": KIND_IMAGES, "taken": len(out), "skipped": 0,
                 "images_available": len(imgs), "n_prompts": len(prompts),
                 "prompts_file": prompts_file}


def rows_from_sources(srcs, log, max_chars=0, default_seq_len=SEQ_LEN_DEFAULT,
                      sample="head", seed=0):
    """Read raw sources into corpus rows."""
    all_rows, per_source = [], {}
    for src in srcs:
        if src.n <= 0:
            continue
        if not src.path.exists():
            raise FileNotFoundError(f"--source {src.domain}: no such path: {src.path}")
        kind = src.kind()
        log.info("reading %s from %s (want %d, kind=%s)",
                 src.domain, src.path, src.n, kind)
        target = src.seq_len or default_seq_len
        if kind == KIND_IMAGES:
            rows, meta = _rows_from_images(src, log)
        elif kind == KIND_RAW_TEXT:
            rows, meta = _rows_from_raw_text(src, log, target)
        else:
            eff = _effective_max_chars(max_chars, target, log, src.domain)
            rows, meta = _rows_from_records(src, log, eff, sample, seed)
        if src.seq_len:
            for r in rows:
                r["seq_len"] = src.seq_len
        log.info("  %s -> %d rows (%d skipped)", src.domain, meta["taken"],
                 meta.get("skipped", 0))
        if meta["taken"] < src.n:
            log.warning("  %s: only %d/%d usable rows", src.domain, meta["taken"], src.n)
        meta.update({"path": str(src.path), "dataset": src.path.name,
                     "requested": src.n, "kind": kind})
        per_source[src.domain] = meta
        all_rows.extend(rows)
    return all_rows, per_source


def resolve_image(p, corpus_dir: Path, image_root: Path | None):
    """Find an image referenced by a corpus row.

    A corpus is portable only if its image references survive being moved —
    built on the host but read inside a container where the dataset tree is
    mounted somewhere else entirely. Tried in order:
      1. --image-root joined to the path (explicit override wins)
      2. the path as given (absolute, same machine)
      3. relative to the corpus file's own directory (ships alongside it)
    """
    cand = []
    q = Path(p)
    if image_root is not None:
        cand.append(image_root / q.name)
        if not q.is_absolute():
            cand.append(image_root / q)
    cand.append(q)
    cand.append(corpus_dir / q)
    if not q.is_absolute():
        cand.append(corpus_dir / q.name)
    for c in cand:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"corpus image not found: {p}. Tried {[str(c) for c in cand]}. "
        f"Pass --image-root pointing at the folder holding the images as "
        f"they are visible to THIS process (inside the container, that is the "
        f"mount point, not the host path)."
    )


def manifest_name(path):
    """Sidecar manifest path for a corpus, tolerating a .gz suffix."""
    n = str(path)
    if n.endswith(".gz"):
        n = n[:-3]
    if n.endswith(".jsonl"):
        n = n[:-6]
    return n + ".manifest.json"


def _open_corpus(path: Path):
    """Open a corpus, transparently gunzipping a .gz.

    A corpus is ~20 MB of JSON text and compresses ~3x, which is what makes it
    cheap to bake into the docker image rather than bind-mount at run time.
    """
    if path.suffix.lower() == ".gz":
        import gzip

        return gzip.open(path, "rt")
    return path.open()


def read_corpus(path: Path, image_root: Path | None = None, skip_images=False,
                log=None):
    """Read a corpus file.

    `skip_images` drops vision rows instead of resolving their image paths.
    Consumers that only need token ids (the activation-aware quantizer, which
    concatenates text into a fixed token budget and never sends a chat request)
    would otherwise fail on a mixed corpus whose images are not mounted, for
    rows they were going to discard anyway.
    """
    rows = []
    n_skipped = 0
    corpus_dir = path.parent
    with _open_corpus(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(
                    f"{path}:{lineno}: corpus row has no 'messages' key. "
                    f"See --help for the expected corpus shape."
                )
            row.setdefault("domain", "unknown")
            row.setdefault("dataset", path.name)
            if row.get("images"):
                if skip_images:
                    n_skipped += 1
                    continue
                row["images"] = [
                    resolve_image(p, corpus_dir, image_root) for p in row["images"]
                ]
            rows.append(row)
    if n_skipped and log is not None:
        log.info("skipped %d vision row(s): this consumer reads token ids only",
                 n_skipped)
    return rows


# ── Tokenization ─────────────────────────────────────────────────────────────


def _tokenize(tokenizer, msgs, seq_len):
    out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # transformers >= 5 returns a BatchEncoding, which subclasses UserDict and
    # is therefore NOT an instance of dict — an isinstance(out, dict) check
    # silently yields the KEY LIST instead of token ids, and every row then
    # looks two tokens long. Duck-type on .keys() instead.
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if len(ids) and not isinstance(ids[0], int):
        ids = ids[0]  # batched [1, N]
    return list(ids[:seq_len])


def to_prompts(tokenizer, rows, seq_len, log):
    """Corpus rows -> engine prompts.

    Text rows carry token ids (exact truncation). Vision rows carry their
    messages and image paths instead: they go through the chat path so vLLM's
    multimodal processor expands the image placeholders itself, which we cannot
    do correctly from token ids alone.
    """
    prompts = []
    failed, short, first_error = 0, 0, None
    for row in rows:
        row_len = int(row.get("seq_len") or seq_len)
        if row.get("images"):
            prompts.append({
                "kind": "vision",
                "messages": row["messages"],
                "images": row["images"],
                "domain": row["domain"],
                "dataset": row["dataset"],
                # Approximate: real length depends on image-token expansion.
                "approx_tokens": row_len,
            })
            continue
        try:
            ids = _tokenize(tokenizer, row["messages"], row_len)
        except Exception as e:
            failed += 1
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
            continue
        if len(ids) < MIN_TOKENS:
            short += 1
            continue
        prompts.append({
            "kind": "text",
            "token_ids": ids,
            "domain": row["domain"],
            "dataset": row["dataset"],
        })
    if failed:
        log.warning("%d row(s) failed chat templating; first error: %s", failed, first_error)
    if short:
        log.warning("%d row(s) tokenized to < %d tokens and were dropped", short, MIN_TOKENS)
    if rows and not prompts:
        raise RuntimeError(
            f"every one of the {len(rows)} calibration row(s) was dropped "
            f"({failed} template failures, {short} under {MIN_TOKENS} tokens). "
            f"Check that the model's chat template accepts these message roles."
        )
    return prompts


def build(tokenizer, seq_len, log, corpus=None, srcs=None, seed=0, shuffle=True,
          image_root=None, skip_images=False):
    """Return (prompts, provenance)."""
    from collections import Counter

    if corpus:
        path = Path(corpus)
        if not path.exists():
            raise FileNotFoundError(f"--corpus: no such file: {path}")
        rows = read_corpus(path, Path(image_root) if image_root else None,
                           skip_images=skip_images, log=log)
        # A prebuilt corpus carries its run order; re-shuffling would break
        # comparability between models mapped against the same file.
        manifest = Path(manifest_name(path))
        provenance = {
            "source": "corpus",
            "corpus": str(path),
            "manifest": str(manifest) if manifest.exists() else None,
        }
        log.info("corpus %s: %d rows", path, len(rows))
    elif srcs:
        rows, per_source = rows_from_sources(srcs, log, default_seq_len=seq_len, seed=seed)
        if shuffle:
            random.Random(seed).shuffle(rows)
        provenance = {"source": "sources", "seed": seed, "sources": per_source}
    else:
        raise ValueError("no calibration input: pass --corpus or --source")

    prompts = to_prompts(tokenizer, rows, seq_len, log)
    counts = Counter(p["domain"] for p in prompts)
    n_vision = sum(1 for p in prompts if p["kind"] == "vision")
    provenance["domains"] = dict(counts)
    provenance["sequences"] = len(prompts)
    provenance["vision_sequences"] = n_vision
    provenance["seq_len"] = seq_len
    log.info(
        "calibration: %d sequences (%d vision) %s (~%d text tokens)",
        len(prompts), n_vision, dict(counts),
        sum(len(p.get("token_ids", ())) for p in prompts),
    )
    return prompts, provenance
