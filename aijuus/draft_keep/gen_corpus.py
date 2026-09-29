#!/usr/bin/env python3
"""Build the draft-vocabulary keep files from a representative generated corpus.

The first freq ranking came from the 29-prompt betterbench corpus x2 samples (~21.8k tokens /
3.2k unique), so the size sweep mostly measured size/composition, not ranking quality. This builds
a much larger corpus: a diverse synthesized prompt set (cross-products of task templates) plus the
betterbench prompts, sampled several completions each, then ranks the model's own output tokens by
frequency.

Run INSIDE a vLLM container (needs the tokenizer + a reachable endpoint):
    /opt/vllm/bin/python3 /patches/aijuus/draft_keep/gen_corpus.py \
        --url http://localhost:8000 --model Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp \
        --samples 3 --max-tokens 512 --concurrency 6 --out-prefix keep

Writes <out-prefix>-{24k,49k,65k,96k}-v2.txt and <out-prefix>-union-v2.txt into
aijuus/draft_keep/, plus the raw corpus text to /tmp/vocab_corpus_v2.txt.
"""
import argparse
import collections
import glob
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LANGS = ["Python", "JavaScript", "TypeScript", "Rust", "Go", "Bash", "SQL", "C++"]
CODE_TASKS = [
    "merges two sorted lists, with a docstring and tests",
    "reads a large CSV in chunks and computes per-column stats",
    "implements a thread-safe LRU cache",
    "parses and validates a nested JSON config with clear errors",
    "retries an HTTP request with exponential backoff and jitter",
    "diffs two unordered dictionaries of records",
    "normalizes unicode text and strips diacritics",
    "implements binary search with a custom comparator",
    "watches a directory for changes and debounces events",
    "converts a nested object to a flat dotted-key form",
]
TOPICS = [
    "the Roman Republic", "how TCP congestion control works", "photosynthesis",
    "the theory of general relativity", "the history of the printing press",
    "how a modern CPU pipeline executes instructions", "ocean acidification",
    "the origins of the Silk Road", "how vaccines train the immune system",
    "the industrial revolution in Britain", "plate tectonics", "how GPS positioning works",
    "the Hubble Space Telescope's discoveries", "the invention of writing",
    "how antibiotics work and resistance develops", "the Byzantine Empire",
    "how compiler optimization passes work", "the water cycle",
    "the history of cryptography", "how mRNA vaccines are designed",
]
JSON_THINGS = [
    "a list of 20 users with id, name, email, role, active",
    "a config for a web server with ports, timeouts, and TLS",
    "a product catalog of 15 items with price and stock",
    "an API error response schema with codes and messages",
    "a nested org chart with departments and managers",
    "a list of 12 log events with timestamp, level, and message",
    "an OpenAPI path item for a paginated list endpoint",
    "a list of 10 tasks with priority, due date, and tags",
]
MATH = [
    "integrate x^2 * exp(x) dx, step by step",
    "find the eigenvalues of [[2,1],[1,2]]",
    "prove that sqrt(2) is irrational",
    "solve the recurrence T(n) = 2T(n/2) + n",
    "compute the determinant of a 4x4 matrix with a zero row",
    "find the limit of (1 + 1/n)^n as n grows",
    "derive the quadratic formula",
    "compute the expected value of a geometric random variable",
]
REASONING = [
    "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. What does the ball cost?",
    "Three switches control three bulbs in a closed room; how do you map them in one visit?",
    "You have two ropes that each burn in 60 minutes unevenly; measure 45 minutes.",
    "How many trailing zeros are in 100 factorial?",
    "Prove that among any five points on a sphere, four lie in a closed hemisphere.",
    "A farmer must cross a river with a wolf, a goat, and cabbage; describe the sequence.",
]
SUMMARIZE = [
    "Summarize the causes and consequences of the 2008 financial crisis.",
    "Summarize how transformer attention changed NLP.",
    "Summarize the main arguments in favor of and against nuclear power.",
    "Summarize the plot of a mystery novel you make up, in one paragraph.",
    "Summarize the key differences between SQL and NoSQL databases.",
]
CHAT = [
    "Plan a 5-day trip to Kyoto on a moderate budget.",
    "Give me a weekly workout plan for a beginner runner.",
    "Help me debug why my sourdough bread is dense.",
    "Explain compound interest to a teenager.",
    "Write a polite email declining a meeting invitation.",
    "Suggest three books for someone who liked Dune.",
]
FILE_EDIT = [
    "Rewrite this function to remove the nested loops and keep behavior: def f(a):\n    out=[]\n    for x in a:\n        for y in a:\n            if x<y: out.append((x,y))\n    return out",
    "Fix the off-by-one bug and explain: total=0\nfor i in range(1,len(x)+1):\n    total+=x[i]",
    "Refactor to early-return style: def g(p):\n    if p:\n        return 1\n    else:\n        return 0",
]
TOOL = [
    "Call the weather tool for Paris tomorrow, then the calendar tool to check 3pm.",
    "Use the search tool to find the latest Python release, then summarize it.",
    "Use the calculator tool to compute 19% of 47,300 and format the result.",
]
MULTILINGUAL = [
    "Translate 'The library closes at six on weekdays' into French, German, and Spanish.",
    "Write a short poem about autumn in English and then in Italian.",
    "Explain recursion in simple Chinese, then in simple English.",
]


def build_prompts():
    ps = []
    for lang in LANGS:
        for task in CODE_TASKS:
            ps.append(f"Write a {lang} function that {task}.")
    for topic in TOPICS:
        ps.append(f"Explain {topic} in detail, with dates and key figures where relevant.")
        ps.append(f"Write a detailed, well-structured essay about {topic}.")
        ps.append(f"What are the most common misconceptions about {topic}, and why are they wrong?")
    for thing in JSON_THINGS:
        ps.append(f"Respond ONLY with JSON: {thing}.")
    ps += MATH
    ps += REASONING
    ps += SUMMARIZE
    ps += CHAT
    ps += FILE_EDIT
    ps += TOOL
    ps += MULTILINGUAL
    return ps


def load_bb_prompts(globs):
    out = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    out.append([{"role": "user",
                                 "content": " ".join(x.get("content", "") for x in d["messages"])}])
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp")
    ap.add_argument("--tok-path", default="/models/Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out-prefix", default="keep")
    ap.add_argument("--bb-globs", nargs="*", default=["/tmp/bbcorpus/*.jsonl"])
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")

    prompts = [[{"role": "user", "content": t}] for t in build_prompts()]
    prompts += load_bb_prompts(a.bb_globs)
    print(f"prompts: {len(prompts)} x {a.samples} samples", flush=True)

    def gen(msgs, temp):
        body = json.dumps({"model": a.model, "messages": msgs,
                           "max_tokens": a.max_tokens, "temperature": temp}).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=600) as f:
            d = json.load(f)
        m = d["choices"][0]["message"]
        return m.get("content") or m.get("reasoning") or ""

    jobs = [(p, 0.7) for p in prompts for _ in range(a.samples)]
    texts = [p[0]["content"] for p in prompts]

    def work(job):
        msgs, temp = job
        try:
            return gen(msgs, temp)
        except Exception as e:
            return f"__ERR__ {e!r}"

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, out in enumerate(ex.map(work, jobs)):
            if not out.startswith("__ERR__"):
                texts.append(out)
            if (i + 1) % 100 == 0:
                print(f"generated {i+1}/{len(jobs)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok_path)
    freq = collections.Counter()
    for t in texts:
        freq.update(tok.encode(t))
    open("/tmp/vocab_corpus_v2.txt", "w").write("\n\n".join(texts))

    V = tok.vocab_size
    ranked = sorted(range(V), key=lambda i: (-freq.get(i, 0), i))
    seed = [int(x) for x in open("/patches/aijuus/draft_keep/qwen38-draft-vocab-49152.txt")]

    def write(name, ids):
        ids = sorted(set(i for i in ids if 0 <= i < V))
        p = "/patches/aijuus/draft_keep/" + name
        open(p, "w").write("\n".join(map(str, ids)) + "\n")
        print(f"{name}: {len(ids)} ids", flush=True)

    write(f"{a.out_prefix}-24k-v2.txt", ranked[:24576])
    write(f"{a.out_prefix}-49k-v2.txt", ranked[:49152])
    write(f"{a.out_prefix}-65k-v2.txt", ranked[:65536])
    write(f"{a.out_prefix}-96k-v2.txt", ranked[:98304])
    write(f"{a.out_prefix}-union-v2.txt", list(set(seed) | set(ranked[:49152])))
    print(f"corpus tokens: {sum(freq.values())}  unique: {len(freq)}", flush=True)


if __name__ == "__main__":
    main()
