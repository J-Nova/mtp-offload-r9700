#!/usr/bin/env python3
"""Quick prefill throughput benchmark.

Measures prompt-processing (prefill) throughput by sending prompts of
varying lengths with max_tokens=1 and computing prompt_tokens / TTFT.

Usage:
    python3 bench-prefill.py [--url URL] [--depths 2000,8000,16000,32000,64000] [--runs N] [--auth KEY]
"""
import argparse
import json
import time
import urllib.request


def make_prompt(target_tokens, seed=0):
    """Build a prompt that is approximately target_tokens long.

    Uses a repeating pattern of common words with a unique seed to avoid
    prefix cache hits. Each "hello world" is ~2 tokens.
    """
    # "hello world" is typically 2 tokens in most tokenizers
    words_per_token = 2
    n_phrases = target_tokens // words_per_token
    # Add a unique seed at the start to prevent prefix cache hits
    seed_str = f"seed-{seed}-"
    return seed_str + ("hello world " * n_phrases).strip()


def send_request(url, prompt, max_tokens=1, auth=None):
    """Send a chat completion request and return (ttft_ms, prompt_tokens)."""
    headers = {
        "Content-Type": "application/json",
    }
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    payload = {
        "model": "Qwen3.8-3.6-27B-blend-MXFP4-OCP-GPTQ-mtp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)

    start = time.perf_counter()
    ttft = None
    prompt_tokens = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    chunk = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                if ttft is None and chunk.get("choices"):
                    ttft = (time.perf_counter() - start) * 1000
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    prompt_tokens = usage.get("prompt_tokens")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

    return ttft, prompt_tokens


def main():
    parser = argparse.ArgumentParser(description="Quick prefill throughput benchmark")
    parser.add_argument("--url", default="http://vllm-0:8000/v1/chat/completions",
                        help="API endpoint URL")
    parser.add_argument("--depths", default="2000,8000,16000,32000,64000",
                        help="Comma-separated prompt depths in tokens")
    parser.add_argument("--runs", type=int, default=8,
                        help="Number of runs per depth")
    parser.add_argument("--auth", default=None,
                        help="Bearer auth key")
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]

    print(f"Prefill benchmark: {args.url}")
    print(f"Depths: {depths}, runs: {args.runs}")
    print()

    results = {}
    for depth in depths:
        print(f"Depth {depth}:")
        ttfts = []
        actual_tokens = []

        # Each run uses a unique seed to avoid prefix cache hits
        for i in range(args.runs):
            prompt = make_prompt(depth, seed=i)
            ttft, tokens = send_request(args.url, prompt, max_tokens=1, auth=args.auth)
            if ttft is not None and tokens is not None:
                ttfts.append(ttft)
                actual_tokens.append(tokens)
                pp_tps = tokens / (ttft / 1000)
                print(f"  Run {i+1}: TTFT={ttft:.0f}ms tokens={tokens} PP={pp_tps:.0f} t/s")
            else:
                print(f"  Run {i+1}: FAILED")

        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            avg_tokens = sum(actual_tokens) / len(actual_tokens)
            avg_pp = avg_tokens / (avg_ttft / 1000)
            results[depth] = {
                "avg_ttft_ms": round(avg_ttft, 1),
                "avg_tokens": round(avg_tokens),
                "avg_pp_tps": round(avg_pp),
                "runs": len(ttfts),
            }
            print(f"  AVERAGE: TTFT={avg_ttft:.0f}ms tokens={avg_tokens:.0f} PP={avg_pp:.0f} t/s")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY (avg PP t/s):")
    for depth in depths:
        if depth in results:
            print(f"  {depth:>6} tokens: {results[depth]['avg_pp_tps']} t/s "
                  f"(TTFT {results[depth]['avg_ttft_ms']}ms, actual {results[depth]['avg_tokens']} tokens)")
    print("=" * 60)

    # Save results
    out_path = "results/prefill-aiter.json"
    with open(out_path, "w") as f:
        json.dump({"endpoint": args.url, "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
