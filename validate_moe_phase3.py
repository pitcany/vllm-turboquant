#!/usr/bin/env python3
"""
Phase 3: Measure actual context extension capability.

Since vLLM is running with max_model_len=8192, we measure:
1. How much KV cache memory the model actually uses at different lengths
2. Perplexity-like quality metric via logprobs at different context lengths
3. Whether the model maintains coherence at 8192 (its max)
"""

import json
import time
import subprocess
import math

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"


def curl_post(endpoint, data, timeout=180):
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{BASE_URL}/{endpoint}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data),
        "--max-time", str(timeout),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if result.returncode != 0:
        return {"error": result.stderr}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {result.stdout[:500]}"}


def get_metrics():
    cmd = ["curl", "-s", "http://localhost:8000/metrics"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    metrics = {}
    for line in result.stdout.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            metrics["kv_usage"] = float(line.split()[-1])
    return metrics


# ============================================================
# Test 1: Perplexity-like quality via logprobs
# ============================================================

print("=" * 60)
print("  Logprobs Quality Test at Different Context Lengths")
print("=" * 60)
print()

# Use a structured passage that requires understanding context
test_passages = {
    "short": (
        "The Fibonacci sequence starts with 0 and 1. Each subsequent number is the "
        "sum of the two preceding ones. So the sequence goes: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34. "
        "The ratio of consecutive Fibonacci numbers approaches the golden ratio, which is approximately"
    ),
    "medium": None,  # will be padded version
    "long": None,    # will be padded version
}

# Filler that's semantically unrelated
filler = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Stars twinkle in the night sky as the moon rises above the hills. "
    "The ocean waves crash against the rocky shore in steady rhythm. "
)

core_passage = test_passages["short"]

print("Testing: Can the model complete 'golden ratio is approximately...' after varying filler")
print()

for target_tokens, label in [(200, "~200 tok"), (1000, "~1k tok"), (3000, "~3k tok"),
                              (5000, "~5k tok"), (7000, "~7k tok"), (8000, "~8k tok")]:
    # Build prompt with filler BEFORE the passage
    filler_needed = max(0, target_tokens * 4 - len(core_passage))
    padded_filler = ""
    while len(padded_filler) < filler_needed:
        padded_filler += filler
    padded_filler = padded_filler[:filler_needed]

    prompt = padded_filler + "\n\n" + core_passage

    t0 = time.time()
    resp = curl_post("chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nComplete the number:"}],
        "max_tokens": 20,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    elapsed = time.time() - t0

    if "error" in resp:
        print(f"  {label}: ERROR - {resp['error'][:100]}")
        continue

    usage = resp.get("usage", {})
    output = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""

    # Check if model knows the golden ratio
    correct = any(x in output for x in ["1.618", "1.62", "phi", "1.6180"])

    # Get logprobs from response
    logprobs_data = resp["choices"][0].get("logprobs", {})
    token_logprobs = []
    if logprobs_data and logprobs_data.get("content"):
        for tok_info in logprobs_data["content"]:
            if tok_info.get("logprob") is not None:
                token_logprobs.append(tok_info["logprob"])

    avg_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0
    perplexity = math.exp(-avg_logprob) if avg_logprob < 0 else float("inf")

    print(f"  {label:>10} | tokens={usage.get('prompt_tokens', '?'):>5} | "
          f"ppl={perplexity:>6.2f} | correct={correct} | "
          f"time={elapsed:.2f}s | output: {output[:50]}")


# ============================================================
# Test 2: Multi-needle at maximum context
# ============================================================

print()
print("=" * 60)
print("  Multi-Needle at Maximum Context (8192)")
print("=" * 60)
print()

facts = [
    ("The password for vault Alpha is PHOENIX-2891.", "PHOENIX-2891", 0.1),
    ("The activation code for system Beta is 7743-OMEGA.", "7743-OMEGA", 0.3),
    ("Agent Rivera's emergency contact number is 555-0173.", "555-0173", 0.5),
    ("The launch sequence for Project Gamma is DELTA-ECHO-9.", "DELTA-ECHO-9", 0.7),
    ("The encryption key for Channel Sigma is XJ7-KAPPA-412.", "XJ7-KAPPA-412", 0.9),
]

# Build prompt that fills nearly all of 8192 tokens
target_chars = 7500 * 4  # leave room for question
filler_block = (
    "In the vast expanse of the digital frontier, data streams flow like rivers. "
    "Each byte carries information across networks spanning the globe. "
    "Servers hum in climate-controlled rooms, processing millions of requests. "
    "The architecture of modern computing relies on layers of abstraction. "
    "From silicon transistors to high-level programming languages, each layer "
    "builds upon the previous one to create increasingly complex systems. "
)

segments = []
chars_per_segment = target_chars // (len(facts) + 1)

for i, (fact, _, pos) in enumerate(facts):
    fill = ""
    while len(fill) < chars_per_segment:
        fill += filler_block
    fill = fill[:chars_per_segment]
    segments.append(fill)
    segments.append(f"\n\n{fact}\n\n")

# Final filler
fill = ""
while len(fill) < chars_per_segment:
    fill += filler_block
segments.append(fill[:chars_per_segment])

question = "\n\nAnswer these questions with just the answers, one per line:\n"
question += "1. What is the password for vault Alpha?\n"
question += "2. What is the activation code for system Beta?\n"
question += "3. What is Agent Rivera's emergency contact number?\n"
question += "4. What is the launch sequence for Project Gamma?\n"
question += "5. What is the encryption key for Channel Sigma?\n"

prompt = "".join(segments) + question

t0 = time.time()
resp = curl_post("chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 200,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}, timeout=180)
elapsed = time.time() - t0

if "error" in resp:
    print(f"ERROR: {resp['error'][:200]}")
else:
    usage = resp.get("usage", {})
    output = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""

    print(f"Prompt tokens: {usage.get('prompt_tokens', '?')}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Output:\n{output}")
    print()

    found = 0
    for fact_text, expected, _ in facts:
        if expected.lower() in output.lower():
            found += 1
            print(f"  FOUND: {expected}")
        else:
            print(f"  MISSED: {expected}")

    print(f"\nCoherence score: {found}/{len(facts)} facts retrieved at near-max context")


# ============================================================
# Test 3: Generation quality — does output make sense at 8k context?
# ============================================================

print()
print("=" * 60)
print("  Generation Quality at Max Context")
print("=" * 60)
print()

# Fill context then ask a complex question requiring reasoning
filler_chars = 7000 * 4
padded = ""
while len(padded) < filler_chars:
    padded += filler_block
padded = padded[:filler_chars]

complex_question = (
    "\n\nIgnore all the text above. Now answer this question:\n"
    "A farmer has 3 fields. Field A produces 120 bushels of wheat per acre. "
    "Field B produces 85 bushels per acre. Field C produces 150 bushels per acre. "
    "If the farmer plants 10 acres of A, 15 acres of B, and 8 acres of C, "
    "how many total bushels does he produce? Show your calculation."
)

prompt = padded + complex_question

t0 = time.time()
resp = curl_post("chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 300,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}, timeout=180)
elapsed = time.time() - t0

if "error" in resp:
    print(f"ERROR: {resp['error'][:200]}")
else:
    usage = resp.get("usage", {})
    output = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""
    correct_answer = 10*120 + 15*85 + 8*150  # = 1200 + 1275 + 1200 = 3675

    print(f"Prompt tokens: {usage.get('prompt_tokens', '?')}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Correct answer: {correct_answer}")
    print(f"Output:\n{output[:500]}")
    print(f"\nAnswer correct: {'3675' in output}")

print()
print("=" * 60)
print("  DONE")
print("=" * 60)
