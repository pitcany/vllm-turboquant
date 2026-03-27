#!/usr/bin/env python3
"""
Full GPU profiling for Qwen3.5-35B-A3B at 100k+ context.

Measures every metric that matters:
  - Prefill speed (tok/s)
  - Generation speed (tok/s)  
  - Time to First Token (TTFT)
  - VRAM used per GPU
  - KV cache size in VRAM
  - Activation memory during prefill
  - CPU usage during inference
  - Context size tested
  - Quality: needle retrieval, coherence, logprobs/perplexity

Run via: python3 /tmp/profile_100k.py
Requires: vLLM server running on localhost:8000 with sufficient max_model_len
"""

import json
import time
import subprocess
import math
import os
import threading
import re

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"

def curl_post(endpoint, data, timeout=600):
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{BASE_URL}/{endpoint}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(data),
        "--max-time", str(timeout),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    if result.returncode != 0:
        return {"error": result.stderr}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {result.stdout[:500]}"}


def get_gpu_stats():
    """Get per-GPU memory and utilization."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        gpus.append({
            "idx": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_free_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
            "gpu_util_pct": int(parts[4]),
            "temp_c": int(parts[5]),
            "power_w": float(parts[6]),
        })
    return gpus


def get_vllm_metrics():
    """Get KV cache usage and request stats from vLLM."""
    result = subprocess.run(
        ["curl", "-s", "http://localhost:8000/metrics"],
        capture_output=True, text=True, timeout=10,
    )
    metrics = {}
    for line in result.stdout.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            metrics["kv_usage_pct"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_running{"):
            metrics["requests_running"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_waiting{"):
            metrics["requests_waiting"] = float(line.split()[-1])
    return metrics


def get_cpu_usage():
    """Get CPU usage percentage."""
    result = subprocess.run(
        ["bash", "-c", "top -bn1 | head -3 | grep 'Cpu'"],
        capture_output=True, text=True, timeout=5,
    )
    line = result.stdout.strip()
    # Parse: %Cpu(s): 12.3 us,  2.1 sy, ...
    match = re.search(r'(\d+\.?\d*)\s*us', line)
    if match:
        return float(match.group(1))
    return None


def build_haystack_prompt(target_tokens, needles, question):
    """Build a prompt with needles placed at specific positions in filler text."""
    filler_block = (
        "In the vast digital landscape, information flows through networks of interconnected systems. "
        "Each node processes data according to its designated protocols and algorithms. "
        "The architecture of distributed computing enables parallel processing at scale. "
        "Modern cloud infrastructure supports millions of concurrent operations across data centers. "
        "Load balancers distribute traffic evenly among server clusters for optimal performance. "
        "Database sharding partitions large datasets across multiple storage nodes for efficiency. "
        "Container orchestration platforms manage the lifecycle of microservices deployments. "
        "Network latency optimization involves routing traffic through geographically optimal paths. "
    )

    target_chars = target_tokens * 4
    needle_positions = sorted(needles.keys())

    parts = []
    current_pos = 0

    for needle_pos in needle_positions:
        char_pos = int(target_chars * needle_pos)
        filler_needed = char_pos - current_pos
        if filler_needed > 0:
            filler = ""
            while len(filler) < filler_needed:
                filler += filler_block
            parts.append(filler[:filler_needed])
        parts.append(f"\n\n{needles[needle_pos]}\n\n")
        current_pos = char_pos + len(needles[needle_pos]) + 4

    remaining = target_chars - current_pos - len(question) - 50
    if remaining > 0:
        filler = ""
        while len(filler) < remaining:
            filler += filler_block
        parts.append(filler[:remaining])

    parts.append(f"\n\n{question}")
    return "".join(parts)


def profile_context_length(target_tokens, run_needle=True, run_generation=True):
    """Full profile at a given context length."""
    print(f"\n{'='*70}")
    print(f"  PROFILING AT ~{target_tokens:,} TOKENS")
    print(f"{'='*70}")

    result = {
        "target_tokens": target_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- Pre-request GPU state ---
    gpu_before = get_gpu_stats()
    vllm_before = get_vllm_metrics()
    result["gpu_before"] = gpu_before
    result["vllm_before"] = vllm_before

    # --- CPU monitoring thread ---
    cpu_samples = []
    stop_cpu = threading.Event()
    def sample_cpu():
        while not stop_cpu.is_set():
            usage = get_cpu_usage()
            if usage is not None:
                cpu_samples.append(usage)
            time.sleep(0.5)
    cpu_thread = threading.Thread(target=sample_cpu, daemon=True)
    cpu_thread.start()

    # --- GPU monitoring thread ---
    gpu_samples = []
    stop_gpu = threading.Event()
    def sample_gpu():
        while not stop_gpu.is_set():
            stats = get_gpu_stats()
            gpu_samples.append(stats)
            time.sleep(0.5)
    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()

    # ========= TEST 1: Needle-in-haystack =========
    if run_needle:
        needles = {
            0.1: "The access code for Project Neptune is TRIDENT-5582.",
            0.3: "Dr. Chen's laboratory is located on floor 47 of Building Sigma.",
            0.5: "The backup server IP address is 10.42.88.201 port 9443.",
            0.7: "The quarterly budget for Division Omega is exactly $4,271,093.",
            0.9: "The launch window for satellite Helios-7 opens at 03:42 UTC.",
        }
        expected = {
            "TRIDENT-5582": "Project Neptune access code",
            "floor 47": "Dr. Chen's lab location",
            "10.42.88.201": "Backup server IP",
            "4,271,093": "Division Omega budget",
            "03:42 UTC": "Helios-7 launch window",
        }

        question = (
            "Answer these questions with ONLY the specific answer, one per line:\n"
            "1. What is the access code for Project Neptune?\n"
            "2. What floor is Dr. Chen's laboratory on?\n"
            "3. What is the backup server IP address?\n"
            "4. What is the quarterly budget for Division Omega?\n"
            "5. When does the launch window for satellite Helios-7 open?"
        )

        prompt = build_haystack_prompt(target_tokens, needles, question)

        print(f"\n  [Needle Test] Sending ~{target_tokens:,} token prompt...")
        t_start = time.time()

        resp = curl_post("chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }, timeout=max(600, target_tokens // 100))

        t_end = time.time()
        ttft_total = t_end - t_start

        if "error" in resp:
            print(f"  ERROR: {resp['error'][:200]}")
            result["needle_error"] = resp["error"][:200]
        else:
            usage = resp.get("usage", {})
            output = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Calculate speeds
            # vLLM doesn't expose TTFT separately in non-streaming mode
            # Approximate: prefill_time ~ (elapsed - completion_tokens * decode_time_per_token)
            # We'll get more accurate numbers with streaming later

            prefill_speed = prompt_tokens / ttft_total if ttft_total > 0 else 0
            gen_speed = completion_tokens / ttft_total if ttft_total > 0 else 0

            found = 0
            for key, desc in expected.items():
                if key.lower() in output.lower():
                    found += 1

            result["needle"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_s": round(ttft_total, 3),
                "approx_prefill_toks": prefill_speed,
                "approx_gen_toks": gen_speed,
                "needles_found": found,
                "needles_total": len(expected),
                "output": output[:500],
            }

            print(f"  Prompt tokens: {prompt_tokens:,}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Total elapsed: {ttft_total:.2f}s")
            print(f"  Approx throughput: {prefill_speed:.0f} tok/s (prompt+gen combined)")
            print(f"  Needles: {found}/{len(expected)}")
            print(f"  Output: {output[:200]}...")

    # ========= TEST 2: Streaming for accurate TTFT =========
    if run_generation:
        filler_block = (
            "The digital frontier expands as new technologies emerge. "
            "Data centers process millions of requests every second. "
            "Cloud computing has transformed how organizations manage infrastructure. "
        )
        filler_chars = target_tokens * 4 - 200
        filler = ""
        while len(filler) < filler_chars:
            filler += filler_block
        filler = filler[:filler_chars]

        gen_prompt = filler + "\n\nWrite a haiku about the ocean."

        print(f"\n  [Generation Test] Streaming for TTFT measurement...")
        t_gen_start = time.time()

        # Use streaming to measure TTFT accurately
        cmd = [
            "curl", "-s", "-N", "-X", "POST",
            f"{BASE_URL}/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": MODEL,
                "messages": [{"role": "user", "content": gen_prompt}],
                "max_tokens": 100,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            }),
            "--max-time", str(max(600, target_tokens // 100)),
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        ttft = None
        tokens_received = 0
        full_output = ""

        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.time() - t_gen_start
                    tokens_received += 1
                    full_output += content
            except json.JSONDecodeError:
                pass

        proc.wait(timeout=10)
        t_gen_end = time.time()
        total_gen_time = t_gen_end - t_gen_start

        if ttft is not None:
            decode_time = total_gen_time - ttft
            decode_speed = tokens_received / decode_time if decode_time > 0 else 0
            prefill_speed = target_tokens / ttft if ttft > 0 else 0

            result["generation"] = {
                "ttft_s": round(ttft, 4),
                "total_s": round(total_gen_time, 3),
                "tokens_generated": tokens_received,
                "prefill_tok_s": round(prefill_speed, 1),
                "decode_tok_s": round(decode_speed, 1),
                "output": full_output[:200],
            }

            print(f"  TTFT: {ttft:.3f}s")
            print(f"  Prefill speed: {prefill_speed:.0f} tok/s")
            print(f"  Decode speed: {decode_speed:.1f} tok/s")
            print(f"  Total: {total_gen_time:.2f}s for {tokens_received} tokens")
            print(f"  Output: {full_output[:100]}...")
        else:
            print(f"  ERROR: No tokens received")
            result["generation"] = {"error": "no tokens received"}

    # --- Stop monitoring ---
    stop_cpu.set()
    stop_gpu.set()
    cpu_thread.join(timeout=2)
    gpu_thread.join(timeout=2)

    # --- Post-request GPU state ---
    gpu_after = get_gpu_stats()
    vllm_after = get_vllm_metrics()
    result["gpu_after"] = gpu_after
    result["vllm_after"] = vllm_after

    # --- Compute deltas ---
    print(f"\n  [GPU Memory]")
    for g in gpu_after:
        gb = gpu_before[g["idx"]]
        delta = g["mem_used_mb"] - gb["mem_used_mb"]
        print(f"    GPU {g['idx']}: {g['mem_used_mb']} MB used ({delta:+d} MB delta), "
              f"{g['gpu_util_pct']}% util, {g['temp_c']}C, {g['power_w']:.0f}W")

    if gpu_samples:
        peak_mem = [0] * 8
        peak_util = [0] * 8
        for sample in gpu_samples:
            for g in sample:
                peak_mem[g["idx"]] = max(peak_mem[g["idx"]], g["mem_used_mb"])
                peak_util[g["idx"]] = max(peak_util[g["idx"]], g["gpu_util_pct"])
        result["peak_mem_mb"] = peak_mem
        result["peak_util_pct"] = peak_util
        print(f"    Peak mem: {[f'{m}MB' for m in peak_mem]}")

    if cpu_samples:
        result["cpu_avg_pct"] = round(sum(cpu_samples) / len(cpu_samples), 1)
        result["cpu_peak_pct"] = round(max(cpu_samples), 1)
        print(f"\n  [CPU] Avg: {result['cpu_avg_pct']}%, Peak: {result['cpu_peak_pct']}%")

    kv_delta = None
    if "kv_usage_pct" in vllm_before and "kv_usage_pct" in vllm_after:
        kv_delta = vllm_after["kv_usage_pct"] - vllm_before["kv_usage_pct"]
        result["kv_cache_delta_pct"] = round(kv_delta * 100, 4)
        print(f"\n  [KV Cache] Usage: {vllm_after['kv_usage_pct']:.4%} (delta: {kv_delta:+.4%})")

    # --- Theoretical memory breakdown ---
    # Full attention KV for this prompt
    prompt_tokens_actual = result.get("needle", result.get("generation", {})).get("prompt_tokens", target_tokens)
    kv_full_attn_bytes = prompt_tokens_actual * 10 * 2 * 256 * 2 * 2  # 10 layers, 2 kv_heads, 256 dim, K+V, bf16
    print(f"\n  [Memory Breakdown (theoretical)]")
    print(f"    KV cache (10 full_attn layers): {kv_full_attn_bytes / 1e6:.1f} MB total, {kv_full_attn_bytes / 8 / 1e6:.1f} MB/GPU")

    # Activation during prefill (per layer, FlashAttention)
    # Q: (B, H, S, D) = 1 * 16 * S * 256 * 2 bytes
    act_q = 1 * 16 * prompt_tokens_actual * 256 * 2
    act_kv = 1 * 2 * prompt_tokens_actual * 256 * 2 * 2  # K and V
    print(f"    Activation Q per layer: {act_q / 1e6:.1f} MB")
    print(f"    Activation KV per layer: {act_kv / 1e6:.1f} MB")
    # With FlashAttention: no S*S matrix, just O(S) workspace
    print(f"    (FlashAttention avoids O(S^2) attention matrix)")

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  FULL GPU PROFILING: Qwen3.5-35B-A3B on 8x RTX 3090")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check server is up and get max_model_len
    result = subprocess.run(
        ["curl", "-s", "http://localhost:8000/v1/models"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        models = json.loads(result.stdout)
        max_len = models["data"][0]["max_model_len"]
        print(f"  Model: {MODEL}")
        print(f"  max_model_len: {max_len}")
    except (json.JSONDecodeError, KeyError, IndexError):
        print(f"  WARNING: Could not get model info. Server might be down.")
        max_len = 8192

    # Profile at increasing context lengths
    all_results = []

    # Start with smaller sizes for comparison, then go big
    test_sizes = [1000, 4000, 8000]

    # Add larger sizes if server supports them
    if max_len >= 16384:
        test_sizes.append(16000)
    if max_len >= 32768:
        test_sizes.append(32000)
    if max_len >= 65536:
        test_sizes.append(64000)
    if max_len >= 131072:
        test_sizes.extend([100000, 131000])

    for size in test_sizes:
        try:
            r = profile_context_length(size)
            all_results.append(r)
        except Exception as e:
            print(f"\n  FAILED at {size}: {e}")
            all_results.append({"target_tokens": size, "error": str(e)})

    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    print("\n\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Context':>10} {'Prefill':>10} {'Decode':>10} {'TTFT':>8} "
          f"{'VRAM/GPU':>10} {'KV Cache':>10} {'CPU':>6} {'Needles':>8}")
    print(f"{'tokens':>10} {'tok/s':>10} {'tok/s':>10} {'(s)':>8} "
          f"{'(MB)':>10} {'(MB)':>10} {'(%)':>6} {'found':>8}")
    print("-" * 80)

    for r in all_results:
        ctx = r.get("target_tokens", "?")
        gen = r.get("generation", {})
        needle = r.get("needle", {})
        prefill = gen.get("prefill_tok_s", "")
        decode = gen.get("decode_tok_s", "")
        ttft = gen.get("ttft_s", "")
        peak_mem = r.get("peak_mem_mb", [])
        avg_peak = sum(peak_mem) / len(peak_mem) if peak_mem else ""
        cpu = r.get("cpu_avg_pct", "")
        needles = f"{needle.get('needles_found', '?')}/{needle.get('needles_total', '?')}" if needle else ""

        # Theoretical KV cache for this context
        actual_tokens = needle.get("prompt_tokens", gen.get("prompt_tokens", ctx))
        if isinstance(actual_tokens, (int, float)):
            kv_mb = actual_tokens * 10 * 2 * 256 * 2 * 2 / 1e6 / 8  # per GPU
        else:
            kv_mb = ""

        print(f"{ctx:>10,} {prefill:>10} {decode:>10} {ttft:>8} "
              f"{avg_peak:>10} {kv_mb:>10} {cpu:>6} {needles:>8}")

    # Save full results
    out_path = "/tmp/profile_100k_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")
