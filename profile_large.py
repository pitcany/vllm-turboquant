#!/usr/bin/env python3
"""
Large context profiling (64k-131k) using file-based payloads.
Fixes the 'Argument list too long' error from curl.
"""

import json
import time
import subprocess
import math
import os
import threading
import re
import tempfile

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"


def curl_post_file(endpoint, data, timeout=1200):
    """POST with payload written to a temp file to avoid arg length limits."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(data, f)
        tmppath = f.name
    try:
        cmd = [
            "curl", "-s", "-X", "POST",
            f"{BASE_URL}/{endpoint}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmppath}",
            "--max-time", str(timeout),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        if result.returncode != 0:
            return {"error": result.stderr[:500]}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON: {result.stdout[:500]}"}
    finally:
        os.unlink(tmppath)


def curl_stream_file(endpoint, data, timeout=1200):
    """Streaming POST for TTFT measurement."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(data, f)
        tmppath = f.name
    try:
        cmd = [
            "curl", "-s", "-N", "-X", "POST",
            f"{BASE_URL}/{endpoint}",
            "-H", "Content-Type: application/json",
            "-d", f"@{tmppath}",
            "--max-time", str(timeout),
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        os.unlink(tmppath)
        raise


def get_gpu_stats():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        gpus.append({
            "idx": int(parts[0]), "mem_used_mb": int(parts[1]),
            "mem_free_mb": int(parts[2]), "mem_total_mb": int(parts[3]),
            "gpu_util_pct": int(parts[4]), "temp_c": int(parts[5]),
            "power_w": float(parts[6]),
        })
    return gpus


def get_vllm_metrics():
    result = subprocess.run(["curl", "-s", "http://localhost:8000/metrics"],
                          capture_output=True, text=True, timeout=10)
    metrics = {}
    for line in result.stdout.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc{"):
            metrics["kv_usage_pct"] = float(line.split()[-1])
    return metrics


def get_cpu_usage():
    result = subprocess.run(["bash", "-c", "top -bn1 | head -3 | grep 'Cpu'"],
                          capture_output=True, text=True, timeout=5)
    match = re.search(r'(\d+\.?\d*)\s*us', result.stdout)
    return float(match.group(1)) if match else None


def build_haystack_prompt(target_tokens, needles, question):
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


def profile_large(target_tokens):
    print(f"\n{'='*70}")
    print(f"  PROFILING AT ~{target_tokens:,} TOKENS")
    print(f"{'='*70}")

    needles = {
        0.1: "The access code for Project Neptune is TRIDENT-5582.",
        0.3: "Dr. Chen's laboratory is located on floor 47 of Building Sigma.",
        0.5: "The backup server IP address is 10.42.88.201 port 9443.",
        0.7: "The quarterly budget for Division Omega is exactly $4,271,093.",
        0.9: "The launch window for satellite Helios-7 opens at 03:42 UTC.",
    }
    expected = {
        "TRIDENT-5582": "Project Neptune access code",
        "floor 47": "Dr. Chen lab location",
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
    print(f"  Prompt chars: {len(prompt):,}")

    # GPU monitoring
    gpu_samples = []
    stop_gpu = threading.Event()
    def sample_gpu():
        while not stop_gpu.is_set():
            try:
                gpu_samples.append(get_gpu_stats())
            except:
                pass
            time.sleep(1.0)
    gpu_thread = threading.Thread(target=sample_gpu, daemon=True)
    gpu_thread.start()

    # CPU monitoring
    cpu_samples = []
    stop_cpu = threading.Event()
    def sample_cpu():
        while not stop_cpu.is_set():
            u = get_cpu_usage()
            if u is not None:
                cpu_samples.append(u)
            time.sleep(1.0)
    cpu_thread = threading.Thread(target=sample_cpu, daemon=True)
    cpu_thread.start()

    gpu_before = get_gpu_stats()
    vllm_before = get_vllm_metrics()

    # ---- NEEDLE TEST ----
    print(f"  [Needle Test] Sending request...")
    t0 = time.time()
    resp = curl_post_file("chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=max(1200, target_tokens // 50))
    t1 = time.time()
    needle_elapsed = t1 - t0

    if "error" in resp:
        print(f"  NEEDLE ERROR: {resp['error'][:300]}")
    else:
        usage = resp.get("usage", {})
        output = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        found = sum(1 for k in expected if k.lower() in output.lower())
        print(f"  Prompt tokens: {prompt_tokens:,}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Total elapsed: {needle_elapsed:.2f}s")
        print(f"  Needles: {found}/{len(expected)}")
        print(f"  Output: {output[:300]}...")

    # ---- STREAMING TEST (TTFT) ----
    print(f"\n  [Streaming Test] Measuring TTFT...")
    filler_block = "The digital frontier expands as new technologies emerge. Data centers process millions of requests. "
    filler_chars = target_tokens * 4 - 200
    filler = ""
    while len(filler) < filler_chars:
        filler += filler_block
    filler = filler[:filler_chars]
    gen_prompt = filler + "\n\nWrite a haiku about the ocean."

    stream_data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": gen_prompt}],
        "max_tokens": 100,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }

    # Write to file for streaming
    stream_path = "/tmp/stream_payload.json"
    with open(stream_path, 'w') as f:
        json.dump(stream_data, f)

    t_gen_start = time.time()
    proc = subprocess.Popen(
        ["curl", "-s", "-N", "-X", "POST", f"{BASE_URL}/chat/completions",
         "-H", "Content-Type: application/json", "-d", f"@{stream_path}",
         "--max-time", str(max(1200, target_tokens // 50))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

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
    proc.wait(timeout=30)
    t_gen_end = time.time()
    total_gen_time = t_gen_end - t_gen_start

    if ttft is not None:
        decode_time = total_gen_time - ttft
        decode_speed = tokens_received / decode_time if decode_time > 0 else 0
        prefill_speed = target_tokens / ttft if ttft > 0 else 0
        print(f"  TTFT: {ttft:.3f}s")
        print(f"  Prefill speed: {prefill_speed:.0f} tok/s")
        print(f"  Decode speed: {decode_speed:.1f} tok/s")
        print(f"  Total: {total_gen_time:.2f}s for {tokens_received} tokens")
        print(f"  Output: {full_output[:100]}...")
    else:
        print("  ERROR: No tokens received from streaming")
        prefill_speed = 0
        decode_speed = 0
        ttft = 0

    # Stop monitoring
    stop_gpu.set()
    stop_cpu.set()
    gpu_thread.join(timeout=2)
    cpu_thread.join(timeout=2)

    gpu_after = get_gpu_stats()
    vllm_after = get_vllm_metrics()

    print(f"\n  [GPU Memory]")
    for g in gpu_after:
        gb = gpu_before[g["idx"]]
        delta = g["mem_used_mb"] - gb["mem_used_mb"]
        print(f"    GPU {g['idx']}: {g['mem_used_mb']} MB ({delta:+d}), "
              f"{g['gpu_util_pct']}% util, {g['temp_c']}C, {g['power_w']:.0f}W")

    peak_mem = [0] * 8
    for sample in gpu_samples:
        for g in sample:
            peak_mem[g["idx"]] = max(peak_mem[g["idx"]], g["mem_used_mb"])
    print(f"    Peak: {[f'{m}' for m in peak_mem]}")

    if cpu_samples:
        print(f"  [CPU] Avg: {sum(cpu_samples)/len(cpu_samples):.1f}%, Peak: {max(cpu_samples):.1f}%")

    kv_pct = vllm_after.get("kv_usage_pct", 0)
    print(f"  [KV Cache] Usage: {kv_pct:.4%}")

    # Theoretical
    prompt_tok = resp.get("usage", {}).get("prompt_tokens", target_tokens) if "error" not in resp else target_tokens
    kv_mb = prompt_tok * 10 * 2 * 256 * 2 * 2 / 1e6
    print(f"  [Theoretical KV] {kv_mb:.1f} MB total, {kv_mb/8:.1f} MB/GPU")

    os.unlink(stream_path)

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": resp.get("usage", {}).get("prompt_tokens", "?") if "error" not in resp else "?",
        "prefill_tok_s": round(prefill_speed, 1),
        "decode_tok_s": round(decode_speed, 1),
        "ttft_s": round(ttft, 3) if ttft else "?",
        "needles": f"{found}/{len(expected)}" if "error" not in resp else "ERR",
        "peak_mem_avg_mb": round(sum(peak_mem) / len(peak_mem)),
        "kv_cache_mb_total": round(kv_mb, 1),
        "cpu_avg": round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else "?",
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  LARGE CONTEXT PROFILING (64k-131k)")
    print("=" * 70)

    results = []
    for size in [64000, 100000, 131000]:
        try:
            r = profile_large(size)
            results.append(r)
        except Exception as e:
            print(f"\n  FAILED at {size}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"target_tokens": size, "error": str(e)})

    print("\n\n" + "=" * 70)
    print("  LARGE CONTEXT SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Context':>10} {'Prefill':>10} {'Decode':>10} {'TTFT':>8} {'PeakVRAM':>10} {'KV total':>10} {'CPU':>6} {'Needles':>8}")
    print(f"{'tokens':>10} {'tok/s':>10} {'tok/s':>10} {'(s)':>8} {'(MB/GPU)':>10} {'(MB)':>10} {'(%)':>6} {'found':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['target_tokens']:>10,} {'ERROR':>10}")
            continue
        print(f"{r['target_tokens']:>10,} {r['prefill_tok_s']:>10} {r['decode_tok_s']:>10} "
              f"{r['ttft_s']:>8} {r['peak_mem_avg_mb']:>10} {r['kv_cache_mb_total']:>10} "
              f"{r['cpu_avg']:>6} {r['needles']:>8}")

    with open("/tmp/profile_large_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to /tmp/profile_large_results.json")
