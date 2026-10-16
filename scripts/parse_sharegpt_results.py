#!/usr/bin/env python3
import glob
import json
import os
import sys

def parse_round_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line.strip():
            return None
        data = json.loads(line)
    
    completed = data.get("completed", 0)
    errors = [e for e in data.get("errors", []) if e]
    total_reqs = completed + len(errors)
    success_rate = (completed / total_reqs * 100.0) if total_reqs > 0 else 100.0

    return {
        "concurrency": data.get("max_concurrency"),
        "tps": data.get("output_throughput", 0.0),
        "ttft_mean": data.get("mean_ttft_ms", 0.0),
        "ttft_median": data.get("median_ttft_ms", 0.0),
        "ttft_p99": data.get("p99_ttft_ms", 0.0),
        "tpot_mean": data.get("mean_tpot_ms", 0.0),
        "tpot_median": data.get("median_tpot_ms", 0.0),
        "tpot_p99": data.get("p99_tpot_ms", 0.0),
        "success_rate": success_rate,
    }

def aggregate_concurrency(round_results):
    if not round_results:
        return None
    n = len(round_results)
    avg_tps = sum(r["tps"] for r in round_results) / n
    avg_ttft_mean = sum(r["ttft_mean"] for r in round_results) / n
    avg_ttft_median = sum(r["ttft_median"] for r in round_results) / n
    avg_ttft_p99 = sum(r["ttft_p99"] for r in round_results) / n
    avg_tpot_mean = sum(r["tpot_mean"] for r in round_results) / n
    avg_tpot_median = sum(r["tpot_median"] for r in round_results) / n
    avg_tpot_p99 = sum(r["tpot_p99"] for r in round_results) / n
    avg_succ = sum(r["success_rate"] for r in round_results) / n
    return {
        "concurrency": round_results[0]["concurrency"],
        "rounds": n,
        "tps": avg_tps,
        "ttft_mean": avg_ttft_mean,
        "ttft_median": avg_ttft_median,
        "ttft_p99": avg_ttft_p99,
        "tpot_mean": avg_tpot_mean,
        "tpot_median": avg_tpot_median,
        "tpot_p99": avg_tpot_p99,
        "success_rate": avg_succ,
    }

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/tools/q38-bw100-sglang/result/release/sharegpt"
    concurrencies = [1, 4, 16, 32, 64]
    
    print(f"Scanning base directory: {base_dir}")
    summary_by_c = {}
    
    for c in concurrencies:
        c_dir = os.path.join(base_dir, f"sharegpt-c{c}")
        if not os.path.exists(c_dir):
            continue
        round_files = sorted(glob.glob(os.path.join(c_dir, "round-*.jsonl")))
        round_results = []
        for rf in round_files:
            res = parse_round_file(rf)
            if res:
                round_results.append(res)
        if round_results:
            summary_by_c[c] = aggregate_concurrency(round_results)
            print(f"  C{c}: found {len(round_results)} completed rounds")

    # Baseline data
    baseline = {
        1:  {"tps": 21.454,  "ttft": 200.866, "tpot": 45.727,  "succ": 100},
        4:  {"tps": 106.207, "ttft": 189.951, "tpot": 35.851,  "succ": 100},
        16: {"tps": 307.450, "ttft": 189.460, "tpot": 49.725,  "succ": 100},
        32: {"tps": 435.996, "ttft": 199.106, "tpot": 69.259,  "succ": 100},
        64: {"tps": 528.158, "ttft": 215.717, "tpot": 103.927, "succ": 100},
    }

    print("\n### ShareGPT 基准性能对比表\n")
    print("| 并发数 | TPS (基线) | TPS (实测) | TTFT (基线) | TTFT (实测 mean/median/p99) | TPOT (基线) | TPOT (实测 mean/median/p99) | 成功率 |")
    print("|---|---|---|---|---|---|---|---|")
    
    for c in concurrencies:
        b = baseline[c]
        if c in summary_by_c:
            m = summary_by_c[c]
            ttft_str = f"{m['ttft_mean']:.2f} / {m['ttft_median']:.2f} / {m['ttft_p99']:.2f} ms"
            tpot_str = f"{m['tpot_mean']:.2f} / {m['tpot_median']:.2f} / {m['tpot_p99']:.2f} ms"
            print(f"| {c} | {b['tps']:.3f} | {m['tps']:.3f} | {b['ttft']:.3f} ms | {ttft_str} | {b['tpot']:.3f} ms | {tpot_str} | {m['success_rate']:.1f}% |")
        else:
            print(f"| {c} | {b['tps']:.3f} | 运行中... | {b['ttft']:.3f} ms | 运行中... | {b['tpot']:.3f} ms | 运行中... | 运行中... |")

if __name__ == "__main__":
    main()
