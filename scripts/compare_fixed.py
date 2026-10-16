#!/usr/bin/env python3
"""Compare fixed benchmark results against baseline."""

import argparse
import json
import pathlib
import sys


def parse_summary(summary_file: pathlib.Path) -> dict:
    if not summary_file.exists():
        return None
    data = json.loads(summary_file.read_text())
    return {
        "concurrency": data.get("concurrency"),
        "input_len": data.get("input_len"),
        "output_len": data.get("output_len"),
        "tps": data["output_throughput_tps"]["mean"],
        "ttft_p50": data["ttft_p50_s"]["mean"],
        "tpot_p50": data["tpot_p50_ms"]["mean"],
        "status": data.get("status", "UNKNOWN"),
        "successful_rounds": data.get("successful_rounds", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare fixed benchmark results")
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=pathlib.Path("result/base/20261015-092257/fixed"),
        help="Baseline directory containing fixed scenario folders",
    )
    parser.add_argument(
        "--candidate",
        type=pathlib.Path,
        default=pathlib.Path("result/fixed"),
        help="Candidate directory containing fixed scenario folders",
    )
    args = parser.parse_args()

    concurrencies = [1, 8, 16, 32, 64, 128]
    rows = []

    for c in concurrencies:
        scene = f"4096in_1024out_c{c}"
        base_f = args.baseline / scene / "summary.json"
        cand_f = args.candidate / scene / "summary.json"

        b_data = parse_summary(base_f)
        c_data = parse_summary(cand_f)

        if b_data is None:
            continue

        if c_data is None or c_data["tps"] is None or c_data["status"] != "PASS":
            status = c_data["status"] if c_data else "PENDING"
            rows.append({
                "scene": f"4K/1K C{c}",
                "concurrency": c,
                "base_tps": b_data["tps"],
                "cand_tps": None,
                "tps_diff": None,
                "base_ttft": b_data["ttft_p50"],
                "cand_ttft": None,
                "ttft_diff": None,
                "base_tpot": b_data["tpot_p50"],
                "cand_tpot": None,
                "tpot_diff": None,
                "status": status,
            })
            continue

        tps_diff = (c_data["tps"] / b_data["tps"] - 1.0) * 100.0
        ttft_diff = (1.0 - c_data["ttft_p50"] / b_data["ttft_p50"]) * 100.0
        tpot_diff = (1.0 - c_data["tpot_p50"] / b_data["tpot_p50"]) * 100.0

        rows.append({
            "scene": f"4K/1K C{c}",
            "concurrency": c,
            "base_tps": b_data["tps"],
            "cand_tps": c_data["tps"],
            "tps_diff": tps_diff,
            "base_ttft": b_data["ttft_p50"],
            "cand_ttft": c_data["ttft_p50"],
            "ttft_diff": ttft_diff,
            "base_tpot": b_data["tpot_p50"],
            "cand_tpot": c_data["tpot_p50"],
            "tpot_diff": tpot_diff,
            "status": c_data["status"],
        })

    # Print markdown table
    print("| 场景 | 并发 | TPS 安擎基线 | TPS 候选最新 | TPS 相对变化 | TTFT 安擎基线 (s) | TTFT 候选最新 (s) | TTFT 相对提升 | TPOT 安擎基线 (ms) | TPOT 候选最新 (ms) | TPOT 相对提升 | 状态 |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in rows:
        if r["cand_tps"] is None:
            print(f"| {r['scene']} | {r['concurrency']} | {r['base_tps']:.3f} | — | — | {r['base_ttft']:.3f} | — | — | {r['base_tpot']:.3f} | — | — | {r['status']} |")
        else:
            tps_sign = "+" if r["tps_diff"] >= 0 else ""
            ttft_sign = "+" if r["ttft_diff"] >= 0 else ""
            tpot_sign = "+" if r["tpot_diff"] >= 0 else ""
            print(
                f"| {r['scene']} | {r['concurrency']} | {r['base_tps']:.3f} | {r['cand_tps']:.3f} | "
                f"{tps_sign}{r['tps_diff']:.2f}% | {r['base_ttft']:.3f} | {r['cand_ttft']:.3f} | "
                f"{ttft_sign}{r['ttft_diff']:.2f}% | {r['base_tpot']:.3f} | {r['cand_tpot']:.3f} | "
                f"{tpot_sign}{r['tpot_diff']:.2f}% | {r['status']} |"
            )


if __name__ == "__main__":
    main()
