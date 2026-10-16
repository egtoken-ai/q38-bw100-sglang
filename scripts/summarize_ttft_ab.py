#!/usr/bin/env python3
"""Summarize a repeated A/B produced by scripts/run_ttft_ab_repeat.sh.

Per round and case: TTFT P50 / mean, TPOT P50, output TPS for both arms and
the TTFT P50 improvement. Acceptance: every round and every case must reach
the threshold; cross-case averaging is not used.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

CASES = ("4096in_1024out_c64", "65536in_1024out_c1")


def load_case(run_dir, case):
    rows = [json.loads(line) for line in open(run_dir / case / "round-01" / "requests.jsonl")]
    summary = json.load(open(run_dir / case / "summary.json"))
    rnd = summary["rounds"][0]
    ttft = [r["ttft_s"] for r in rows]
    digest = hashlib.sha256("".join(sorted(r["output_sha256"] for r in rows)).encode()).hexdigest()
    return {
        "ttft_p50_s": statistics.median(ttft),
        "ttft_mean_s": statistics.mean(ttft),
        "tpot_p50_ms": rnd["tpot_ms"]["p50"],
        "output_tps": rnd["output_throughput_tps"],
        "status": rnd["status"],
        "outputs_digest": digest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ab_dir", type=Path, help="result/ab/<tag>")
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--threshold", type=float, default=20.0)
    args = ap.parse_args()

    rounds = sorted({p.name.split("-")[1] for p in args.ab_dir.glob("round-*-*.path")}, key=int)
    report = {"ab_dir": str(args.ab_dir), "threshold_pct": args.threshold, "rounds": []}
    lines = ["| 轮次 | 场景 | 安擎 TTFT P50 | 候选 TTFT P50 | 提升 | 安擎 TPOT P50 | 候选 TPOT P50 | 候选 TPS |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    all_pass = True
    cand_digests = {case: set() for case in CASES}
    for r in rounds:
        entry = {"round": int(r), "cases": {}}
        for case in CASES:
            arms = {}
            for arm in ("native", "candidate"):
                rel = (args.ab_dir / f"round-{r}-{arm}.path").read_text().strip()
                arms[arm] = load_case(args.project / rel, case) | {"dir": rel}
            impr = 100 * (1 - arms["candidate"]["ttft_p50_s"] / arms["native"]["ttft_p50_s"])
            ok = impr >= args.threshold and all(a["status"] == "PASS" for a in arms.values())
            all_pass &= ok
            cand_digests[case].add(arms["candidate"]["outputs_digest"])
            entry["cases"][case] = {**arms, "ttft_p50_improvement_pct": impr, "pass": ok}
            n, c = arms["native"], arms["candidate"]
            lines.append(f"| {r} | {case} | {n['ttft_p50_s']:.3f} | {c['ttft_p50_s']:.3f} | {impr:.2f}% "
                         f"| {n['tpot_p50_ms']:.2f} | {c['tpot_p50_ms']:.2f} | {c['output_tps']:.1f} |")
        report["rounds"].append(entry)
    report["candidate_outputs_stable_across_rounds"] = {k: len(v) == 1 for k, v in cand_digests.items()}
    for case in CASES:
        imps = [e["cases"][case]["ttft_p50_improvement_pct"] for e in report["rounds"]]
        report.setdefault("per_case", {})[case] = {
            "min_improvement_pct": min(imps), "median_improvement_pct": statistics.median(imps)}
    report["all_rounds_all_cases_pass"] = all_pass
    (args.ab_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines.append("")
    for case, s in report["per_case"].items():
        lines.append(f"- {case}: 最小提升 {s['min_improvement_pct']:.2f}%，中位提升 {s['median_improvement_pct']:.2f}%")
    lines.append(f"- 候选输出跨轮一致: {report['candidate_outputs_stable_across_rounds']}")
    lines.append(f"- 每轮每档 TTFT P50 提升 ≥{args.threshold}%: {'通过' if all_pass else '未通过'}")
    (args.ab_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
