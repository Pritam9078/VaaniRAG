import json
from pathlib import Path

def print_pretty_report(json_path="backend/data/benchmark_report.json"):
    path = Path(json_path)
    if not path.exists():
        print(f"Error: Could not find report at {path}")
        return

    with open(path, "r") as f:
        report = json.load(f)

    print("======================================================================")
    print("📊 STAGE-BY-STAGE LATENCY BREAKDOWN (POST-STT)")
    print("======================================================================")
    print(f"{'Stage':<30} {'P50 (ms)':>10} {'P70 (ms)':>10} {'P95 (ms)':>10} {'P100 (ms)':>10}")
    print("-" * 70)

    stages = [k for k in report.keys() if k.endswith("_ms")]
    if "total_ms" in stages:
        stages.remove("total_ms")
        stages.append("total_ms") # ensure total is last

    for stage in stages:
        if stage not in report: continue
        data = report[stage]
        # Handle cases where P95 might not exist, but we have P100 instead, let's map them
        p50 = data.get("P50", 0.0)
        p70 = data.get("P70", 0.0)
        
        # We'll use P100 as the top tail if P95 isn't available
        p95 = data.get("P95", data.get("P100", 0.0))
        p99 = data.get("P99", data.get("P100", 0.0))

        print(f"{stage:<30} {p50:>10.2f} {p70:>10.2f} {p95:>10.2f} {p99:>10.2f}")

    print("======================================================================")
    
    total_data = report.get("total_ms", {})
    t_p50 = total_data.get("P50", 0.0)
    t_p95 = total_data.get("P95", total_data.get("P100", 0.0))
    t_p99 = total_data.get("P99", total_data.get("P100", 0.0))
    
    print(f"TOTAL / WALL (P50: {t_p50:.1f}ms | P95: {t_p95:.2f}ms | P99: {t_p99:.2f}ms)")
    
    target = 200.0
    status = "PASS" if t_p95 <= target else "FAIL"
    print(f"🎯 Latency budget target: {int(target)}ms | Status: {status} ({t_p95:.2f}ms <= {int(target)}ms)")
    print("======================================================================")

if __name__ == "__main__":
    print_pretty_report()
