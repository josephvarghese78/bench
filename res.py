import json
import numpy as np
import matplotlib.pyplot as plt
import sys


def analyze_performance(data):

    # Extract values
    threads = []
    avg_latency = []
    p95 = []
    p99 = []

    for row in data:
        threads.append(row["max_threads"])
        avg_latency.append(row["avg_response_time"])
        p95.append(row["percentile_95"])
        p99.append(row["percentile_99"])

    threads = np.array(threads)
    avg_latency = np.array(avg_latency)
    p95 = np.array(p95)
    p99 = np.array(p99)

    # ---- Throughput (Little's Law) ----
    throughput = threads / avg_latency

    # ---- Efficiency ----
    efficiency = throughput / threads

    # ---- Detect instability ----
    # Instability if P99 jumps > 2x previous average P99
    p99_baseline = np.mean(p99[:-1])
    unstable = p99[-1] > (2 * p99_baseline)

    # Safe capacity = max throughput before instability
    if unstable:
        safe_threads = threads[-2]
        safe_throughput = throughput[-2]
    else:
        idx = np.argmax(throughput)
        safe_threads = threads[idx]
        safe_throughput = throughput[idx]



    # ---- VERDICT ----
    print("\n===== PERFORMANCE VERDICT =====\n")
    print(f"Peak Throughput: {round(np.max(throughput), 2)} req/sec")
    print(f"Recommended Safe Concurrency: {safe_threads} threads")
    print(f"Estimated Safe Throughput: {round(safe_throughput, 2)} req/sec")

    if unstable:
        print("\n⚠ Instability detected due to P99 spike.")
        print("Likely cause: resource pool or queue bottleneck.")
    else:
        print("\n✅ No instability detected.")

    print("\nSystem Behavior:")
    print("- Average latency trend:",
          "Stable" if np.std(avg_latency) < 0.01 else "Degrading")
    print("- Tail latency trend:",
          "Stable" if np.std(p99[:-1]) < 0.05 else "Volatile")
    print("- Efficiency trend:",
          "Healthy scaling" if efficiency[-1] >= efficiency[-2] else "Efficiency drop detected")

    print("\n================================\n")



with open(f"./t.json", 'r') as f:
    raw_json = json.load(f)

# Your JSON has query as key, so extract first value
#data = list(raw_json.values())[0]

print(raw_json)

analyze_performance(raw_json)