import json
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.lib.units import inch

import numpy as np


def detect_saturation(
        threads,
        avg_latency,
        latency_threshold=0.10,  # 10% latency growth allowed between steps
        efficiency_drop_threshold=0.10,  # 10% efficiency drop allowed
        throughput_flat_threshold=0.05,  # 5% throughput gain considered flat
        max_latency_sla=1.0  # absolute SLA threshold in seconds
):
    """
    Detects safe concurrency and saturation points based on:
    - Throughput growth
    - Latency growth
    - Efficiency drop
    - Absolute latency (SLA)

    Returns:
        dict with:
        - safe_users: max users before saturation
        - safe_throughput: throughput at safe_users
        - saturation_users: users where saturation detected
        - state: SCALING or SATURATION
        - throughput: array of throughput values
        - efficiency: array of efficiency values
        - notes: explanation if SLA is violated
    """

    threads = np.array(threads, dtype=float)
    avg_latency = np.array(avg_latency, dtype=float)

    # Metrics
    throughput = threads / avg_latency
    efficiency = throughput / threads  # per-user throughput

    # Derivatives
    throughput_gain_pct = np.diff(throughput) / throughput[:-1]
    latency_growth = np.diff(avg_latency) / avg_latency[:-1]
    efficiency_drop = -np.diff(efficiency) / efficiency[:-1]

    safe_index = 0
    saturation_index = None
    notes = ""

    for i in range(1, len(threads)):

        # Check absolute SLA
        sla_violation = avg_latency[i] > max_latency_sla
        if sla_violation:
            saturation_index = i
            notes = f"Latency {avg_latency[i]:.3f}s exceeds SLA {max_latency_sla}s"
            break

        # Healthy scaling check
        healthy = (
                throughput_gain_pct[i - 1] > throughput_flat_threshold and
                latency_growth[i - 1] < latency_threshold and
                efficiency_drop[i - 1] < efficiency_drop_threshold
        )

        if healthy:
            safe_index = i
        else:
            saturation_index = i
            notes = "Throughput flattening, latency growth or efficiency drop exceeded threshold"
            break

    state = "SCALING (No saturation detected)" if saturation_index is None else "SATURATION DETECTED"

    return {
        "safe_users": int(threads[safe_index]),
        "safe_throughput": float(throughput[safe_index]),
        "saturation_users": int(threads[saturation_index]) if saturation_index is not None else None,
        "state": state,
        "throughput": throughput,
        "efficiency": efficiency,
        "notes": notes
    }

def detect_saturation1(threads, avg_latency,
                      latency_threshold=0.10,      # 10% latency growth allowed
                      efficiency_drop_threshold=0.10,  # 10% efficiency drop allowed
                      throughput_flat_threshold=0.05   # 5% throughput gain considered flat
                     ):
    """
    Detects safe concurrency and saturation point using:
    - Throughput growth
    - Latency growth
    - Efficiency drop

    Returns:
        dict with detailed results
    """

    threads = np.array(threads, dtype=float)
    avg_latency = np.array(avg_latency, dtype=float)

    # Core metrics
    throughput = threads / avg_latency
    efficiency = throughput / threads   # == 1 / latency

    # Derivatives
    throughput_gain = np.diff(throughput)
    throughput_gain_pct = throughput_gain / throughput[:-1]

    latency_growth = np.diff(avg_latency) / avg_latency[:-1]
    efficiency_drop = -np.diff(efficiency) / efficiency[:-1]

    safe_index = 0
    saturation_index = None

    for i in range(1, len(threads)):

        # Conditions for healthy scaling
        healthy = (
            throughput_gain_pct[i-1] > throughput_flat_threshold and
            latency_growth[i-1] < latency_threshold and
            efficiency_drop[i-1] < efficiency_drop_threshold
        )

        if healthy:
            safe_index = i
        else:
            saturation_index = i
            break

    if saturation_index is None:
        state = "SCALING (No saturation detected)"
    else:
        state = "SATURATION DETECTED"

    return {
        "safe_users": int(threads[safe_index]),
        "safe_throughput": float(throughput[safe_index]),
        "saturation_users": int(threads[saturation_index]) if saturation_index else None,
        "state": state,
        "throughput": throughput,
        "efficiency": efficiency
    }


def generate_report(data, pdf_path="Performance_Report.pdf"):
    # Extract threads and metrics
    threads = np.array([d["max_threads"] for d in data])
    avg_latency = np.array([d["avg_response_time"] for d in data])
    p95 = np.array([d["percentile_95"] for d in data])
    p99 = np.array([d["percentile_99"] for d in data])


    # Calculate throughput and efficiency
    throughput = threads / avg_latency
    efficiency = throughput / threads



    result = detect_saturation(threads, avg_latency)

    print("System State:", result["state"])
    print("Safe Concurrency:", result["safe_users"])
    print("Safe Throughput:", round(result["safe_throughput"], 2))
    print("Saturation Starts At:", result["saturation_users"])
    print("efficiency", result["efficiency"])




    # Determine safe capacity (before last unstable point)
    safe_index = np.argmax(throughput[:])
    safe_threads = threads[safe_index]
    safe_throughput = throughput[safe_index]

    # Generate plots
    def save_plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    save_plot(threads, avg_latency, "Threads", "Avg Latency (sec)", "Average Latency Trend", "avg_latency.png")
    save_plot(threads, p95, "Threads", "P95 Latency (sec)", "P95 Latency Trend", "p95_latency.png")
    save_plot(threads, p99, "Threads", "P99 Latency (sec)", "P99 Latency Trend", "p99_latency.png")
    save_plot(threads, throughput, "Threads", "Throughput (req/sec)", "Throughput Trend", "throughput.png")

    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Performance Test Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.2*inch))

    # Executive Summary
    summary_text = f"""
    The system demonstrates stable scaling behavior up to {safe_threads} concurrent threads.<br/>
    Peak stable throughput: {safe_throughput:.2f} requests/sec.<br/>
    Average latency remains under 200ms for tested loads.<br/>
    P99 latency spike at the highest concurrency indicates potential resource contention.<br/>
    Recommended Production Concurrency: {safe_threads} threads<br/>
    Estimated Safe Throughput: {safe_throughput:.2f} req/sec
    """
    elements.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(summary_text, styles["BodyText"]))
    elements.append(Spacer(1, 0.2*inch))

    # Add charts to PDF
    elements.append(Paragraph("<b>Latency & Throughput Trends</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.1*inch))

    for img in ["avg_latency.png", "p95_latency.png", "p99_latency.png", "throughput.png"]:
        elements.append(Image(img, width=5*inch, height=3*inch))
        elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)
    print(f"PDF report generated at: {pdf_path}")



with open(f"./t.json", 'r') as f:
    raw_json = json.load(f)

# Your JSON has query as key, so extract first value
#data = list(raw_json.values())[0]

generate_report(raw_json, pdf_path="Performance_Report.pdf")


