"""
Kafka-to-Dashboard Bridge
==========================
Consumes fall predictions from Kafka, runs severity analysis + RAG triage,
and pushes real-time alerts to the dashboard API.

This is the connection between:
  Step 1 (Spark inference → Kafka 'predictions' topic)
  Step 2 (Dashboard server → real-time UI)
  Step 3 (RAG pipeline → contextual triage briefs)
"""
import os
import sys
import json
import time
import requests
import numpy as np
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from severity_analyzer import analyze_severity
from rag_pipeline import run_triage

# ── Config ─────────────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
PREDICTIONS_TOPIC = os.environ.get('PREDICTIONS_TOPIC', 'predictions')
DASHBOARD_URL = os.environ.get('DASHBOARD_URL', 'http://localhost:5050')
DASHBOARD_ALERT_ENDPOINT = f"{DASHBOARD_URL}/api/alerts"

# Scenario index mapping (matches config.py)
SCENARIO_TO_IDX = {
    'bed': 0, 'chair': 1, 'clap': 2, 'cloth': 3, 'eat': 4,
    'fall1': 5, 'fall2': 6, 'fall3': 7, 'fall4': 8, 'fall5': 9, 'fall6': 10,
    'hair': 11, 'shoe': 12, 'stair': 13, 'teeth': 14, 'walk': 15,
    'wash': 16, 'write': 17, 'zip': 18,
}


def connect_kafka(broker, topic, retries=30, delay=5):
    """Connect to Kafka consumer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=[broker],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='dashboard-bridge',
                fetch_max_bytes=10485760,
                max_partition_fetch_bytes=10485760,
            )
            print(f"[✓] Connected to Kafka at {broker}, subscribed to '{topic}'")
            return consumer
        except NoBrokersAvailable:
            print(f"[…] Kafka not ready, retrying in {delay}s ({attempt}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Kafka at {broker} after {retries} attempts")


def wait_for_dashboard(url, retries=60, delay=3):
    """Wait for the dashboard server to be ready."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{url}/api/alerts", timeout=3)
            if r.status_code == 200:
                print(f"[✓] Dashboard is ready at {url}")
                return True
        except requests.ConnectionError:
            pass
        print(f"[…] Dashboard not ready, retrying in {delay}s ({attempt}/{retries})")
        time.sleep(delay)
    print(f"[⚠] Dashboard not reachable at {url} — will continue trying on each alert")
    return False


def process_prediction(prediction):
    """
    Process a single prediction from Kafka.
    If it's a fall with sensor data, run severity analysis + RAG + push to dashboard.
    """
    subject_id = prediction.get('subject_id', 'unknown')
    scenario = prediction.get('scenario', 'unknown')
    pred_label = prediction.get('prediction_label', 'unknown')
    confidence = prediction.get('confidence', 0.0)
    sensor_data = prediction.get('sensor_data', None)

    # Only process fall predictions
    if pred_label != 'fall':
        print(f"  🟢 {subject_id}/{scenario} → non-fall (skipping)")
        return None

    print(f"  🔴 FALL DETECTED: {subject_id}/{scenario} (conf: {confidence:.4f})")

    # Get scenario index for severity analyzer
    fall_type_idx = SCENARIO_TO_IDX.get(scenario, 5)  # default to fall1

    # Run severity analysis
    severity_result = None
    triage_brief = None

    if sensor_data is not None:
        sensor_array = np.array(sensor_data, dtype=np.float64)
        print(f"     Running severity analysis on {sensor_array.shape} sensor window...")

        severity_result = analyze_severity(
            sensor_window=sensor_array,
            fall_type_idx=fall_type_idx,
            confidence=confidence,
            is_fall=True,
        )

        print(f"     Severity: {severity_result.severity} | "
              f"Impact: {severity_result.impact_type} | "
              f"Lateral G: {severity_result.lateral_g}g | "
              f"RAG trigger: {severity_result.trigger_rag}")

        # Run RAG pipeline if severity is HIGH
        if severity_result.trigger_rag:
            try:
                triage_brief = run_triage(severity_result)
                if triage_brief:
                    print(f"     ✦ RAG triage brief generated: {triage_brief.triage_color} priority")
                else:
                    print(f"     ✦ RAG triage: not triggered (severity below threshold)")
            except Exception as e:
                print(f"     ⚠ RAG pipeline error: {e}")
    else:
        print(f"     ⚠ No sensor data in prediction — running with simulated severity")
        # Fallback: create a minimal severity result
        from severity_analyzer import SeverityResult
        severity_result = SeverityResult(
            severity="MEDIUM", impact_type="fall_no_sensor_data",
            lateral_g=0.0, vertical_g=0.0, total_g=0.0,
            jerk=0.0, rotation_speed=0.0,
            heart_rate_baseline=0.0, heart_rate_peak=0.0, heart_rate_delta=0.0,
            impact_direction="unknown", final_orientation="unknown",
            fall_type=scenario, confidence=confidence,
            trigger_rag=False, risk_factors=[],
        )

    # Build the alert payload for the dashboard
    alert_payload = {
        'subject_id': subject_id,
        'scenario': scenario,
        'prediction': prediction.get('prediction', 0),
        'prediction_label': pred_label,
        'confidence': confidence,
        'ground_truth': prediction.get('ground_truth', -1),
        'ground_truth_label': prediction.get('ground_truth_label', 'unknown'),
        'severity_analysis': severity_result.to_dict() if severity_result else None,
        'triage_brief': triage_brief.to_dict() if triage_brief else None,
    }

    # Push to dashboard
    try:
        r = requests.post(
            DASHBOARD_ALERT_ENDPOINT,
            json=alert_payload,
            timeout=5,
        )
        if r.status_code in (200, 201):
            resp = r.json()
            alert_id = resp.get('summary', {}).get('alert_id', 'unknown')
            print(f"     ✓ Alert pushed to dashboard: {alert_id}")
        else:
            print(f"     ⚠ Dashboard responded with {r.status_code}: {r.text[:200]}")
    except requests.ConnectionError:
        print(f"     ⚠ Dashboard not reachable at {DASHBOARD_URL}")
    except Exception as e:
        print(f"     ⚠ Error pushing to dashboard: {e}")

    return alert_payload


def main():
    print("=" * 60)
    print("  Kafka → Dashboard Bridge")
    print("  Severity Analysis + RAG Triage Pipeline")
    print("=" * 60)
    print(f"  Kafka broker:    {KAFKA_BROKER}")
    print(f"  Predictions topic: {PREDICTIONS_TOPIC}")
    print(f"  Dashboard URL:   {DASHBOARD_URL}")
    print("=" * 60)

    # Connect to Kafka
    consumer = connect_kafka(KAFKA_BROKER, PREDICTIONS_TOPIC)

    # Wait for dashboard
    wait_for_dashboard(DASHBOARD_URL)

    # Process predictions
    print(f"\n[✓] Listening for predictions on '{PREDICTIONS_TOPIC}'...\n")
    fall_count = 0
    total_count = 0

    for message in consumer:
        total_count += 1
        prediction = message.value

        print(f"\n{'─'*50}")
        print(f"[Message {total_count}] Received prediction:")

        result = process_prediction(prediction)
        if result:
            fall_count += 1
            print(f"     📊 Falls detected so far: {fall_count}/{total_count}")


if __name__ == "__main__":
    main()
