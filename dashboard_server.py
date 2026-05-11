"""
Dashboard Server
================
Flask backend serving the fall detection dashboard with RAG triage support.

Endpoints:
    GET  /                    → Dashboard UI
    GET  /api/alerts          → List all alerts
    GET  /api/alerts/<id>     → Get specific alert with triage brief
    POST /api/alerts/simulate → Simulate a fall alert (for demo/testing)

Usage:
    python dashboard_server.py
"""

import os
import json
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import datetime
from severity_analyzer import analyze_severity, SeverityResult
from rag_pipeline import run_triage, get_pipeline

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "dashboard"),
    static_folder=os.path.join(os.path.dirname(__file__), "dashboard", "static"),
)

# ── In-memory alert store (replace with DB in production) ───────────────
alerts_store = []


def _generate_simulated_sensor_data(severity_level: str = "HIGH") -> np.ndarray:
    """
    Generate simulated sensor data for demo purposes.
    Returns array of shape (100, 11) matching feature_columns order:
    [w, x, y, z, droll, dpitch, dyaw, ax, ay, az, heart]
    """
    seq_len = 100
    data = np.zeros((seq_len, 11))

    # Quaternion (start upright, end on side for lateral fall)
    t = np.linspace(0, 1, seq_len)
    data[:, 0] = np.cos(t * np.pi / 4)  # w
    data[:, 1] = np.sin(t * np.pi / 4)  # x (roll)
    data[:, 2] = t * 0.1                # y
    data[:, 3] = t * 0.05               # z

    if severity_level == "HIGH":
        # Severe lateral impact
        impact_idx = 60
        data[:, 4] = np.random.normal(0, 0.5, seq_len)  # droll
        data[impact_idx:impact_idx+5, 4] = np.random.uniform(5, 8, 5)
        data[:, 5] = np.random.normal(0, 0.3, seq_len)  # dpitch
        data[:, 6] = np.random.normal(0, 0.3, seq_len)  # dyaw

        # Acceleration — big lateral spike
        data[:, 7] = np.random.normal(0, 0.3, seq_len)  # ax
        data[impact_idx:impact_idx+3, 7] = np.random.uniform(3, 5, 3)
        data[:, 8] = np.random.normal(0, 0.3, seq_len)  # ay
        data[impact_idx:impact_idx+3, 8] = np.random.uniform(2, 4, 3)
        data[:, 9] = np.random.normal(0, 0.2, seq_len)  # az
        data[impact_idx:impact_idx+3, 9] = np.random.uniform(1, 2, 3)

        # Heart rate — baseline 72, spike to 110+
        data[:impact_idx, 10] = np.random.normal(72, 3, impact_idx)
        data[impact_idx:, 10] = np.random.normal(115, 5, seq_len - impact_idx)

    elif severity_level == "MEDIUM":
        impact_idx = 60
        data[:, 4] = np.random.normal(0, 0.3, seq_len)
        data[:, 7] = np.random.normal(0, 0.2, seq_len)
        data[impact_idx:impact_idx+3, 7] = np.random.uniform(1.5, 2.5, 3)
        data[:, 8] = np.random.normal(0, 0.2, seq_len)
        data[:, 9] = np.random.normal(0, 0.2, seq_len)
        data[:, 10] = np.random.normal(75, 3, seq_len)

    else:  # LOW
        data[:, 4:7] = np.random.normal(0, 0.1, (seq_len, 3))
        data[:, 7:10] = np.random.normal(0, 0.3, (seq_len, 3))
        data[:, 10] = np.random.normal(72, 2, seq_len)

    return data


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard UI."""
    return render_template("index.html")


@app.route("/api/alerts", methods=["GET", "POST"])
def handle_alerts():
    """
    GET:  Return all stored alerts.
    POST: Accept a real fall alert from the Kafka bridge (with severity + triage data).
    """
    if request.method == "GET":
        return jsonify({
            "alerts": [a["summary"] for a in alerts_store],
            "count": len(alerts_store),
        })

    # POST — real alert from kafka_to_dashboard bridge
    body = request.get_json(silent=True) or {}
    subject_id = body.get("subject_id", "unknown")
    scenario = body.get("scenario", "unknown")
    confidence = body.get("confidence", 0.0)
    severity_data = body.get("severity_analysis", {})
    triage_data = body.get("triage_brief", None)

    severity = severity_data.get("severity", "MEDIUM") if severity_data else "MEDIUM"
    impact_type = severity_data.get("impact_type", "unknown") if severity_data else "unknown"
    fall_type = severity_data.get("fall_type", scenario) if severity_data else scenario

    alert_id = f"FALL-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(alerts_store)+1:03d}"

    alert_record = {
        "summary": {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "subject_id": subject_id,
            "fall_type": fall_type,
            "severity": severity,
            "impact_type": impact_type,
            "lateral_g": severity_data.get("lateral_g", 0.0),
            "rotation_speed": severity_data.get("rotation_speed", 0.0),
            "heart_rate_delta": severity_data.get("heart_rate_delta", 0.0),
            "impact_direction": severity_data.get("impact_direction", "unknown"),
            "trigger_rag": severity_data.get("trigger_rag", False),
            "confidence": confidence,
            "source": "kafka_stream",
        },
        "severity_analysis": severity_data,
        "triage_brief": triage_data,
    }

    alerts_store.insert(0, alert_record)

    print(f"[{'🔴' if severity == 'HIGH' else '🟡' if severity == 'MEDIUM' else '🟢'}] "
          f"Real alert: {subject_id}/{scenario} → {severity} ({impact_type})")

    return jsonify(alert_record), 201


@app.route("/api/alerts/<alert_id>", methods=["GET"])
def get_alert(alert_id):
    """Return a specific alert with its full triage brief."""
    for alert in alerts_store:
        if alert["summary"]["alert_id"] == alert_id:
            return jsonify(alert)
    return jsonify({"error": "Alert not found"}), 404


@app.route("/api/alerts/simulate", methods=["POST"])
def simulate_alert():
    """
    Simulate a fall detection alert for demo/testing.

    POST body (JSON):
        {
            "severity": "HIGH" | "MEDIUM" | "LOW",   (default: HIGH)
            "fall_type_idx": 5,                       (default: 5 = fall1)
            "subject_id": "subject_14"                (default: subject_14)
        }
    """
    body = request.get_json(silent=True) or {}
    severity_level = body.get("severity", "HIGH")
    fall_type_idx = body.get("fall_type_idx", 5)  # fall1
    subject_id = body.get("subject_id", "subject_14")

    # 1. Generate simulated sensor data
    sensor_data = _generate_simulated_sensor_data(severity_level)

    # 2. Run severity analysis
    severity_result = analyze_severity(
        sensor_window=sensor_data,
        fall_type_idx=fall_type_idx,
        confidence=0.94,
        is_fall=True,
    )

    # 3. Run RAG pipeline if triggered
    triage_brief = None
    triage_dict = None
    try:
        brief = run_triage(severity_result)
        if brief is not None:
            triage_dict = brief.to_dict()
    except Exception as e:
        print(f"⚠️  RAG pipeline error: {e}")
        triage_dict = {"error": str(e)}

    # 4. Build alert record
    alert_id = f"FALL-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(alerts_store)+1:03d}"
    alert_record = {
        "summary": {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "subject_id": subject_id,
            "fall_type": severity_result.fall_type,
            "severity": severity_result.severity,
            "impact_type": severity_result.impact_type,
            "lateral_g": severity_result.lateral_g,
            "rotation_speed": severity_result.rotation_speed,
            "heart_rate_delta": severity_result.heart_rate_delta,
            "impact_direction": severity_result.impact_direction,
            "trigger_rag": severity_result.trigger_rag,
            "source": "simulation",
        },
        "severity_analysis": severity_result.to_dict(),
        "triage_brief": triage_dict,
    }

    alerts_store.insert(0, alert_record)  # newest first

    return jsonify(alert_record), 201


@app.route("/api/knowledge/stats", methods=["GET"])
def knowledge_stats():
    """Return stats about the knowledge base."""
    try:
        pipeline = get_pipeline()
        count = pipeline.collection.count()
        return jsonify({
            "collection": COLLECTION_NAME if 'COLLECTION_NAME' in dir() else "triage_knowledge",
            "document_chunks": count,
            "status": "ready",
        })
    except Exception as e:
        return jsonify({"status": "not_initialized", "error": str(e)}), 500


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Fall Detection Dashboard Server")
    print("  RAG Triage Pipeline Enabled")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:5050")
    print(f"  API:       http://localhost:5050/api/alerts")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=True)
