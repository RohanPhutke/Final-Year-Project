"""
Severity Analyzer Module
========================
Analyzes raw sensor signals after a fall is classified to determine
impact severity and characteristics. Sits downstream of the FL model
and triggers the RAG pipeline when a severe lateral impact is detected.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, List
from config import scenario_mapping, feature_columns


# ── Thresholds (tunable) ────────────────────────────────────────────────
LATERAL_G_HIGH = 3.0        # Peak lateral acceleration (g) for HIGH severity
LATERAL_G_MEDIUM = 1.5      # Peak lateral acceleration (g) for MEDIUM severity
JERK_THRESHOLD = 50.0       # Rate of acceleration change threshold
ROTATION_THRESHOLD = 5.0    # Angular velocity (rad/s) at impact
HR_SPIKE_THRESHOLD = 30     # BPM increase over baseline
HR_BASELINE_WINDOW = 10     # Number of initial samples to estimate baseline HR

# Fall type labels that correspond to lateral / high-risk falls
LATERAL_FALL_TYPES = {"fall1", "fall2", "fall3", "fall4", "fall5", "fall6"}


@dataclass
class SeverityResult:
    """Structured output from the severity analysis."""
    severity: str                          # "HIGH", "MEDIUM", "LOW"
    impact_type: str                       # "lateral_impact", "frontal_impact", "backward_fall", etc.
    lateral_g: float                       # Peak lateral acceleration magnitude (g)
    vertical_g: float                      # Peak vertical acceleration (g)
    total_g: float                         # Peak total acceleration magnitude (g)
    jerk: float                            # Maximum jerk (rate of accel change)
    rotation_speed: float                  # Peak angular velocity magnitude (rad/s)
    heart_rate_baseline: float             # Estimated pre-fall HR
    heart_rate_peak: float                 # Post-fall peak HR
    heart_rate_delta: float                # Change in HR
    impact_direction: str                  # "left", "right", "forward", "backward"
    final_orientation: str                 # Approximate resting position description
    fall_type: str                         # Model's classification label (e.g. "fall3")
    confidence: float                      # Model's classification confidence
    trigger_rag: bool                      # Whether RAG pipeline should fire
    risk_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_impact_direction(ax_peak: float, ay_peak: float, az_peak: float) -> str:
    """Estimate the primary direction of impact from peak accelerations."""
    abs_vals = {
        "left": abs(min(ax_peak, 0)),
        "right": abs(max(ax_peak, 0)),
        "forward": abs(max(ay_peak, 0)),
        "backward": abs(min(ay_peak, 0)),
    }
    return max(abs_vals, key=abs_vals.get)


def _estimate_final_orientation(w: float, x: float, y: float, z: float) -> str:
    """Rough estimation of final resting orientation from quaternion."""
    # Convert quaternion to approximate roll/pitch
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)

    if abs(roll_deg) > 60:
        return "lying_on_side"
    elif pitch_deg > 45:
        return "face_down"
    elif pitch_deg < -45:
        return "face_up"
    else:
        return "upright_or_sitting"


def analyze_severity(
    sensor_window: np.ndarray,
    fall_type_idx: int,
    confidence: float = 1.0,
    is_fall: bool = True,
) -> SeverityResult:
    """
    Analyze the severity of a detected fall from raw sensor data.

    Parameters
    ----------
    sensor_window : np.ndarray
        Shape (seq_len, num_features). Features in order of config.feature_columns:
        [w, x, y, z, droll, dpitch, dyaw, ax, ay, az, heart]
    fall_type_idx : int
        Predicted scenario index from the model (maps to scenario_mapping).
    confidence : float
        Softmax confidence of the prediction.
    is_fall : bool
        Whether the binary classifier flagged this as a fall.

    Returns
    -------
    SeverityResult
        Full severity analysis with RAG trigger flag.
    """
    fall_type = scenario_mapping.get(fall_type_idx, f"unknown_{fall_type_idx}")

    # If not a fall, return LOW severity immediately
    if not is_fall:
        return SeverityResult(
            severity="NONE",
            impact_type="no_fall",
            lateral_g=0.0, vertical_g=0.0, total_g=0.0,
            jerk=0.0, rotation_speed=0.0,
            heart_rate_baseline=0.0, heart_rate_peak=0.0, heart_rate_delta=0.0,
            impact_direction="none", final_orientation="upright_or_sitting",
            fall_type=fall_type, confidence=confidence,
            trigger_rag=False, risk_factors=[],
        )

    # ── Extract individual signal channels ──────────────────────────────
    w_quat  = sensor_window[:, 0]   # quaternion w
    x_quat  = sensor_window[:, 1]   # quaternion x
    y_quat  = sensor_window[:, 2]   # quaternion y
    z_quat  = sensor_window[:, 3]   # quaternion z
    droll   = sensor_window[:, 4]   # angular velocity roll
    dpitch  = sensor_window[:, 5]   # angular velocity pitch
    dyaw    = sensor_window[:, 6]   # angular velocity yaw
    ax      = sensor_window[:, 7]   # linear acceleration x
    ay      = sensor_window[:, 8]   # linear acceleration y
    az      = sensor_window[:, 9]   # linear acceleration z
    heart   = sensor_window[:, 10]  # heart rate

    # ── Acceleration analysis ───────────────────────────────────────────
    # Lateral acceleration magnitude (x-y plane)
    lateral_accel = np.sqrt(ax ** 2 + ay ** 2)
    lateral_g = float(np.max(lateral_accel))

    # Vertical acceleration
    vertical_g = float(np.max(np.abs(az)))

    # Total acceleration magnitude
    total_accel = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    total_g = float(np.max(total_accel))

    # Jerk (maximum rate of change of total acceleration)
    accel_diff = np.diff(total_accel)
    jerk = float(np.max(np.abs(accel_diff))) if len(accel_diff) > 0 else 0.0

    # ── Angular velocity analysis ───────────────────────────────────────
    rotation_mag = np.sqrt(droll ** 2 + dpitch ** 2 + dyaw ** 2)
    rotation_speed = float(np.max(rotation_mag))

    # ── Heart rate analysis ─────────────────────────────────────────────
    valid_hr = heart[heart > 0]  # filter zero readings
    if len(valid_hr) > HR_BASELINE_WINDOW:
        hr_baseline = float(np.mean(valid_hr[:HR_BASELINE_WINDOW]))
        hr_peak = float(np.max(valid_hr[HR_BASELINE_WINDOW:]))
    elif len(valid_hr) > 0:
        hr_baseline = float(np.mean(valid_hr))
        hr_peak = float(np.max(valid_hr))
    else:
        hr_baseline = 0.0
        hr_peak = 0.0
    hr_delta = hr_peak - hr_baseline

    # ── Impact direction ────────────────────────────────────────────────
    peak_idx = int(np.argmax(total_accel))
    impact_direction = _estimate_impact_direction(
        float(ax[peak_idx]), float(ay[peak_idx]), float(az[peak_idx])
    )

    # ── Final orientation ───────────────────────────────────────────────
    final_orientation = _estimate_final_orientation(
        float(w_quat[-1]), float(x_quat[-1]),
        float(y_quat[-1]), float(z_quat[-1])
    )

    # ── Risk factors ────────────────────────────────────────────────────
    risk_factors = []
    if lateral_g > LATERAL_G_HIGH:
        risk_factors.append("extreme_lateral_force")
    if jerk > JERK_THRESHOLD:
        risk_factors.append("high_jerk_sudden_deceleration")
    if rotation_speed > ROTATION_THRESHOLD:
        risk_factors.append("uncontrolled_rotation")
    if hr_delta > HR_SPIKE_THRESHOLD:
        risk_factors.append("heart_rate_spike_post_fall")
    if final_orientation == "lying_on_side":
        risk_factors.append("lateral_resting_position")
    if impact_direction in ("left", "right"):
        risk_factors.append("lateral_impact_direction")

    # ── Severity classification ─────────────────────────────────────────
    if lateral_g > LATERAL_G_HIGH and (jerk > JERK_THRESHOLD or rotation_speed > ROTATION_THRESHOLD):
        severity = "HIGH"
        impact_type = "severe_lateral_impact"
    elif lateral_g > LATERAL_G_HIGH or (jerk > JERK_THRESHOLD and rotation_speed > ROTATION_THRESHOLD):
        severity = "HIGH"
        impact_type = "high_energy_fall"
    elif lateral_g > LATERAL_G_MEDIUM:
        severity = "MEDIUM"
        impact_type = "moderate_lateral_impact" if impact_direction in ("left", "right") else "moderate_fall"
    else:
        severity = "LOW"
        impact_type = "minor_fall"

    # Upgrade severity if HR spike is dramatic
    if hr_delta > HR_SPIKE_THRESHOLD * 2 and severity == "MEDIUM":
        severity = "HIGH"
        risk_factors.append("severity_upgraded_hr_spike")

    # ── Determine RAG trigger ───────────────────────────────────────────
    trigger_rag = severity == "HIGH"

    return SeverityResult(
        severity=severity,
        impact_type=impact_type,
        lateral_g=round(lateral_g, 3),
        vertical_g=round(vertical_g, 3),
        total_g=round(total_g, 3),
        jerk=round(jerk, 3),
        rotation_speed=round(rotation_speed, 3),
        heart_rate_baseline=round(hr_baseline, 1),
        heart_rate_peak=round(hr_peak, 1),
        heart_rate_delta=round(hr_delta, 1),
        impact_direction=impact_direction,
        final_orientation=final_orientation,
        fall_type=fall_type,
        confidence=round(confidence, 4),
        trigger_rag=trigger_rag,
        risk_factors=risk_factors,
    )


def analyze_from_tensor(
    sensor_tensor,
    binary_pred: int,
    multi_pred: int,
    confidence: float = 1.0,
) -> SeverityResult:
    """
    Convenience wrapper that accepts a PyTorch tensor (single sample)
    and model predictions. Converts to numpy and calls analyze_severity.
    """
    import torch
    if isinstance(sensor_tensor, torch.Tensor):
        sensor_np = sensor_tensor.cpu().numpy()
    else:
        sensor_np = np.array(sensor_tensor)

    # If shape is (features, seq_len), transpose to (seq_len, features)
    if sensor_np.shape[0] == len(feature_columns) and sensor_np.ndim == 2:
        sensor_np = sensor_np.T

    is_fall = (binary_pred == 0)  # 0 = fall in class_mapping
    return analyze_severity(sensor_np, multi_pred, confidence, is_fall)
