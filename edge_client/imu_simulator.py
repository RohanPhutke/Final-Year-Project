"""
Edge Client — IMU Sensor Data Simulator

Reads real .mat sensor recordings from the dataset and streams them
to Kafka as complete activity windows for fall detection inference.
Each Kafka message represents one activity recording from one subject.
"""
import time
import json
import os
import glob
import random
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import scipy.io as sio

# ── Config ─────────────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
TOPIC = os.environ.get('KAFKA_TOPIC', 'imu-data')
DATASET_DIR = os.environ.get('DATASET_DIR', '/app/dataset')
SEND_INTERVAL = float(os.environ.get('SEND_INTERVAL', '3'))  # seconds between sends

FEATURE_COLUMNS = ['w', 'x', 'y', 'z', 'droll', 'dpitch', 'dyaw', 'ax', 'ay', 'az', 'heart']

# Scenarios that are falls (class 0 = fall)
FALL_SCENARIOS = {'fall1', 'fall2', 'fall3', 'fall4', 'fall5', 'fall6'}


def connect_kafka(broker, retries=30, delay=5):
    """Connect to Kafka with retry logic (Kafka takes time to start in Docker)."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                max_request_size=10485760,  # 10 MB — activity windows can be large
            )
            print(f"[✓] Connected to Kafka at {broker} (attempt {attempt})")
            return producer
        except NoBrokersAvailable:
            print(f"[…] Kafka not ready, retrying in {delay}s ({attempt}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Kafka at {broker} after {retries} attempts")


def load_mat_file(filepath):
    """
    Load a .mat file and extract the 11 sensor features.
    Returns a list of lists (timesteps × features) for JSON serialization.
    """
    data = sio.loadmat(filepath)
    num_timesteps = data['w'].shape[0]

    # Build feature matrix: (num_timesteps, 11)
    features = np.zeros((num_timesteps, len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, col in enumerate(FEATURE_COLUMNS):
        features[:, i] = data[col].flatten()

    return features.tolist()


def discover_recordings(dataset_dir):
    """
    Walk the dataset directory and find all .mat files.
    Returns a list of dicts: {path, subject_id, scenario, is_fall}
    """
    recordings = []
    # Pattern: dataset_dir/subject_XX/{fall,non-fall}/scenario.mat
    mat_files = glob.glob(os.path.join(dataset_dir, 'subject_*', '*', '*.mat'))

    for fpath in sorted(mat_files):
        parts = fpath.split(os.sep)
        # Find subject_XX and the activity class
        subject_id = None
        activity_class = None
        for part in parts:
            if part.startswith('subject_'):
                subject_id = part
            if part in ('fall', 'non-fall'):
                activity_class = part

        scenario = os.path.splitext(os.path.basename(fpath))[0]  # e.g., "fall1", "walk"
        is_fall = scenario in FALL_SCENARIOS

        recordings.append({
            'path': fpath,
            'subject_id': subject_id,
            'scenario': scenario,
            'activity_class': activity_class,
            'is_fall': is_fall,
        })

    return recordings


def main():
    print(f"[*] Edge Client starting — dataset: {DATASET_DIR}, broker: {KAFKA_BROKER}")

    # Discover all recordings
    recordings = discover_recordings(DATASET_DIR)
    if not recordings:
        print(f"[✗] No .mat files found in {DATASET_DIR}")
        print(f"    Listing directory: {os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else 'DIR NOT FOUND'}")
        return

    print(f"[✓] Found {len(recordings)} recordings across {len(set(r['subject_id'] for r in recordings))} subjects")
    fall_count = sum(1 for r in recordings if r['is_fall'])
    print(f"    Falls: {fall_count}, Non-falls: {len(recordings) - fall_count}")

    # Connect to Kafka
    producer = connect_kafka(KAFKA_BROKER)

    # Continuously stream recordings
    cycle = 0
    while True:
        cycle += 1
        random.shuffle(recordings)  # Shuffle to simulate unpredictable arrival
        print(f"\n{'='*60}")
        print(f"[Cycle {cycle}] Streaming {len(recordings)} recordings...")
        print(f"{'='*60}")

        for idx, rec in enumerate(recordings):
            # Load the actual sensor data
            try:
                features = load_mat_file(rec['path'])
            except Exception as e:
                print(f"[✗] Error reading {rec['path']}: {e}")
                continue

            # Build the Kafka message
            message = {
                'subject_id': rec['subject_id'],
                'scenario': rec['scenario'],
                'activity_class': rec['activity_class'],
                'ground_truth': 0 if rec['is_fall'] else 1,  # 0=fall, 1=non-fall
                'ground_truth_label': 'fall' if rec['is_fall'] else 'non-fall',
                'num_timesteps': len(features),
                'num_features': len(FEATURE_COLUMNS),
                'feature_names': FEATURE_COLUMNS,
                'data': features,  # (timesteps, 11) as list of lists
                'timestamp': time.time(),
            }

            producer.send(TOPIC, message)
            producer.flush()

            status = "🔴 FALL" if rec['is_fall'] else "🟢 non-fall"
            print(f"  [{idx+1}/{len(recordings)}] Sent: {rec['subject_id']}/{rec['scenario']} "
                  f"({len(features)} timesteps) → {status}")

            time.sleep(SEND_INTERVAL)

        print(f"\n[Cycle {cycle} complete] Restarting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
