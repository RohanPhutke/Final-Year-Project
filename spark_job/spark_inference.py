"""
Spark Streaming Inference Job — Fall Detection

Consumes complete activity windows from Kafka, runs the trained
federated LSTM model, and outputs fall/non-fall predictions with
confidence scores.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, from_json, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, ArrayType, FloatType
)


# ── Model Definition (must match training code) ───────────────────
class LSTMModel(nn.Module):
    """Same architecture as in run.py / networks.py"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ── Config ─────────────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
INPUT_TOPIC = os.environ.get('INPUT_TOPIC', 'imu-data')
OUTPUT_TOPIC = os.environ.get('OUTPUT_TOPIC', 'predictions')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/app/output/stream_results')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model/model.pth')

# Model hyperparameters (must match training)
INPUT_SIZE = 11
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 2  # binary: fall(0) / non-fall(1)

CLASS_LABELS = {0: 'fall', 1: 'non-fall'}


def load_model(model_path):
    """Load the trained LSTM model."""
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"[✓] Loaded model from {model_path}")
    else:
        print(f"[⚠] Model file not found at {model_path} — using randomly initialized model")

    model.eval()
    return model


def run_inference(model, features_list):
    """
    Run inference on a single activity window.

    Args:
        model: loaded LSTMModel
        features_list: list of lists (timesteps × 11 features)

    Returns:
        (predicted_class, confidence, predicted_label)
    """
    # Convert to tensor: (1, seq_length, 11)
    features_array = np.array(features_list, dtype=np.float32)
    tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)  # (1, 2)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    pred_class = predicted_class.item()
    conf = confidence.item()
    pred_label = CLASS_LABELS.get(pred_class, f'unknown_{pred_class}')

    return pred_class, conf, pred_label


def process_batch(batch_df, batch_id, model, spark):
    """Process each micro-batch of Kafka messages."""
    if batch_df.rdd.isEmpty():
        return

    print(f"\n{'─'*50}")
    print(f"[Batch {batch_id}] Processing {batch_df.count()} messages...")

    # Collect rows to driver for PyTorch inference
    rows = batch_df.collect()
    results = []

    for row in rows:
        try:
            # Parse the JSON message from Kafka
            message = json.loads(row['value'])

            subject_id = message.get('subject_id', 'unknown')
            scenario = message.get('scenario', 'unknown')
            ground_truth = message.get('ground_truth', -1)
            ground_truth_label = message.get('ground_truth_label', 'unknown')
            features_data = message.get('data', [])
            num_timesteps = message.get('num_timesteps', 0)
            send_timestamp = message.get('timestamp', 0)

            if not features_data:
                print(f"  [✗] Empty data for {subject_id}/{scenario}")
                continue

            # Run inference
            pred_class, confidence, pred_label = run_inference(model, features_data)

            # Check if prediction matches ground truth
            correct = pred_class == ground_truth
            status = "✓" if correct else "✗"

            is_fall = pred_label == 'fall'
            icon = "🔴" if is_fall else "🟢"

            print(f"  [{status}] {subject_id}/{scenario} ({num_timesteps} steps) → "
                  f"{icon} {pred_label} (conf: {confidence:.4f}) "
                  f"[truth: {ground_truth_label}]")

            results.append({
                'subject_id': subject_id,
                'scenario': scenario,
                'num_timesteps': num_timesteps,
                'prediction': pred_class,
                'prediction_label': pred_label,
                'confidence': round(confidence, 6),
                'ground_truth': ground_truth,
                'ground_truth_label': ground_truth_label,
                'correct': correct,
                'inference_timestamp': time.time(),
                'send_timestamp': send_timestamp,
            })

        except Exception as e:
            print(f"  [✗] Error processing message: {e}")
            import traceback
            traceback.print_exc()

    if results:
        # Write results to CSV
        results_df = spark.createDataFrame(results)
        results_df.write.mode("append").option("header", "true").csv(OUTPUT_PATH)
        print(f"  [✓] Wrote {len(results)} predictions to {OUTPUT_PATH}")

        # Also write to Kafka predictions topic (for dashboard integration)
        try:
            for r in results:
                result_row = spark.createDataFrame([{'value': json.dumps(r)}])
                result_row.write \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
                    .option("topic", OUTPUT_TOPIC) \
                    .save()
        except Exception as e:
            print(f"  [⚠] Could not write to Kafka '{OUTPUT_TOPIC}' topic: {e}")

    # Print batch summary
    if results:
        total = len(results)
        correct_count = sum(1 for r in results if r['correct'])
        falls_detected = sum(1 for r in results if r['prediction_label'] == 'fall')
        print(f"\n  ── Batch {batch_id} Summary ──")
        print(f"  Total: {total} | Correct: {correct_count}/{total} ({100*correct_count/total:.1f}%) "
              f"| Falls detected: {falls_detected}")


def main():
    print("=" * 60)
    print("  Spark Streaming Inference — Fall Detection")
    print("=" * 60)
    print(f"  Kafka broker:  {KAFKA_BROKER}")
    print(f"  Input topic:   {INPUT_TOPIC}")
    print(f"  Output topic:  {OUTPUT_TOPIC}")
    print(f"  Model path:    {MODEL_PATH}")
    print(f"  Output path:   {OUTPUT_PATH}")
    print("=" * 60)

    # Load model
    model = load_model(MODEL_PATH)

    # Create Spark session
    spark = SparkSession.builder \
        .appName("FallDetection-StreamInference") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", INPUT_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("kafka.max.partition.fetch.bytes", "10485760") \
        .option("fetchOffset.numRetries", "5") \
        .load()

    # Select value column as string
    messages_df = kafka_df.selectExpr("CAST(value AS STRING)")

    # Process using foreachBatch — allows us to run PyTorch inference
    query = messages_df.writeStream \
        .foreachBatch(lambda df, batch_id: process_batch(df, batch_id, model, spark)) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/spark_checkpoint_fall_detection") \
        .trigger(processingTime="5 seconds") \
        .start()

    print("\n[✓] Streaming query started. Waiting for data...\n")
    query.awaitTermination()


if __name__ == "__main__":
    main()
