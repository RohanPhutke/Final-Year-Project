from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, IntegerType
import torch
import json
import os

KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/app/output/results.txt')
MODEL_PATH = '/app/model/model.pth'

# Define IMU schema
schema = StructType([
    StructField('timestamp', DoubleType()),
    StructField('ax', DoubleType()),
    StructField('ay', DoubleType()),
    StructField('az', DoubleType()),
    StructField('gx', DoubleType()),
    StructField('gy', DoubleType()),
    StructField('gz', DoubleType()),
    StructField('heart', IntegerType())
])

# Dummy model for demonstration (replace with actual model class)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.1, 0.9]])  # Always predicts class 1

def load_model():
    # Replace DummyModel with your actual model class
    model = DummyModel()
    # model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def predict_udf(*cols):
    # Convert input columns to tensor and run model
    # Replace with actual preprocessing and inference
    return int(1)  # Always predicts class 1

if __name__ == "__main__":
    spark = SparkSession.builder.appName("IMUInference").getOrCreate()
    model = load_model()

    df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", "imu-data") \
        .load()

    df = df.selectExpr("CAST(value AS STRING)")
    df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

    # For demonstration, just write the raw data to file
    query = df.writeStream \
        .outputMode("append") \
        .format("csv") \
        .option("path", OUTPUT_PATH) \
        .option("checkpointLocation", "/tmp/spark_checkpoint") \
        .start()

    query.awaitTermination()
