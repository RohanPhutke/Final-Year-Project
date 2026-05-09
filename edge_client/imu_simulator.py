import time
import json
import random
from kafka import KafkaProducer
import os

KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
TOPIC = 'imu-data'

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_imu_sample():
    # Simulate IMU data (replace with real data as needed)
    return {
        'timestamp': time.time(),
        'ax': random.uniform(-2, 2),
        'ay': random.uniform(-2, 2),
        'az': random.uniform(-2, 2),
        'gx': random.uniform(-250, 250),
        'gy': random.uniform(-250, 250),
        'gz': random.uniform(-250, 250),
        'heart': random.randint(60, 120)
    }

if __name__ == "__main__":
    while True:
        data = generate_imu_sample()
        producer.send(TOPIC, data)
        print(f"Sent: {data}")
        time.sleep(0.1)  # 10 Hz
