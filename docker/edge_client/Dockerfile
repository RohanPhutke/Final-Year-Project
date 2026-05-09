FROM python:3.9-slim
WORKDIR /app
COPY edge_client/ ./edge_client/
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "edge_client/imu_simulator.py"]
