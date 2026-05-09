FROM python:3.9-slim
WORKDIR /app
COPY ../../edge_client/imu_simulator.py .
COPY ../../docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "imu_simulator.py"]
