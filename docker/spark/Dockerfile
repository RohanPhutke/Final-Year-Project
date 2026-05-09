# Change this:
# FROM bitnami/spark:3.5.0

# To this:
FROM bitnamilegacy/spark:3.5.0

USER root
WORKDIR /app
COPY ../../spark_job/spark_inference.py .
COPY ../../model/model.pth ./model.pth
COPY ../../docker/requirements.txt .
RUN pip install --no-cache-dir torch pyspark kafka-python
CMD ["sh", "-c", "sleep 15 && spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 --master spark://spark-master:7077 spark_inference.py"]