FROM bitnamilegacy/spark:3.5.0

USER root
WORKDIR /app
COPY spark_job/spark_inference.py .
COPY docker/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pyspark kafka-python numpy
CMD ["sh", "-c", "sleep 20 && spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 --master spark://spark-master:7077 --driver-memory 2g --conf spark.driver.maxResultSize=1g spark_inference.py"]