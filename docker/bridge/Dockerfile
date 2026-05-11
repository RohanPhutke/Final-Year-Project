FROM python:3.9-slim
WORKDIR /app
COPY kafka_to_dashboard.py .
COPY severity_analyzer.py .
COPY rag_pipeline.py .
COPY ingest_knowledge.py .
COPY config.py .
COPY knowledge_base/ ./knowledge_base/
RUN pip install --no-cache-dir kafka-python requests numpy scipy chromadb scikit-learn
# Initialize the knowledge base at build time
RUN python ingest_knowledge.py
CMD ["python", "kafka_to_dashboard.py"]
