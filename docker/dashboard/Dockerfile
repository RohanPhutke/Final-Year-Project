FROM python:3.9-slim
WORKDIR /app
COPY dashboard_server.py .
COPY severity_analyzer.py .
COPY rag_pipeline.py .
COPY ingest_knowledge.py .
COPY config.py .
COPY dashboard/ ./dashboard/
COPY knowledge_base/ ./knowledge_base/
RUN pip install --no-cache-dir flask chromadb scikit-learn numpy
# Initialize the knowledge base at build time
RUN python ingest_knowledge.py
EXPOSE 5050
CMD ["python", "dashboard_server.py"]
