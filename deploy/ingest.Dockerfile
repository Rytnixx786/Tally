FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "psycopg[binary]>=3.2" "opentelemetry-proto>=1.27" "pydantic>=2.9" "pyyaml>=6.0"
COPY src/tally/__init__.py src/tally/__init__.py
COPY src/tally/ingest src/tally/ingest
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8090
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8090/health').status==200 else 1)"
CMD ["uvicorn", "tally.ingest.app:app", "--host", "0.0.0.0", "--port", "8090"]
