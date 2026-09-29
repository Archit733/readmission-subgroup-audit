# Serves the scoring API. Build after running `make pipeline` (needs reports/results/model_for_serving.joblib).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY reports/results/model_for_serving.joblib ./reports/results/model_for_serving.joblib
ENV READMIT_MODEL_PATH=/app/reports/results/model_for_serving.joblib PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "readmit_audit.api:app", "--host", "0.0.0.0", "--port", "8000"]
