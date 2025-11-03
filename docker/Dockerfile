# Production-ready minimal image for the inference service.
# Uses Python 3.11 slim image and installs pip requirements.
# The container expects a model file downloaded from MLflow during build.
FROM python:3.11-slim

# Do not write .pyc files and run in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install. Keep image small using --no-cache-dir.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir mlflow click

# Copy application code
COPY src /app/src

# Copy MLflow download script
COPY docker/download_model.py /app/download_model.py

# Download model from MLflow Model Registry during build
ARG MODEL_URI
ARG MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
RUN python /app/download_model.py --model-uri "${MODEL_URI}" --dst-path /app/model

# Expose default FastAPI port
EXPOSE 8000

# Default env var pointing to model path downloaded from MLflow
# MLflow creates subdirectory structure: /app/model/model/model.pkl
ENV MODEL_PATH=/app/model/model/model.pkl
ENV LOG_LEVEL=INFO

# Start uvicorn server. Use host 0.0.0.0 so container accepts external connections.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
