# Production-ready minimal image for the inference service.
# Uses Python 3.11 slim image and installs pip requirements.
# Models are pulled at runtime via the ModelManager, so the image contains only application code.
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

# Prepare a writable cache directory for runtime model downloads
RUN mkdir -p /var/cache/ml-model

# Expose default FastAPI port
EXPOSE 8000

ENV MODEL_SOURCE=mlflow
ENV MODEL_CACHE_DIR=/var/cache/ml-model
ENV LOG_LEVEL=INFO

# Start uvicorn server. Use host 0.0.0.0 so container accepts external connections.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
