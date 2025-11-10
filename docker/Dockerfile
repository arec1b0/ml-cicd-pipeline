# Production-ready minimal image for the inference service.
# Uses Python 3.11 slim image and installs pip requirements.
# Models are pulled at runtime via the ModelManager, so the image contains only application code.
# Pinned to specific digest for security and reproducibility
FROM python:3.11-slim@sha256:a3c4bc9fe3a7cb781fac7f3c00000553802d405bb878c1c41ae2ccccdecdd59a

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

# Create non-root user (UID 1000) and prepare writable cache directory
RUN groupadd -r appuser -g 1000 \
 && useradd -r -u 1000 -g appuser -m -d /home/appuser -s /sbin/nologin appuser \
 && mkdir -p /var/cache/ml-model \
 && chown -R appuser:appuser /var/cache/ml-model /app

# Expose default FastAPI port
EXPOSE 8000

ENV MODEL_SOURCE=mlflow
ENV MODEL_CACHE_DIR=/var/cache/ml-model
ENV LOG_LEVEL=INFO

# Switch to non-root user
USER appuser

# Start uvicorn server. Use host 0.0.0.0 so container accepts external connections.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
