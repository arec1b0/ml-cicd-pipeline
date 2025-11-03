# Production-ready minimal image for the inference service.
# Uses Python 3.11 slim image and installs pip requirements.
# The container expects a model file at $MODEL_PATH (default /app/model_registry/model.pkl).
FROM python:3.11-slim

# Do not write .pyc files and run in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install. Keep image small using --no-cache-dir.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src /app/src

# Expose default FastAPI port
EXPOSE 8000

# Default env var pointing to model path
ENV MODEL_PATH=/app/model_registry/model.pkl
ENV LOG_LEVEL=INFO

# Start uvicorn server. Use host 0.0.0.0 so container accepts external connections.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
