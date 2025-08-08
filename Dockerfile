# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r olfactory && useradd -r -g olfactory olfactory

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM base as production

# Copy application code
COPY --chown=olfactory:olfactory . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R olfactory:olfactory /app

# Set security headers and limits
RUN echo "olfactory soft nofile 65536" >> /etc/security/limits.conf && \
    echo "olfactory hard nofile 65536" >> /etc/security/limits.conf

# Health check script
COPY --chown=olfactory:olfactory docker/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Switch to non-root user
USER olfactory

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Default command
CMD ["python", "-m", "olfactory_transformer.api.server", "--host", "0.0.0.0", "--port", "8000"]