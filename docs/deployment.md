# Production Deployment Guide

This guide covers deploying the Olfactory Transformer in production environments, from single-server setups to large-scale distributed deployments.

## 🏗️ Architecture Overview

### Deployment Topologies

#### Single Server Deployment
```
┌─────────────────────────────────────┐
│             Server                   │
│  ┌─────────────────────────────────┐ │
│  │    Olfactory Transformer       │ │
│  │  ┌─────────┐  ┌─────────────┐  │ │
│  │  │  Model  │  │   Cache     │  │ │
│  │  └─────────┘  └─────────────┘  │ │
│  │  ┌─────────┐  ┌─────────────┐  │ │
│  │  │Sensors  │  │ Monitoring  │  │ │
│  │  └─────────┘  └─────────────┘  │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

#### Distributed Deployment
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Load Balancer│  │    Gateway   │  │   Database   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
┌──────┴───────────────────────────────────┴───────┐
│              Application Layer                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │Model A  │  │Model B  │  │   Sensor Array  │   │
│  │(GPU-1)  │  │(GPU-2)  │  │                 │   │
│  └─────────┘  └─────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────┐
│                Cache Layer                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │ Redis   │  │ Disk    │  │   Monitoring    │   │
│  │ Cache   │  │ Cache   │  │                 │   │
│  └─────────┘  └─────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────┘
```

## 🐋 Docker Deployment

### Basic Docker Setup

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash olfactory
RUN chown -R olfactory:olfactory /app
USER olfactory

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import olfactory_transformer; print('OK')" || exit 1

# Start command
CMD ["python", "-m", "olfactory_transformer.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  olfactory-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/olfactory-base-v1
      - CACHE_SIZE=1000
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./cache:/app/cache
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-cache:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Multi-Stage Production Build
```dockerfile
# Build stage
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN useradd --create-home --shell /bin/bash olfactory

WORKDIR /app
COPY --chown=olfactory:olfactory . .

USER olfactory

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install the package
RUN pip install --user -e .

EXPOSE 8000

CMD ["python", "-m", "olfactory_transformer.api"]
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: olfactory-transformer
  labels:
    app: olfactory-transformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: olfactory-transformer
  template:
    metadata:
      labels:
        app: olfactory-transformer
    spec:
      containers:
      - name: olfactory-transformer
        image: olfactory-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/olfactory-base-v1"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: cache-storage
          mountPath: /cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: cache-storage
        emptyDir: {}
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: olfactory-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: olfactory-transformer
```

#### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: olfactory-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: olfactory-transformer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Helm Chart
```yaml
# Chart.yaml
apiVersion: v2
name: olfactory-transformer
description: A Helm chart for Olfactory Transformer
type: application
version: 0.1.0
appVersion: "0.1.0"

# values.yaml
replicaCount: 3

image:
  repository: olfactory-transformer
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

resources:
  requests:
    memory: "2Gi"
    cpu: "1"
    nvidia.com/gpu: 1
  limits:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 100Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    labels:
      prometheus: kube-prometheus
```

## ☁️ Cloud Deployments

### AWS Deployment

#### ECS with Fargate
```yaml
# ecs-task-definition.json
{
  "family": "olfactory-transformer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "olfactory-transformer",
      "image": "your-account.dkr.ecr.region.amazonaws.com/olfactory-transformer:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/models/olfactory-base-v1"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-west-2"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "model-storage",
          "containerPath": "/models",
          "readOnly": true
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/olfactory-transformer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ],
  "volumes": [
    {
      "name": "model-storage",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/models"
      }
    }
  ]
}
```

#### Terraform Configuration
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "olfactory_cluster" {
  name = "olfactory-transformer"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_service" "olfactory_service" {
  name            = "olfactory-transformer"
  cluster         = aws_ecs_cluster.olfactory_cluster.id
  task_definition = aws_ecs_task_definition.olfactory_task.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.olfactory_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.olfactory_tg.arn
    container_name   = "olfactory-transformer"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.olfactory_listener]
}

resource "aws_appautoscaling_target" "olfactory_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.olfactory_cluster.name}/${aws_ecs_service.olfactory_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "olfactory_up" {
  name               = "olfactory-scale-up"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.olfactory_target.resource_id
  scalable_dimension = aws_appautoscaling_target.olfactory_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.olfactory_target.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
```

### GCP Deployment

#### Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: olfactory-transformer
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/gpu-type: "nvidia-tesla-t4"
        run.googleapis.com/gpu: "1"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/olfactory-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/olfactory-base-v1"
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        csi:
          driver: gcsfuse.csi.storage.gke.io
          readOnly: true
          volumeAttributes:
            bucketName: "your-model-bucket"
            mountOptions: "implicit-dirs"
```

### Azure Deployment

#### Container Instances
```yaml
# azure-container-instance.yaml
apiVersion: '2021-09-01'
location: eastus
name: olfactory-transformer
properties:
  containers:
  - name: olfactory-transformer
    properties:
      image: youracr.azurecr.io/olfactory-transformer:latest
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: MODEL_PATH
        value: /models/olfactory-base-v1
      - name: AZURE_STORAGE_ACCOUNT
        value: your-storage-account
      resources:
        requests:
          memoryInGB: 4
          cpu: 2
        limits:
          memoryInGB: 8
          cpu: 4
      volumeMounts:
      - name: model-volume
        mountPath: /models
        readOnly: true
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: olfactory-transformer
  volumes:
  - name: model-volume
    azureFile:
      shareName: models
      storageAccountName: your-storage-account
      storageAccountKey: your-storage-key
      readOnly: true
type: Microsoft.ContainerInstance/containerGroups
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core Configuration
MODEL_PATH=/models/olfactory-base-v1
TOKENIZER_PATH=/models/olfactory-base-v1
DEVICE=cuda  # or cpu, auto

# Cache Configuration
CACHE_DIR=/cache
CACHE_SIZE_MB=1024
CACHE_TTL_HOURS=24
REDIS_URL=redis://localhost:6379

# Performance Configuration
BATCH_SIZE=32
MAX_WORKERS=4
INFERENCE_TIMEOUT=30

# Monitoring Configuration
LOG_LEVEL=INFO
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080

# Security Configuration
API_KEY_REQUIRED=true
RATE_LIMIT_PER_MINUTE=100
CORS_ORIGINS=https://yourdomain.com

# Sensor Configuration
SENSOR_PORT=/dev/ttyUSB0
SENSOR_BAUDRATE=9600
SENSOR_TIMEOUT=5
```

### Configuration Files
```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', '/models/olfactory-base-v1')
    DEVICE = os.getenv('DEVICE', 'auto')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_SIZE_MB = int(os.getenv('CACHE_SIZE_MB', '1024'))
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Performance settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    INFERENCE_TIMEOUT = int(os.getenv('INFERENCE_TIMEOUT', '30'))
    
    # Monitoring settings
    METRICS_ENABLED = True
    HEALTH_CHECK_ENABLED = True
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Security settings
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true'
    RATE_LIMIT_ENABLED = True
    CORS_ENABLED = True
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
REQUEST_COUNT = Counter('olfactory_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('olfactory_request_duration_seconds', 'Request duration')
REQUEST_ERRORS = Counter('olfactory_errors_total', 'Total errors', ['error_type'])

# Model metrics
INFERENCE_DURATION = Histogram('olfactory_inference_duration_seconds', 'Inference duration')
BATCH_SIZE = Histogram('olfactory_batch_size', 'Batch size distribution')
MODEL_ACCURACY = Gauge('olfactory_model_accuracy', 'Model accuracy')

# System metrics
MEMORY_USAGE = Gauge('olfactory_memory_usage_bytes', 'Memory usage')
GPU_UTILIZATION = Gauge('olfactory_gpu_utilization_percent', 'GPU utilization')
CACHE_HIT_RATE = Gauge('olfactory_cache_hit_rate', 'Cache hit rate')

# Sensor metrics
SENSOR_READINGS = Counter('olfactory_sensor_readings_total', 'Sensor readings', ['sensor_type'])
SENSOR_ERRORS = Counter('olfactory_sensor_errors_total', 'Sensor errors', ['sensor_type', 'error'])
CALIBRATION_STATUS = Gauge('olfactory_sensor_calibration_status', 'Sensor calibration status', ['sensor_id'])
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Olfactory Transformer - Production",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(olfactory_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(olfactory_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(olfactory_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "singlestat",
        "targets": [
          {
            "expr": "olfactory_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "olfactory_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory (GB)"
          },
          {
            "expr": "olfactory_gpu_utilization_percent",
            "legendFormat": "GPU Utilization %"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```python
# logging.conf
[loggers]
keys=root,olfactory

[handlers]
keys=consoleHandler,fileHandler,jsonHandler

[formatters]
keys=simpleFormatter,jsonFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_olfactory]
level=INFO
handlers=consoleHandler,fileHandler,jsonHandler
qualname=olfactory_transformer
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.RotatingFileHandler
level=INFO
formatter=simpleFormatter
args=('/var/log/olfactory/app.log', 'a', 10485760, 5)

[handler_jsonHandler]
class=StreamHandler
level=WARNING
formatter=jsonFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_jsonFormatter]
format={"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}
```

## 🚀 Performance Optimization

### Model Optimization
```python
# deployment/optimize.py
import torch
from olfactory_transformer import OlfactoryTransformer
from olfactory_transformer.utils.optimization import ModelOptimizer

def optimize_for_production(model_path: str, output_path: str):
    """Optimize model for production deployment."""
    
    # Load model
    model = OlfactoryTransformer.from_pretrained(model_path)
    model.eval()
    
    optimizer = ModelOptimizer(model)
    
    # Apply optimizations
    print("Applying quantization...")
    quantized_model = optimizer.quantize_model(method="dynamic")
    
    print("Exporting to TorchScript...")
    optimizer.export_torchscript(
        f"{output_path}/model.pt",
        method="trace"
    )
    
    print("Exporting to ONNX...")
    optimizer.export_onnx(
        f"{output_path}/model.onnx",
        optimize_for_mobile=True
    )
    
    # Benchmark performance
    print("Benchmarking optimizations...")
    results = optimizer.compare_optimizations()
    
    for opt_name, metrics in results.items():
        print(f"{opt_name}:")
        print(f"  Avg inference time: {metrics['avg_inference_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/s")
        print(f"  Latency per sample: {metrics['latency_per_sample_ms']:.1f}ms")
    
    return results

if __name__ == "__main__":
    optimize_for_production("./models/olfactory-base-v1", "./optimized")
```

### Caching Strategy
```python
# deployment/cache_config.py
from olfactory_transformer.utils.caching import ModelCache, PredictionCache

def setup_production_cache():
    """Setup production caching configuration."""
    
    # Multi-tier caching
    model_cache = ModelCache(
        cache_dir="/cache/model",
        memory_cache_size=2000,
        disk_cache_size_gb=10
    )
    
    prediction_cache = PredictionCache(model_cache)
    
    # Cache warming
    common_molecules = [
        "CCO", "CC(C)O", "C1=CC=CC=C1", "CC(=O)OCC",
        "COC1=CC(=CC=C1O)C=O", "CC(C)CC1=CC=C(C=C1)C(C)C"
    ]
    
    def warm_predict(smiles):
        # Mock prediction for warming
        from olfactory_transformer.core.config import ScentPrediction
        return ScentPrediction(primary_notes=["test"], intensity=5.0)
    
    print("Warming cache with common molecules...")
    prediction_cache.get_or_predict(common_molecules, warm_predict)
    
    # Setup cache cleanup
    import threading
    import time
    
    def cache_cleanup_worker():
        while True:
            time.sleep(3600)  # Run every hour
            model_cache.cleanup_old_cache(max_age_days=7)
    
    cleanup_thread = threading.Thread(target=cache_cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    return model_cache, prediction_cache
```

## 🔒 Security Configuration

### API Security
```python
# security/auth.py
import jwt
from functools import wraps
from flask import request, jsonify, current_app

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
            
        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated

def validate_api_key(api_key: str) -> bool:
    """Validate API key against database or environment."""
    valid_keys = current_app.config.get('VALID_API_KEYS', [])
    return api_key in valid_keys

def rate_limit(requests_per_minute: int = 100):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Implement rate limiting logic
            client_ip = request.remote_addr
            
            # Check rate limit for this IP
            if exceeds_rate_limit(client_ip, requests_per_minute):
                return jsonify({'error': 'Rate limit exceeded'}), 429
                
            return f(*args, **kwargs)
        return decorated
    return decorator
```

### Input Validation
```python
# security/validation.py
import re
from typing import Optional

def validate_smiles(smiles: str) -> tuple[bool, Optional[str]]:
    """Validate SMILES string for security and correctness."""
    
    if not smiles:
        return False, "SMILES string cannot be empty"
    
    if len(smiles) > 1000:
        return False, "SMILES string too long"
    
    # Check for potentially dangerous characters
    dangerous_chars = ['<', '>', '&', '"', "'", '`']
    if any(char in smiles for char in dangerous_chars):
        return False, "SMILES contains invalid characters"
    
    # Basic SMILES pattern validation
    smiles_pattern = r'^[A-Za-z0-9@+\-\[\]()=:#\/\\\.%]+$'
    if not re.match(smiles_pattern, smiles):
        return False, "Invalid SMILES format"
    
    return True, None

def sanitize_input(data: dict) -> dict:
    """Sanitize input data."""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            value = re.sub(r'[<>&"\'`]', '', value)
            # Limit string length
            value = value[:1000]
        
        sanitized[key] = value
    
    return sanitized
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Model artifacts available and validated
- [ ] Dependencies installed and compatible
- [ ] Configuration files prepared
- [ ] Security settings configured
- [ ] Monitoring setup completed
- [ ] Load testing performed
- [ ] Backup procedures established

### Deployment
- [ ] Infrastructure provisioned
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Monitoring active
- [ ] Logging configured
- [ ] Cache warmed up
- [ ] Security scans completed

### Post-deployment
- [ ] Performance metrics baseline established
- [ ] Alerts configured and tested
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Rollback procedures tested
- [ ] Capacity planning reviewed

## 🔄 Maintenance & Updates

### Rolling Updates
```bash
#!/bin/bash
# rolling_update.sh

# Zero-downtime deployment script
NEW_VERSION=$1
NAMESPACE=${2:-default}

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version> [namespace]"
    exit 1
fi

echo "Starting rolling update to version $NEW_VERSION"

# Update deployment
kubectl set image deployment/olfactory-transformer \
    olfactory-transformer=olfactory-transformer:$NEW_VERSION \
    --namespace=$NAMESPACE

# Wait for rollout to complete
kubectl rollout status deployment/olfactory-transformer --namespace=$NAMESPACE

# Verify deployment
kubectl get pods --namespace=$NAMESPACE -l app=olfactory-transformer

echo "Rolling update completed successfully"
```

### Health Monitoring
```python
# health/monitor.py
import requests
import time
import logging
from typing import Dict, Any

class HealthMonitor:
    def __init__(self, endpoints: list[str], check_interval: int = 60):
        self.endpoints = endpoints
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
    
    def check_health(self, endpoint: str) -> Dict[str, Any]:
        """Check health of a single endpoint."""
        try:
            response = requests.get(f"{endpoint}/health", timeout=10)
            return {
                "endpoint": endpoint,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": "error",
                "error": str(e)
            }
    
    def monitor_continuously(self):
        """Continuously monitor all endpoints."""
        while True:
            for endpoint in self.endpoints:
                health = self.check_health(endpoint)
                
                if health["status"] != "healthy":
                    self.logger.error(f"Health check failed: {health}")
                    # Send alert (implement your alerting logic)
                else:
                    self.logger.info(f"Health check passed: {endpoint}")
            
            time.sleep(self.check_interval)

if __name__ == "__main__":
    endpoints = [
        "http://olfactory-1:8000",
        "http://olfactory-2:8000",
        "http://olfactory-3:8000"
    ]
    
    monitor = HealthMonitor(endpoints)
    monitor.monitor_continuously()
```

This deployment guide provides comprehensive coverage of production deployment scenarios, from simple single-server setups to complex distributed systems across major cloud platforms. The key is to start simple and scale up based on your specific requirements and traffic patterns.