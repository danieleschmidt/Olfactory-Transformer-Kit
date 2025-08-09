# Olfactory Transformer - Production Deployment Guide

This guide covers deploying the Olfactory Transformer in production environments with full scalability, reliability, and monitoring.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Docker (optional)
- Kubernetes (for auto-scaling)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd olfactory-transformer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install optional dependencies for enhanced features
pip install rdkit torch-geometric tensorrt onnxruntime redis
```

### Basic Usage

```python
from olfactory_transformer import OlfactoryTransformer, MoleculeTokenizer
from olfactory_transformer.core.config import OlfactoryConfig

# Initialize model
config = OlfactoryConfig(vocab_size=1000, hidden_size=256)
model = OlfactoryTransformer(config)
tokenizer = MoleculeTokenizer(vocab_size=1000)

# Build vocabulary
sample_molecules = ["CCO", "CC(C)O", "C1=CC=CC=C1", "COC1=CC(=CC=C1O)C=O"]
tokenizer.build_vocab_from_smiles(sample_molecules)

# Make prediction
prediction = model.predict_scent("CCO", tokenizer)
print(f"Primary notes: {prediction.primary_notes}")
print(f"Intensity: {prediction.intensity}/10")
print(f"Confidence: {prediction.confidence:.2%}")
```

## 🏗️ Architecture Overview

### Core Components

1. **OlfactoryTransformer**: Main model for scent prediction
2. **MoleculeTokenizer**: SMILES string tokenization and molecular features
3. **Reliability Layer**: Circuit breakers, retries, health checks
4. **Observability**: Structured logging, metrics, distributed tracing
5. **Auto-scaling**: Dynamic instance management and load balancing
6. **Security**: Input validation, rate limiting, secure configuration

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Gateway    │────│  Auto Scaler    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Instances                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Instance 1 │  │  Instance 2 │  │  Instance N │             │
│  │             │  │             │  │             │             │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │             │
│  │ │ Model   │ │  │ │ Model   │ │  │ │ Model   │ │             │
│  │ │         │ │  │ │         │ │  │ │         │ │             │
│  │ │Tokenizer│ │  │ │Tokenizer│ │  │ │Tokenizer│ │             │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Support Services                            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Redis     │  │ Prometheus  │  │   Grafana   │             │
│  │   Cache     │  │  Metrics    │  │ Dashboards  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Health      │  │ Security    │  │   Logging   │             │
│  │ Monitoring  │  │ Manager     │  │ Aggregation │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from olfactory_transformer.utils.reliability import reliability_manager; \
                   status = reliability_manager.health_checker.get_system_health(); \
                   exit(0 if status['healthy'] else 1)"

# Run the application
EXPOSE 8000
CMD ["python", "-m", "olfactory_transformer.api.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  olfactory-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
      - MODEL_CONFIG=production
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

## ☸️ Kubernetes Deployment

### Application Deployment

```yaml
# k8s/deployment.yaml
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
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: MODEL_CONFIG
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: olfactory-transformer-service
spec:
  selector:
    app: olfactory-transformer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: olfactory-transformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: olfactory-transformer
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'olfactory-transformer'
    static_configs:
      - targets: ['olfactory-api:8000']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
```

### Grafana Dashboard

Key metrics to monitor:

1. **Performance Metrics**:
   - Request latency (P50, P95, P99)
   - Throughput (requests/second)
   - Error rate
   - Queue depth

2. **System Metrics**:
   - CPU usage
   - Memory usage
   - GPU utilization (if available)
   - Disk I/O

3. **Application Metrics**:
   - Circuit breaker states
   - Cache hit rates
   - Prediction accuracy
   - Model inference time

4. **Business Metrics**:
   - Active users
   - Popular molecules
   - Prediction categories

## 🔒 Security Configuration

### Environment Variables

```bash
# Required
OLFACTORY_SECRET_KEY=your-secret-key-here
REDIS_PASSWORD=your-redis-password

# Optional security settings
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
MAX_REQUEST_SIZE=1MB
ALLOWED_ORIGINS=https://yourdomain.com

# SSL/TLS
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Security Best Practices

1. **Input Validation**: All SMILES strings are validated
2. **Rate Limiting**: Configurable per-IP rate limits
3. **CORS**: Configurable allowed origins
4. **SSL/TLS**: Support for HTTPS termination
5. **Secret Management**: Environment-based secrets
6. **Security Logging**: All security events logged

## ⚡ Performance Optimization

### Model Optimization

```python
from olfactory_transformer.utils.optimization import ModelOptimizer

# Quantize model for faster inference
optimizer = ModelOptimizer(model)
quantized_model = optimizer.quantize_model(method="dynamic")

# Export to TorchScript
optimizer.export_torchscript("model.pt", method="trace")

# Convert to ONNX (if supported)
optimizer.export_onnx("model.onnx")
```

### Caching Configuration

```python
from olfactory_transformer.utils.caching import ModelCache

# Configure caching
cache = ModelCache(
    backend="redis",
    redis_url="redis://localhost:6379",
    max_memory_mb=500,
    default_ttl=3600  # 1 hour
)

# Enable prediction caching
model.enable_caching(cache)
```

### Auto-scaling Configuration

```python
from olfactory_transformer.utils.autoscaling import create_autoscaler

# Create auto-scaler
autoscaler = create_autoscaler(
    min_instances=2,
    max_instances=10,
    model_config={
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_hidden_layers": 8,
    }
)

# Start auto-scaling
autoscaler.start()

# Make predictions through auto-scaler
prediction = await autoscaler.predict("CCO")
```

## 🔧 Configuration Management

### Production Configuration

```python
# config/production.py
from olfactory_transformer.core.config import OlfactoryConfig

PRODUCTION_CONFIG = OlfactoryConfig(
    vocab_size=5000,
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=16,
    molecular_features=256,
    max_position_embeddings=1024,
    
    # Performance optimizations
    dropout=0.05,  # Reduced for inference
    
    # Production settings
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

### Environment-specific Overrides

```python
import os
from olfactory_transformer.core.config import OlfactoryConfig

def get_config():
    env = os.getenv("ENVIRONMENT", "development")
    
    base_config = {
        "vocab_size": int(os.getenv("VOCAB_SIZE", "1000")),
        "hidden_size": int(os.getenv("HIDDEN_SIZE", "256")),
        "num_hidden_layers": int(os.getenv("NUM_LAYERS", "8")),
    }
    
    if env == "production":
        base_config.update({
            "hidden_size": 512,
            "num_hidden_layers": 12,
        })
    
    return OlfactoryConfig(**base_config)
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Reduce batch size
   - Enable model quantization
   - Use CPU fallback

2. **Performance Issues**:
   - Enable caching
   - Use TorchScript
   - Optimize model size

3. **Connectivity Issues**:
   - Check Redis connection
   - Verify network policies
   - Check health endpoints

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed error reporting
from olfactory_transformer.utils.reliability import reliability_manager
status = reliability_manager.get_system_status()
print(json.dumps(status, indent=2))
```

### Health Check Endpoints

- `GET /health` - Basic health check
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /status` - Detailed system status

## 📈 Scaling Guidelines

### Vertical Scaling (Single Instance)

- **Small**: 1 CPU, 2GB RAM → ~2 RPS
- **Medium**: 2 CPU, 4GB RAM → ~5 RPS  
- **Large**: 4 CPU, 8GB RAM → ~10 RPS
- **XL**: 8 CPU, 16GB RAM → ~20 RPS

### Horizontal Scaling (Multiple Instances)

- Use auto-scaling based on CPU/memory/latency
- Target 70% CPU utilization
- Scale up fast, scale down slowly
- Minimum 2 instances for availability

### GPU Acceleration

```python
# Enable GPU if available
config = OlfactoryConfig(device="cuda")
model = OlfactoryTransformer(config)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

## 📚 API Documentation

### REST API

```bash
# Predict scent from SMILES
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"smiles_list": ["CCO", "CC(C)O", "CCC"]}'

# Get system status
curl "http://localhost:8000/status"
```

### Response Format

```json
{
  "success": true,
  "prediction": {
    "primary_notes": ["woody", "fresh", "green"],
    "intensity": 6.5,
    "confidence": 0.87,
    "chemical_family": "alcohols",
    "similar_perfumes": ["Reference A", "Reference B"]
  },
  "metadata": {
    "model_version": "1.0.0",
    "inference_time_ms": 45.2,
    "instance_id": "instance_abc123"
  }
}
```

## 🔄 Continuous Integration/Deployment

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest
    - name: Run tests
      run: pytest
    - name: Run benchmarks
      run: python benchmark_performance.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout status deployment/olfactory-transformer
```

## 🎯 Production Checklist

Before deploying to production:

- [ ] All tests passing
- [ ] Performance benchmarks acceptable
- [ ] Security scan passed
- [ ] Monitoring configured
- [ ] Alerting rules set up
- [ ] Backup and recovery tested
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

## 📞 Support

For production support:

1. Check system status: `/status` endpoint
2. Review logs and metrics
3. Consult troubleshooting guide
4. Contact development team

---

*This deployment guide ensures your Olfactory Transformer runs reliably in production with full observability, security, and scalability.*