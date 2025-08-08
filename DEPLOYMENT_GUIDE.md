# 🚀 Olfactory Transformer - Production Deployment Guide

## 🌟 System Overview

The **Olfactory Transformer Foundation Model** is now production-ready with comprehensive scaling, security, and monitoring capabilities. This deployment guide covers all aspects from development to enterprise-scale production deployment.

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                       │
│                  SSL/TLS + Rate Limiting                       │
└─────────────────┬───────────────────┬───────────────────────────┘
                  │                   │
            ┌─────▼─────┐         ┌─────▼─────┐
            │ API Node 1│         │ API Node 2│
            │ (Docker)  │   ...   │ (Docker)  │
            └─────┬─────┘         └─────┬─────┘
                  │                     │
            ┌─────▼─────────────────────▼─────┐
            │        Redis Cache              │
            │     (Distributed Cache)         │
            └─────────────┬───────────────────┘
                          │
            ┌─────────────▼───────────────────┐
            │      Monitoring Stack          │
            │ Prometheus + Grafana + Loki    │
            └─────────────────────────────────┘
```

## 🏗️ Deployment Options

### Option 1: Single Node Development
```bash
# Quick start for development
docker compose up -d
```

### Option 2: Production Scaling
```bash
# Auto-scaling production deployment
docker compose -f docker-compose.scale.yml up -d
```

### Option 3: Kubernetes (Enterprise)
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/
```

## 🔧 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ free space
- **Network**: 1Gbps+ bandwidth

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+ (for local development)
- Nginx (for custom load balancing)

## 🚀 Quick Start Guide

### 1. Clone and Setup
```bash
git clone <repository-url>
cd olfactory-transformer
cp .env.example .env  # Configure environment variables
```

### 2. Environment Configuration
```bash
# .env file configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
REDIS_HOST=redis
REDIS_PORT=6379
MAX_BATCH_SIZE=32
WORKER_THREADS=4
ENABLE_CACHING=true
GRAFANA_PASSWORD=secure_password_here
SMTP_HOST=smtp.example.com
SMTP_USER=alerts@example.com
SMTP_PASSWORD=smtp_password
```

### 3. Production Deployment
```bash
# Start auto-scaling production stack
docker compose -f docker-compose.scale.yml up -d

# Verify deployment
curl http://localhost/health
```

## 📈 Scaling Configuration

### Auto-Scaling Parameters
```yaml
# Auto-scaling thresholds
MIN_INSTANCES=2
MAX_INSTANCES=8
SCALE_UP_THRESHOLD=80    # CPU/Memory %
SCALE_DOWN_THRESHOLD=30  # CPU/Memory %
COOLDOWN_PERIOD=300      # seconds
```

### Manual Scaling
```bash
# Scale API instances manually
docker compose -f docker-compose.scale.yml up -d --scale olfactory-api=5

# Monitor scaling status
docker compose -f docker-compose.scale.yml ps
```

## 🛡️ Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/olfactory.key \
  -out nginx/ssl/olfactory.crt
```

### Security Features
- ✅ **Input Validation**: Comprehensive SMILES and data validation
- ✅ **Rate Limiting**: API endpoint protection
- ✅ **Container Security**: Non-root users, read-only filesystems
- ✅ **Network Security**: Internal network isolation
- ✅ **Secret Management**: Environment-based configuration
- ✅ **Security Monitoring**: Real-time threat detection

## 📊 Monitoring and Observability

### Access Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686

### Key Metrics Monitored
- Request throughput and latency
- Model inference performance
- System resource utilization
- Error rates and patterns
- Cache hit/miss ratios
- Security violations

### Alert Configuration
```yaml
# Grafana alerting rules
alerts:
  - name: High CPU Usage
    condition: cpu_usage > 85%
    for: 5m
    actions: [email, slack]
  
  - name: Model Inference Errors
    condition: error_rate > 5%
    for: 2m
    actions: [email, pagerduty]
```

## 🔍 Performance Optimization

### Cache Configuration
```python
# Redis cache settings
REDIS_MAXMEMORY=2gb
REDIS_POLICY=allkeys-lru
CACHE_TTL=3600  # 1 hour default
```

### Model Optimization
```python
# Inference optimization
MAX_BATCH_SIZE=32
ENABLE_QUANTIZATION=true
USE_COMPILED_MODEL=true
WORKER_THREADS=4
```

### Database Tuning
```bash
# Redis performance tuning
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
sysctl -p
```

## 🧪 Testing and Validation

### Health Checks
```bash
# System health validation
curl http://localhost/health
curl http://localhost/metrics

# API functionality test
curl -X POST http://localhost/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost
```

### Performance Benchmarks
```bash
# Run benchmark suite
python benchmark.py --concurrent=10 --requests=1000
```

## 🔄 CI/CD Integration

### GitHub Actions Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to production
      run: |
        docker compose -f docker-compose.scale.yml up -d
        ./scripts/health-check.sh
```

### Deployment Automation
```bash
# Automated deployment script
./scripts/deploy.sh production

# Blue-green deployment
./scripts/blue-green-deploy.sh
```

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats
# Adjust cache settings
export REDIS_MAXMEMORY=1gb
```

#### Slow Response Times
```bash
# Check bottlenecks
curl http://localhost/metrics | grep response_time
# Scale up instances
docker compose -f docker-compose.scale.yml up -d --scale olfactory-api=6
```

#### Connection Errors
```bash
# Check service health
docker compose -f docker-compose.scale.yml ps
# Check logs
docker compose -f docker-compose.scale.yml logs olfactory-api-1
```

### Log Analysis
```bash
# View aggregated logs
docker compose -f docker-compose.scale.yml logs -f --tail=100

# Query specific errors
grep "ERROR" logs/olfactory_transformer.log | tail -20
```

## 📈 Capacity Planning

### Scaling Guidelines
- **Light Load**: 2 API instances, 2GB Redis
- **Medium Load**: 4-6 API instances, 4GB Redis  
- **Heavy Load**: 8+ API instances, 8GB+ Redis
- **Enterprise**: Kubernetes cluster with HPA

### Resource Estimates
```
Per API Instance:
- CPU: 2 cores
- Memory: 4GB
- Storage: 10GB

Supporting Services:
- Redis: 2GB memory, 1 core
- Monitoring: 1GB memory, 0.5 cores
- Load Balancer: 512MB memory, 0.2 cores
```

## 🌍 Multi-Region Deployment

### Global Load Balancing
```yaml
# AWS Route 53 health checks
health_checks:
  - region: us-east-1
    endpoint: https://api-us.olfactory.com/health
  - region: eu-west-1  
    endpoint: https://api-eu.olfactory.com/health
```

### Data Replication
```bash
# Redis cluster setup for global replication
redis-cli --cluster create \
  redis-us:6379 redis-eu:6379 redis-asia:6379
```

## 🔐 Security Hardening

### Production Security Checklist
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules implemented  
- [ ] Container security scanning enabled
- [ ] Secret rotation automated
- [ ] Audit logging configured
- [ ] Network segmentation implemented
- [ ] Vulnerability scanning automated

### Compliance Features
- **GDPR**: Data encryption and deletion capabilities
- **SOC 2**: Audit logging and access controls
- **HIPAA**: Data encryption and access logging
- **ISO 27001**: Security monitoring and incident response

## 📞 Support and Maintenance

### Backup Strategy
```bash
# Automated backup script
./scripts/backup.sh
# - Model checkpoints
# - Configuration files  
# - Cache snapshots
# - Monitoring data
```

### Update Procedure
```bash
# Rolling update with zero downtime
./scripts/rolling-update.sh v2.0.0
```

### Emergency Procedures
```bash
# Emergency scaling
./scripts/emergency-scale.sh 20

# Service recovery
./scripts/disaster-recovery.sh
```

## 📋 Appendices

### Environment Variables Reference
```bash
# Core Configuration
PYTHONPATH=/app
LOG_LEVEL=INFO
ENVIRONMENT=production

# Model Configuration  
MODEL_CACHE_DIR=/app/models
MAX_BATCH_SIZE=32
INFERENCE_TIMEOUT=30
ENABLE_QUANTIZATION=true

# Cache Configuration
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=3600
ENABLE_CACHING=true

# Security Configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
JWT_SECRET_KEY=your-secret-key

# Monitoring Configuration
PROMETHEUS_PORT=8001
ENABLE_METRICS=true
TRACE_SAMPLING_RATE=0.1
```

### API Reference
```bash
# Health Check
GET /health

# Model Prediction
POST /api/v1/predict
{
  "smiles": "CCO",
  "options": {
    "detailed": true,
    "confidence_threshold": 0.8
  }
}

# Batch Prediction
POST /api/v1/predict/batch
{
  "molecules": ["CCO", "CC(C)O"],
  "options": {"format": "json"}
}

# Metrics
GET /metrics
```

---

**🚀 Your Olfactory Transformer system is now production-ready with enterprise-grade scaling, security, and monitoring capabilities!**

For additional support or advanced configurations, please refer to the detailed documentation in the `/docs` directory.