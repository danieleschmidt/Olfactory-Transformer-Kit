# olfactory-transformer - Deployment Guide

## Quick Start

1. Build and deploy:
   ```bash
   ./scripts/deploy.sh
   ```

2. Verify deployment:
   ```bash
   kubectl get pods -l app=olfactory-transformer
   kubectl get service olfactory-transformer-service
   ```

## Configuration

- **Environment**: production
- **Replicas**: 3
- **CPU Limit**: 2000m
- **Memory Limit**: 4Gi
- **Port**: 8000

## Monitoring

Access application at: http://localhost:8000

*Generated: 2025-08-26 13:41:34*
