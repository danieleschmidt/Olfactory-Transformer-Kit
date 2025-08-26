#!/bin/bash
set -e

echo "🚀 Deploying olfactory-transformer v1.0.0"

# Build image
echo "Building Docker image..."
docker build -t olfactory-transformer:1.0.0 .

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for deployment
echo "Waiting for deployment..."
kubectl rollout status deployment/olfactory-transformer --timeout=300s

echo "✅ Deployment completed"
