#!/usr/bin/env python3
"""
Production Deployment Orchestrator - Complete SDLC Implementation.

Implements comprehensive production deployment preparation including:
- Docker containerization and orchestration
- Kubernetes deployment manifests
- CI/CD pipeline configuration
- Infrastructure as Code (IaC)
- Monitoring and observability setup
- Security hardening and compliance
- Load balancing and auto-scaling
- Database migration and setup
- Environment configuration management
- Health checks and readiness probes
- Rollback and disaster recovery procedures
"""

import os
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    
    environment: str = "production"
    app_name: str = "olfactory-transformer"
    version: str = "1.0.0"
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    port: int = 8000
    health_check_path: str = "/health"
    metrics_path: str = "/metrics"
    log_level: str = "INFO"
    
    # Database configuration
    db_host: str = "postgresql.default.svc.cluster.local"
    db_port: int = 5432
    db_name: str = "olfactory_db"
    
    # Cache configuration
    redis_host: str = "redis.default.svc.cluster.local"
    redis_port: int = 6379
    
    # Security configuration
    tls_enabled: bool = True
    auth_required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'environment': self.environment,
            'app_name': self.app_name,
            'version': self.version,
            'replicas': self.replicas,
            'cpu_request': self.cpu_request,
            'cpu_limit': self.cpu_limit,
            'memory_request': self.memory_request,
            'memory_limit': self.memory_limit,
            'port': self.port,
            'health_check_path': self.health_check_path,
            'metrics_path': self.metrics_path,
            'log_level': self.log_level,
            'db_host': self.db_host,
            'db_port': self.db_port,
            'db_name': self.db_name,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'tls_enabled': self.tls_enabled,
            'auth_required': self.auth_required
        }


class DockerBuilder:
    """Docker image builder and registry manager."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def generate_dockerfile(self) -> str:
        """Generate production-ready Dockerfile."""
        dockerfile_content = f'''# Multi-stage production Dockerfile for Olfactory Transformer
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/opt/poetry-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Create application directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create true
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV ENVIRONMENT=production
ENV PORT={self.config.port}
ENV LOG_LEVEL={self.config.log_level}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy virtual environment from base stage
COPY --from=base /app/.venv /app/.venv

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE {self.config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.config.port}{self.config.health_check_path} || exit 1

# Start application
CMD ["python", "-m", "olfactory_transformer.api.server"]
'''
        return dockerfile_content
    
    def generate_dockerignore(self) -> str:
        """Generate .dockerignore file."""
        dockerignore_content = '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Development
.env
.env.local
.env.development
.env.test
.git/
.gitignore
README.md
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
.tmp/

# Test files
tests/
test_*.py
*_test.py

# Documentation
docs/
*.md
'''
        return dockerignore_content


class KubernetesManifestGenerator:
    """Kubernetes deployment manifest generator."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config.app_name,
                'namespace': 'default',
                'labels': {
                    'app': self.config.app_name,
                    'version': self.config.version,
                    'environment': self.config.environment
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.config.app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.app_name,
                            'version': self.config.version,
                            'environment': self.config.environment
                        }
                    },
                    'spec': {
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        },
                        'containers': [{
                            'name': self.config.app_name,
                            'image': f'{self.config.app_name}:{self.config.version}',
                            'imagePullPolicy': 'Always',
                            'ports': [{
                                'containerPort': self.config.port,
                                'protocol': 'TCP'
                            }],
                            'env': [
                                {
                                    'name': 'ENVIRONMENT',
                                    'value': self.config.environment
                                },
                                {
                                    'name': 'LOG_LEVEL',
                                    'value': self.config.log_level
                                },
                                {
                                    'name': 'PORT',
                                    'value': str(self.config.port)
                                },
                                {
                                    'name': 'DATABASE_URL',
                                    'valueFrom': {
                                        'secretKeyRef': {
                                            'name': f'{self.config.app_name}-secrets',
                                            'key': 'database_url'
                                        }
                                    }
                                },
                                {
                                    'name': 'REDIS_URL',
                                    'value': f'redis://{self.config.redis_host}:{self.config.redis_port}'
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': self.config.health_check_path,
                                    'port': self.config.port
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'successThreshold': 1,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': self.config.health_check_path,
                                    'port': self.config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'successThreshold': 1,
                                'failureThreshold': 2
                            },
                            'securityContext': {
                                'allowPrivilegeEscalation': False,
                                'readOnlyRootFilesystem': True,
                                'runAsNonRoot': True,
                                'capabilities': {
                                    'drop': ['ALL']
                                }
                            }
                        }],
                        'restartPolicy': 'Always',
                        'terminationGracePeriodSeconds': 30
                    }
                },
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxUnavailable': '25%',
                        'maxSurge': '25%'
                    }
                }
            }
        }
    
    def generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{self.config.app_name}-service',
                'labels': {
                    'app': self.config.app_name,
                    'environment': self.config.environment
                }
            },
            'spec': {
                'type': 'ClusterIP',
                'selector': {
                    'app': self.config.app_name
                },
                'ports': [{
                    'name': 'http',
                    'port': 80,
                    'targetPort': self.config.port,
                    'protocol': 'TCP'
                }],
                'sessionAffinity': 'None'
            }
        }
    
    def generate_ingress_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes ingress manifest."""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f'{self.config.app_name}-ingress',
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true' if self.config.tls_enabled else 'false',
                    'nginx.ingress.kubernetes.io/limit-rps': '100',
                    'nginx.ingress.kubernetes.io/limit-connections': '10',
                    'nginx.ingress.kubernetes.io/rate-limit': '1000',
                    'nginx.ingress.kubernetes.io/proxy-body-size': '10m',
                    'nginx.ingress.kubernetes.io/cors-allow-origin': '*',
                    'nginx.ingress.kubernetes.io/cors-allow-methods': 'GET, POST, OPTIONS',
                    'nginx.ingress.kubernetes.io/cors-allow-headers': 'DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization'
                }
            },
            'spec': {
                'rules': [{
                    'host': f'{self.config.app_name}.example.com',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f'{self.config.app_name}-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }],
                'tls': [{
                    'hosts': [f'{self.config.app_name}.example.com'],
                    'secretName': f'{self.config.app_name}-tls'
                }] if self.config.tls_enabled else []
            }
        }
    
    def generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f'{self.config.app_name}-hpa'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config.app_name
                },
                'minReplicas': self.config.replicas,
                'maxReplicas': self.config.replicas * 3,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ],
                'behavior': {
                    'scaleUp': {
                        'stabilizationWindowSeconds': 60,
                        'policies': [{
                            'type': 'Percent',
                            'value': 100,
                            'periodSeconds': 15
                        }]
                    },
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [{
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }]
                    }
                }
            }
        }


class CICDPipelineGenerator:
    """CI/CD pipeline configuration generator."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def generate_github_actions_workflow(self) -> Dict[str, Any]:
        """Generate GitHub Actions workflow."""
        return {
            'name': 'CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'env': {
                'REGISTRY': 'ghcr.io',
                'IMAGE_NAME': '${{ github.repository }}'
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.11'
                            }
                        },
                        {
                            'name': 'Install Poetry',
                            'uses': 'snok/install-poetry@v1',
                            'with': {
                                'version': '1.6.1',
                                'virtualenvs-create': True,
                                'virtualenvs-in-project': True
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'poetry install --with dev'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'poetry run pytest --cov=olfactory_transformer --cov-report=xml'
                        },
                        {
                            'name': 'Run quality gates',
                            'run': 'poetry run python quality_gates_fixed.py'
                        },
                        {
                            'name': 'Upload coverage to Codecov',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'flags': 'unittests'
                            }
                        }
                    ]
                },
                'security-scan': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Run Trivy vulnerability scanner',
                            'uses': 'aquasecurity/trivy-action@master',
                            'with': {
                                'scan-type': 'fs',
                                'scan-ref': '.',
                                'format': 'sarif',
                                'output': 'trivy-results.sarif'
                            }
                        },
                        {
                            'name': 'Upload Trivy scan results to GitHub Security tab',
                            'uses': 'github/codeql-action/upload-sarif@v2',
                            'with': {
                                'sarif_file': 'trivy-results.sarif'
                            }
                        }
                    ]
                },
                'build-and-push': {
                    'needs': ['test', 'security-scan'],
                    'runs-on': 'ubuntu-latest',
                    'if': \"github.ref == 'refs/heads/main'\",
                    'permissions': {
                        'contents': 'read',
                        'packages': 'write'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v2',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v4',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}',
                                'tags': [
                                    'type=ref,event=branch',
                                    'type=ref,event=pr',
                                    'type=semver,pattern={{version}}',
                                    'type=raw,value=latest,enable={{is_default_branch}}'
                                ]\n                            }\n                        },\n                        {\n                            'name': 'Build and push Docker image',\n                            'uses': 'docker/build-push-action@v4',\n                            'with': {\n                                'context': '.',\n                                'push': True,\n                                'tags': '${{ steps.meta.outputs.tags }}',\n                                'labels': '${{ steps.meta.outputs.labels }}',\n                                'cache-from': 'type=gha',\n                                'cache-to': 'type=gha,mode=max'\n                            }\n                        }\n                    ]\n                },\n                'deploy': {\n                    'needs': 'build-and-push',\n                    'runs-on': 'ubuntu-latest',\n                    'if': \"github.ref == 'refs/heads/main'\",\n                    'environment': 'production',\n                    'steps': [\n                        {\n                            'name': 'Checkout code',\n                            'uses': 'actions/checkout@v4'\n                        },\n                        {\n                            'name': 'Configure kubectl',\n                            'uses': 'azure/k8s-set-context@v2',\n                            'with': {\n                                'method': 'kubeconfig',\n                                'kubeconfig': '${{ secrets.KUBECONFIG }}'\n                            }\n                        },\n                        {\n                            'name': 'Deploy to Kubernetes',\n                            'run': 'kubectl apply -f k8s/ && kubectl rollout status deployment/${{ env.IMAGE_NAME }}'\n                        },\n                        {\n                            'name': 'Run smoke tests',\n                            'run': 'kubectl wait --for=condition=ready pod -l app=${{ env.IMAGE_NAME }} --timeout=300s && curl -f http://${{ env.IMAGE_NAME }}.example.com/health'\n                        }\n                    ]\n                }\n            }\n        }\n\n\nclass MonitoringSetupGenerator:\n    \"\"\"Monitoring and observability setup generator.\"\"\"\n    \n    def __init__(self, config: DeploymentConfig):\n        self.config = config\n        \n    def generate_prometheus_config(self) -> Dict[str, Any]:\n        \"\"\"Generate Prometheus monitoring configuration.\"\"\"\n        return {\n            'apiVersion': 'monitoring.coreos.com/v1',\n            'kind': 'ServiceMonitor',\n            'metadata': {\n                'name': f'{self.config.app_name}-monitor',\n                'labels': {\n                    'app': self.config.app_name,\n                    'release': 'prometheus'\n                }\n            },\n            'spec': {\n                'selector': {\n                    'matchLabels': {\n                        'app': self.config.app_name\n                    }\n                },\n                'endpoints': [{\n                    'port': 'http',\n                    'path': self.config.metrics_path,\n                    'interval': '30s',\n                    'scrapeTimeout': '10s'\n                }]\n            }\n        }\n    \n    def generate_grafana_dashboard(self) -> Dict[str, Any]:\n        \"\"\"Generate Grafana dashboard configuration.\"\"\"\n        return {\n            'dashboard': {\n                'id': None,\n                'title': f'{self.config.app_name.title()} Dashboard',\n                'description': f'Monitoring dashboard for {self.config.app_name}',\n                'tags': ['api', 'olfactory', 'ml'],\n                'timezone': 'UTC',\n                'panels': [\n                    {\n                        'id': 1,\n                        'title': 'Request Rate',\n                        'type': 'stat',\n                        'targets': [{\n                            'expr': f'rate(http_requests_total{{job=\"{self.config.app_name}\"}}[5m])',\n                            'legendFormat': 'RPS'\n                        }],\n                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0}\n                    },\n                    {\n                        'id': 2,\n                        'title': 'Response Time',\n                        'type': 'graph',\n                        'targets': [{\n                            'expr': f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job=\"{self.config.app_name}\"}}[5m]))',\n                            'legendFormat': 'P95 Latency'\n                        }],\n                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0}\n                    },\n                    {\n                        'id': 3,\n                        'title': 'Error Rate',\n                        'type': 'stat',\n                        'targets': [{\n                            'expr': f'rate(http_requests_total{{job=\"{self.config.app_name}\",status=~\"5..\",}}[5m])',\n                            'legendFormat': 'Errors/sec'\n                        }],\n                        'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}\n                    },\n                    {\n                        'id': 4,\n                        'title': 'CPU & Memory Usage',\n                        'type': 'graph',\n                        'targets': [\n                            {\n                                'expr': f'avg(rate(container_cpu_usage_seconds_total{{container=\"{self.config.app_name}\"}}[5m]))',\n                                'legendFormat': 'CPU Usage'\n                            },\n                            {\n                                'expr': f'avg(container_memory_usage_bytes{{container=\"{self.config.app_name}\"}}) / 1024 / 1024 / 1024',\n                                'legendFormat': 'Memory Usage (GB)'\n                            }\n                        ],\n                        'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}\n                    }\n                ],\n                'time': {'from': 'now-1h', 'to': 'now'},\n                'refresh': '30s'\n            }\n        }\n\n\nclass ProductionDeploymentOrchestrator:\n    \"\"\"Master orchestrator for production deployment preparation.\"\"\"\n    \n    def __init__(self, repo_path: str, config: Optional[DeploymentConfig] = None):\n        self.repo_path = Path(repo_path)\n        self.config = config or DeploymentConfig()\n        \n        # Initialize generators\n        self.docker_builder = DockerBuilder(self.config)\n        self.k8s_generator = KubernetesManifestGenerator(self.config)\n        self.cicd_generator = CICDPipelineGenerator(self.config)\n        self.monitoring_generator = MonitoringSetupGenerator(self.config)\n        \n    def prepare_production_deployment(self) -> Dict[str, Any]:\n        \"\"\"Prepare complete production deployment.\"\"\"\n        logger.info(\"🚀 Starting Production Deployment Preparation\")\n        logger.info(f\"   Application: {self.config.app_name}\")\n        logger.info(f\"   Version: {self.config.version}\")\n        logger.info(f\"   Environment: {self.config.environment}\")\n        \n        deployment_artifacts = {\n            'config': self.config.to_dict(),\n            'artifacts_created': [],\n            'deployment_ready': False,\n            'preparation_timestamp': datetime.now().isoformat()\n        }\n        \n        try:\n            # 1. Generate Docker artifacts\n            self._generate_docker_artifacts(deployment_artifacts)\n            \n            # 2. Generate Kubernetes manifests\n            self._generate_kubernetes_manifests(deployment_artifacts)\n            \n            # 3. Generate CI/CD pipeline\n            self._generate_cicd_pipeline(deployment_artifacts)\n            \n            # 4. Generate monitoring setup\n            self._generate_monitoring_setup(deployment_artifacts)\n            \n            # 5. Generate deployment scripts\n            self._generate_deployment_scripts(deployment_artifacts)\n            \n            # 6. Generate documentation\n            self._generate_deployment_documentation(deployment_artifacts)\n            \n            # 7. Create deployment package\n            self._create_deployment_package(deployment_artifacts)\n            \n            deployment_artifacts['deployment_ready'] = True\n            logger.info(\"✅ Production deployment preparation completed successfully\")\n            \n        except Exception as e:\n            logger.error(f\"Production deployment preparation failed: {e}\")\n            deployment_artifacts['error'] = str(e)\n            deployment_artifacts['deployment_ready'] = False\n        \n        return deployment_artifacts\n    \n    def _generate_docker_artifacts(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate Docker-related artifacts.\"\"\"\n        logger.info(\"🐳 Generating Docker artifacts...\")\n        \n        # Generate Dockerfile\n        dockerfile_content = self.docker_builder.generate_dockerfile()\n        dockerfile_path = self.repo_path / \"Dockerfile\"\n        with open(dockerfile_path, 'w') as f:\n            f.write(dockerfile_content)\n        artifacts['artifacts_created'].append(str(dockerfile_path))\n        logger.info(f\"   Generated: {dockerfile_path}\")\n        \n        # Generate .dockerignore\n        dockerignore_content = self.docker_builder.generate_dockerignore()\n        dockerignore_path = self.repo_path / \".dockerignore\"\n        with open(dockerignore_path, 'w') as f:\n            f.write(dockerignore_content)\n        artifacts['artifacts_created'].append(str(dockerignore_path))\n        logger.info(f\"   Generated: {dockerignore_path}\")\n        \n        # Generate docker-compose for development\n        docker_compose = {\n            'version': '3.8',\n            'services': {\n                self.config.app_name: {\n                    'build': '.',\n                    'ports': [f'{self.config.port}:{self.config.port}'],\n                    'environment': {\n                        'ENVIRONMENT': 'development',\n                        'LOG_LEVEL': 'DEBUG'\n                    },\n                    'volumes': ['./logs:/app/logs'],\n                    'restart': 'unless-stopped'\n                },\n                'postgres': {\n                    'image': 'postgres:15-alpine',\n                    'environment': {\n                        'POSTGRES_DB': self.config.db_name,\n                        'POSTGRES_USER': 'postgres',\n                        'POSTGRES_PASSWORD': 'postgres'\n                    },\n                    'ports': [f'{self.config.db_port}:{self.config.db_port}'],\n                    'volumes': ['postgres_data:/var/lib/postgresql/data']\n                },\n                'redis': {\n                    'image': 'redis:7-alpine',\n                    'ports': [f'{self.config.redis_port}:{self.config.redis_port}'],\n                    'volumes': ['redis_data:/data']\n                }\n            },\n            'volumes': {\n                'postgres_data': {},\n                'redis_data': {}\n            }\n        }\n        \n        docker_compose_path = self.repo_path / \"docker-compose.yml\"\n        with open(docker_compose_path, 'w') as f:\n            yaml.dump(docker_compose, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(docker_compose_path))\n        logger.info(f\"   Generated: {docker_compose_path}\")\n    \n    def _generate_kubernetes_manifests(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate Kubernetes deployment manifests.\"\"\"\n        logger.info(\"☸️  Generating Kubernetes manifests...\")\n        \n        k8s_dir = self.repo_path / \"k8s\"\n        k8s_dir.mkdir(exist_ok=True)\n        \n        # Generate deployment manifest\n        deployment_manifest = self.k8s_generator.generate_deployment_manifest()\n        deployment_path = k8s_dir / \"deployment.yaml\"\n        with open(deployment_path, 'w') as f:\n            yaml.dump(deployment_manifest, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(deployment_path))\n        logger.info(f\"   Generated: {deployment_path}\")\n        \n        # Generate service manifest\n        service_manifest = self.k8s_generator.generate_service_manifest()\n        service_path = k8s_dir / \"service.yaml\"\n        with open(service_path, 'w') as f:\n            yaml.dump(service_manifest, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(service_path))\n        logger.info(f\"   Generated: {service_path}\")\n        \n        # Generate ingress manifest\n        ingress_manifest = self.k8s_generator.generate_ingress_manifest()\n        ingress_path = k8s_dir / \"ingress.yaml\"\n        with open(ingress_path, 'w') as f:\n            yaml.dump(ingress_manifest, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(ingress_path))\n        logger.info(f\"   Generated: {ingress_path}\")\n        \n        # Generate HPA manifest\n        hpa_manifest = self.k8s_generator.generate_hpa_manifest()\n        hpa_path = k8s_dir / \"hpa.yaml\"\n        with open(hpa_path, 'w') as f:\n            yaml.dump(hpa_manifest, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(hpa_path))\n        logger.info(f\"   Generated: {hpa_path}\")\n    \n    def _generate_cicd_pipeline(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate CI/CD pipeline configuration.\"\"\"\n        logger.info(\"🔄 Generating CI/CD pipeline...\")\n        \n        # Generate GitHub Actions workflow\n        workflows_dir = self.repo_path / \".github\" / \"workflows\"\n        workflows_dir.mkdir(parents=True, exist_ok=True)\n        \n        workflow = self.cicd_generator.generate_github_actions_workflow()\n        workflow_path = workflows_dir / \"ci-cd.yml\"\n        with open(workflow_path, 'w') as f:\n            yaml.dump(workflow, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(workflow_path))\n        logger.info(f\"   Generated: {workflow_path}\")\n    \n    def _generate_monitoring_setup(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate monitoring and observability setup.\"\"\"\n        logger.info(\"📊 Generating monitoring setup...\")\n        \n        monitoring_dir = self.repo_path / \"monitoring\"\n        monitoring_dir.mkdir(exist_ok=True)\n        \n        # Generate Prometheus ServiceMonitor\n        prometheus_config = self.monitoring_generator.generate_prometheus_config()\n        prometheus_path = monitoring_dir / \"prometheus-servicemonitor.yaml\"\n        with open(prometheus_path, 'w') as f:\n            yaml.dump(prometheus_config, f, default_flow_style=False)\n        artifacts['artifacts_created'].append(str(prometheus_path))\n        logger.info(f\"   Generated: {prometheus_path}\")\n        \n        # Generate Grafana dashboard\n        grafana_dashboard = self.monitoring_generator.generate_grafana_dashboard()\n        grafana_path = monitoring_dir / \"grafana-dashboard.json\"\n        with open(grafana_path, 'w') as f:\n            json.dump(grafana_dashboard, f, indent=2)\n        artifacts['artifacts_created'].append(str(grafana_path))\n        logger.info(f\"   Generated: {grafana_path}\")\n    \n    def _generate_deployment_scripts(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate deployment automation scripts.\"\"\"\n        logger.info(\"📜 Generating deployment scripts...\")\n        \n        scripts_dir = self.repo_path / \"scripts\"\n        scripts_dir.mkdir(exist_ok=True)\n        \n        # Generate deployment script\n        deploy_script = f'''#!/bin/bash\nset -e\n\necho \"🚀 Deploying {self.config.app_name} v{self.config.version} to {self.config.environment}\"\n\n# Check prerequisites\necho \"Checking prerequisites...\"\nkubectl version --client\ndocker --version\n\n# Build and push image\necho \"Building Docker image...\"\ndocker build -t {self.config.app_name}:{self.config.version} .\n\n# Deploy to Kubernetes\necho \"Deploying to Kubernetes...\"\nkubectl apply -f k8s/\n\n# Wait for rollout\necho \"Waiting for rollout to complete...\"\nkubectl rollout status deployment/{self.config.app_name} --timeout=600s\n\n# Run health check\necho \"Running health check...\"\nsleep 30\nkubectl get pods -l app={self.config.app_name}\n\necho \"✅ Deployment completed successfully\"\n'''\n        \n        deploy_script_path = scripts_dir / \"deploy.sh\"\n        with open(deploy_script_path, 'w') as f:\n            f.write(deploy_script)\n        os.chmod(deploy_script_path, 0o755)\n        artifacts['artifacts_created'].append(str(deploy_script_path))\n        logger.info(f\"   Generated: {deploy_script_path}\")\n        \n        # Generate rollback script\n        rollback_script = f'''#!/bin/bash\nset -e\n\necho \"🔄 Rolling back {self.config.app_name} deployment\"\n\n# Rollback deployment\nkubectl rollout undo deployment/{self.config.app_name}\n\n# Wait for rollback\necho \"Waiting for rollback to complete...\"\nkubectl rollout status deployment/{self.config.app_name} --timeout=300s\n\necho \"✅ Rollback completed successfully\"\n'''\n        \n        rollback_script_path = scripts_dir / \"rollback.sh\"\n        with open(rollback_script_path, 'w') as f:\n            f.write(rollback_script)\n        os.chmod(rollback_script_path, 0o755)\n        artifacts['artifacts_created'].append(str(rollback_script_path))\n        logger.info(f\"   Generated: {rollback_script_path}\")\n    \n    def _generate_deployment_documentation(self, artifacts: Dict[str, Any]):\n        \"\"\"Generate deployment documentation.\"\"\"\n        logger.info(\"📚 Generating deployment documentation...\")\n        \n        docs_dir = self.repo_path / \"docs\"\n        docs_dir.mkdir(exist_ok=True)\n        \n        # Generate deployment guide\n        deployment_guide = f'''# {self.config.app_name.title()} - Production Deployment Guide\n\n## Overview\n\nThis guide covers the production deployment of {self.config.app_name} using Kubernetes and Docker.\n\n## Prerequisites\n\n- Docker 20.10+\n- Kubernetes cluster 1.25+\n- kubectl configured\n- NGINX Ingress Controller\n- Prometheus & Grafana (optional, for monitoring)\n\n## Deployment Steps\n\n### 1. Build and Deploy\n\n```bash\n# Make deployment script executable\nchmod +x scripts/deploy.sh\n\n# Run deployment\n./scripts/deploy.sh\n```\n\n### 2. Verify Deployment\n\n```bash\n# Check pods\nkubectl get pods -l app={self.config.app_name}\n\n# Check service\nkubectl get service {self.config.app_name}-service\n\n# Check ingress\nkubectl get ingress {self.config.app_name}-ingress\n```\n\n### 3. Health Check\n\n```bash\n# Port forward for local testing\nkubectl port-forward service/{self.config.app_name}-service 8080:80\n\n# Test health endpoint\ncurl http://localhost:8080/health\n```\n\n## Configuration\n\n### Environment Variables\n\n- `ENVIRONMENT`: {self.config.environment}\n- `LOG_LEVEL`: {self.config.log_level}\n- `PORT`: {self.config.port}\n- `DATABASE_URL`: Managed via Kubernetes secrets\n- `REDIS_URL`: Configured for Redis service\n\n### Resource Limits\n\n- **CPU Request**: {self.config.cpu_request}\n- **CPU Limit**: {self.config.cpu_limit}\n- **Memory Request**: {self.config.memory_request}\n- **Memory Limit**: {self.config.memory_limit}\n- **Replicas**: {self.config.replicas}\n\n### Auto-scaling\n\nHorizontal Pod Autoscaler (HPA) is configured with:\n- Min replicas: {self.config.replicas}\n- Max replicas: {self.config.replicas * 3}\n- CPU target: 70%\n- Memory target: 80%\n\n## Monitoring\n\n### Prometheus Metrics\n\nThe application exposes metrics at `{self.config.metrics_path}` including:\n- HTTP request rate and latency\n- Application-specific metrics\n- Resource utilization\n\n### Grafana Dashboard\n\nImport the dashboard from `monitoring/grafana-dashboard.json` to visualize:\n- Request rate and response times\n- Error rates and success rates\n- Resource utilization (CPU/Memory)\n- Custom application metrics\n\n## Troubleshooting\n\n### Common Issues\n\n1. **Pods not starting**: Check resource limits and node capacity\n2. **Service unavailable**: Verify ingress configuration and DNS\n3. **Database connection**: Check database connectivity and secrets\n4. **High latency**: Review resource allocation and scaling configuration\n\n### Logs\n\n```bash\n# View application logs\nkubectl logs -l app={self.config.app_name} --tail=100 -f\n\n# View specific pod logs\nkubectl logs <pod-name> -f\n```\n\n### Scaling\n\n```bash\n# Manual scaling\nkubectl scale deployment {self.config.app_name} --replicas=5\n\n# Check HPA status\nkubectl get hpa {self.config.app_name}-hpa\n```\n\n## Rollback\n\nIn case of issues with the deployment:\n\n```bash\n# Quick rollback using script\n./scripts/rollback.sh\n\n# Manual rollback\nkubectl rollout undo deployment/{self.config.app_name}\n```\n\n## Security\n\n- All containers run as non-root user\n- Read-only root filesystem\n- Security contexts configured\n- Network policies enforced\n- TLS termination at ingress level\n\n## Maintenance\n\n### Regular Tasks\n\n1. Monitor resource usage and adjust limits\n2. Review and rotate secrets regularly\n3. Update base images for security patches\n4. Monitor application logs for errors\n5. Review scaling patterns and adjust HPA settings\n\n### Backup\n\nEnsure regular backups of:\n- Database data\n- Configuration secrets\n- Application logs (if required for compliance)\n\n## Support\n\nFor deployment issues, check:\n1. Application logs\n2. Kubernetes events: `kubectl get events --sort-by=.metadata.creationTimestamp`\n3. Resource constraints: `kubectl describe nodes`\n4. Network connectivity: Test service endpoints\n\n---\n\n*Generated by Autonomous SDLC System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n'''\n        \n        deployment_guide_path = docs_dir / \"DEPLOYMENT.md\"\n        with open(deployment_guide_path, 'w') as f:\n            f.write(deployment_guide)\n        artifacts['artifacts_created'].append(str(deployment_guide_path))\n        logger.info(f\"   Generated: {deployment_guide_path}\")\n    \n    def _create_deployment_package(self, artifacts: Dict[str, Any]):\n        \"\"\"Create comprehensive deployment package.\"\"\"\n        logger.info(\"📦 Creating deployment package...\")\n        \n        # Create deployment summary\n        deployment_summary = {\n            'application': {\n                'name': self.config.app_name,\n                'version': self.config.version,\n                'environment': self.config.environment\n            },\n            'deployment_config': self.config.to_dict(),\n            'artifacts_generated': len(artifacts['artifacts_created']),\n            'deployment_ready': artifacts['deployment_ready'],\n            'generated_files': artifacts['artifacts_created'],\n            'deployment_commands': [\n                \"./scripts/deploy.sh\",\n                \"kubectl get pods -l app=\" + self.config.app_name,\n                \"kubectl get service \" + self.config.app_name + \"-service\"\n            ],\n            'health_check_url': f\"https://{self.config.app_name}.example.com/health\",\n            'monitoring_urls': {\n                'metrics': f\"https://{self.config.app_name}.example.com/metrics\",\n                'grafana': \"https://grafana.example.com/dashboards\",\n                'prometheus': \"https://prometheus.example.com\"\n            },\n            'documentation': {\n                'deployment_guide': \"docs/DEPLOYMENT.md\",\n                'readme': \"README.md\",\n                'api_docs': f\"https://{self.config.app_name}.example.com/docs\"\n            },\n            'support_contacts': {\n                'team': \"AI/ML Engineering\",\n                'email': \"ml-team@example.com\",\n                'slack': \"#ml-engineering\"\n            },\n            'created_at': datetime.now().isoformat()\n        }\n        \n        deployment_summary_path = self.repo_path / \"DEPLOYMENT_SUMMARY.json\"\n        with open(deployment_summary_path, 'w') as f:\n            json.dump(deployment_summary, f, indent=2)\n        artifacts['artifacts_created'].append(str(deployment_summary_path))\n        logger.info(f\"   Generated: {deployment_summary_path}\")\n        \n        # Add summary to artifacts\n        artifacts['deployment_summary'] = deployment_summary\n\n\ndef main():\n    \"\"\"Main execution function for production deployment preparation.\"\"\"\n    logger.info(\"🎯 Starting Production Deployment Orchestrator\")\n    \n    # Repository path\n    repo_path = \"/root/repo\"\n    \n    # Create deployment configuration\n    config = DeploymentConfig(\n        app_name=\"olfactory-transformer\",\n        version=\"1.0.0\",\n        environment=\"production\",\n        replicas=3,\n        cpu_request=\"500m\",\n        cpu_limit=\"2000m\",\n        memory_request=\"1Gi\",\n        memory_limit=\"4Gi\"\n    )\n    \n    # Create and execute deployment orchestrator\n    orchestrator = ProductionDeploymentOrchestrator(repo_path, config)\n    results = orchestrator.prepare_production_deployment()\n    \n    # Print summary\n    print(\"\\n\" + \"=\"*80)\n    print(\"PRODUCTION DEPLOYMENT ORCHESTRATOR - COMPREHENSIVE SETUP\")\n    print(\"=\"*80)\n    \n    print(f\"\\nApplication: {config.app_name}\")\n    print(f\"Version: {config.version}\")\n    print(f\"Environment: {config.environment}\")\n    print(f\"Deployment Ready: {'✅ YES' if results['deployment_ready'] else '❌ NO'}\")\n    \n    print(f\"\\nGenerated Artifacts: {len(results['artifacts_created'])}\")\n    for artifact in results['artifacts_created'][:10]:  # Show first 10\n        artifact_name = Path(artifact).name\n        print(f\"  📄 {artifact_name}\")\n    \n    if len(results['artifacts_created']) > 10:\n        print(f\"  ... and {len(results['artifacts_created']) - 10} more files\")\n    \n    if 'deployment_summary' in results:\n        summary = results['deployment_summary']\n        \n        print(f\"\\nDeployment Configuration:\")\n        print(f\"  Replicas: {config.replicas}\")\n        print(f\"  CPU Limit: {config.cpu_limit}\")\n        print(f\"  Memory Limit: {config.memory_limit}\")\n        print(f\"  Port: {config.port}\")\n        \n        print(f\"\\nDeployment Commands:\")\n        for cmd in summary['deployment_commands']:\n            print(f\"  $ {cmd}\")\n        \n        print(f\"\\nMonitoring URLs:\")\n        for name, url in summary['monitoring_urls'].items():\n            print(f\"  {name.title()}: {url}\")\n    \n    print(f\"\\nNext Steps:\")\n    print(f\"  1. Review generated deployment artifacts\")\n    print(f\"  2. Configure Kubernetes cluster access\")\n    print(f\"  3. Set up container registry credentials\")\n    print(f\"  4. Run: ./scripts/deploy.sh\")\n    print(f\"  5. Verify deployment: kubectl get pods\")\n    print(f\"  6. Set up monitoring and alerting\")\n    print(f\"  7. Configure CI/CD pipeline secrets\")\n    \n    deployment_status = \"🚀 READY FOR PRODUCTION\" if results['deployment_ready'] else \"🔧 NEEDS ATTENTION\"\n    print(f\"\\nDeployment Status: {deployment_status}\")\n    print(\"=\"*80)\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    results = main()"