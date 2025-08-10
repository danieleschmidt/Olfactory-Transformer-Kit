"""Production deployment orchestration and management."""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import subprocess

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    service_name: str = "olfactory-transformer"
    version: str = "1.0.0"
    replicas: int = 3
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    gpu_enabled: bool = False
    auto_scaling: bool = True
    health_check_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    log_level: str = "INFO"


class ProductionDeployment:
    """Production deployment orchestrator."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                self.logger.warning(f"Docker client initialization failed: {e}")
                self.docker_client = None
        else:
            self.docker_client = None
    
    def build_docker_image(self, dockerfile_path: Path = Path("Dockerfile")) -> str:
        """Build Docker image for production deployment."""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        tag = f"{self.config.service_name}:{self.config.version}"
        
        self.logger.info(f"Building Docker image: {tag}")
        
        try:
            image, build_logs = self.docker_client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path.name),
                tag=tag,
                rm=True,
                forcerm=True
            )
            
            for log in build_logs:
                if 'stream' in log:
                    self.logger.debug(log['stream'].strip())
            
            self.logger.info(f"Successfully built image: {tag}")
            return tag
            
        except Exception as e:
            self.logger.error(f"Docker build failed: {e}")
            raise
    
    def generate_kubernetes_manifests(self, output_dir: Path) -> List[Path]:
        """Generate Kubernetes deployment manifests."""
        if not HAS_YAML:
            raise RuntimeError("PyYAML required for Kubernetes manifest generation")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifests = []
        
        # Deployment manifest
        deployment_manifest = self._create_deployment_manifest()
        deployment_file = output_dir / "deployment.yaml"
        with open(deployment_file, 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        manifests.append(deployment_file)
        
        # Service manifest
        service_manifest = self._create_service_manifest()
        service_file = output_dir / "service.yaml"
        with open(service_file, 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        manifests.append(service_file)
        
        # ConfigMap for configuration
        configmap_manifest = self._create_configmap_manifest()
        configmap_file = output_dir / "configmap.yaml"
        with open(configmap_file, 'w') as f:
            yaml.dump(configmap_manifest, f, default_flow_style=False)
        manifests.append(configmap_file)
        
        # Horizontal Pod Autoscaler (if enabled)
        if self.config.auto_scaling:
            hpa_manifest = self._create_hpa_manifest()
            hpa_file = output_dir / "hpa.yaml"
            with open(hpa_file, 'w') as f:
                yaml.dump(hpa_manifest, f, default_flow_style=False)
            manifests.append(hpa_file)
        
        self.logger.info(f"Generated {len(manifests)} Kubernetes manifests in {output_dir}")
        return manifests
    
    def _create_deployment_manifest(self) -> Dict[str, Any]:
        """Create Kubernetes Deployment manifest."""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config.service_name,
                'labels': {
                    'app': self.config.service_name,
                    'version': self.config.version
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.config.service_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.service_name,
                            'version': self.config.version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config.service_name,
                            'image': f"{self.config.service_name}:{self.config.version}",
                            'ports': [
                                {'containerPort': 8000, 'name': 'http'},
                                {'containerPort': 8080, 'name': 'metrics'}
                            ],
                            'env': [
                                {'name': 'LOG_LEVEL', 'value': self.config.log_level},
                                {'name': 'SERVICE_NAME', 'value': self.config.service_name},
                                {'name': 'VERSION', 'value': self.config.version}
                            ],
                            'resources': {
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                },
                                'requests': {
                                    'cpu': str(float(self.config.cpu_limit) / 2),
                                    'memory': str(int(self.config.memory_limit[:-2]) // 2) + 'Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': self.config.health_check_endpoint,
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': self.config.health_check_endpoint,
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Add GPU resources if enabled
        if self.config.gpu_enabled:
            deployment['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = '1'
        
        return deployment
    
    def _create_service_manifest(self) -> Dict[str, Any]:
        """Create Kubernetes Service manifest."""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.config.service_name}-service",
                'labels': {
                    'app': self.config.service_name
                }
            },
            'spec': {
                'selector': {
                    'app': self.config.service_name
                },
                'ports': [
                    {
                        'name': 'http',
                        'port': 80,
                        'targetPort': 8000,
                        'protocol': 'TCP'
                    },
                    {
                        'name': 'metrics',
                        'port': 8080,
                        'targetPort': 8080,
                        'protocol': 'TCP'
                    }
                ],
                'type': 'ClusterIP'
            }
        }
    
    def _create_configmap_manifest(self) -> Dict[str, Any]:
        """Create ConfigMap for application configuration."""
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.config.service_name}-config"
            },
            'data': {
                'config.json': json.dumps({
                    'service_name': self.config.service_name,
                    'version': self.config.version,
                    'log_level': self.config.log_level,
                    'health_check_endpoint': self.config.health_check_endpoint,
                    'metrics_endpoint': self.config.metrics_endpoint
                }, indent=2)
            }
        }
    
    def _create_hpa_manifest(self) -> Dict[str, Any]:
        """Create Horizontal Pod Autoscaler manifest."""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config.service_name}-hpa"
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config.service_name
                },
                'minReplicas': 2,
                'maxReplicas': 20,
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
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [{
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }]
                    },
                    'scaleUp': {
                        'stabilizationWindowSeconds': 60,
                        'policies': [{
                            'type': 'Percent',
                            'value': 50,
                            'periodSeconds': 60
                        }]
                    }
                }
            }
        }
    
    def deploy_to_kubernetes(self, manifest_dir: Path, namespace: str = "default") -> bool:
        """Deploy to Kubernetes cluster."""
        manifest_files = list(manifest_dir.glob("*.yaml"))
        
        if not manifest_files:
            self.logger.error(f"No manifest files found in {manifest_dir}")
            return False
        
        try:
            for manifest_file in manifest_files:
                cmd = ["kubectl", "apply", "-f", str(manifest_file), "-n", namespace]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to apply {manifest_file}: {result.stderr}")
                    return False
                
                self.logger.info(f"Applied manifest: {manifest_file}")
            
            self.logger.info(f"Successfully deployed {self.config.service_name} to namespace {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def generate_helm_chart(self, chart_dir: Path) -> Path:
        """Generate Helm chart for deployment."""
        if not HAS_YAML:
            raise RuntimeError("PyYAML required for Helm chart generation")
        
        chart_dir = Path(chart_dir) / self.config.service_name
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = {
            'apiVersion': 'v2',
            'name': self.config.service_name,
            'description': 'Olfactory Transformer AI Service',
            'version': self.config.version,
            'appVersion': self.config.version,
            'type': 'application'
        }
        
        with open(chart_dir / "Chart.yaml", 'w') as f:
            yaml.dump(chart_yaml, f, default_flow_style=False)
        
        # values.yaml
        values_yaml = {
            'replicaCount': self.config.replicas,
            'image': {
                'repository': self.config.service_name,
                'tag': self.config.version,
                'pullPolicy': 'IfNotPresent'
            },
            'service': {
                'type': 'ClusterIP',
                'port': 80,
                'targetPort': 8000
            },
            'resources': {
                'limits': {
                    'cpu': self.config.cpu_limit,
                    'memory': self.config.memory_limit
                },
                'requests': {
                    'cpu': str(float(self.config.cpu_limit) / 2),
                    'memory': str(int(self.config.memory_limit[:-2]) // 2) + 'Gi'
                }
            },
            'autoscaling': {
                'enabled': self.config.auto_scaling,
                'minReplicas': 2,
                'maxReplicas': 20,
                'targetCPUUtilizationPercentage': 70
            },
            'serviceAccount': {
                'create': True,
                'name': ''
            },
            'podSecurityContext': {},
            'securityContext': {}
        }
        
        with open(chart_dir / "values.yaml", 'w') as f:
            yaml.dump(values_yaml, f, default_flow_style=False)
        
        # Create templates directory and basic templates
        templates_dir = chart_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Generated Helm chart in {chart_dir}")
        return chart_dir
    
    def create_monitoring_stack(self, output_dir: Path) -> List[Path]:
        """Create monitoring stack with Prometheus and Grafana."""
        if not HAS_YAML:
            raise RuntimeError("PyYAML required for monitoring stack creation")
        
        output_dir = Path(output_dir) / "monitoring"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifests = []
        
        # Prometheus configuration
        prometheus_config = self._create_prometheus_config()
        prometheus_file = output_dir / "prometheus-config.yaml"
        with open(prometheus_file, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        manifests.append(prometheus_file)
        
        # ServiceMonitor for the application
        service_monitor = self._create_service_monitor()
        service_monitor_file = output_dir / "service-monitor.yaml"
        with open(service_monitor_file, 'w') as f:
            yaml.dump(service_monitor, f, default_flow_style=False)
        manifests.append(service_monitor_file)
        
        self.logger.info(f"Generated monitoring stack in {output_dir}")
        return manifests
    
    def _create_prometheus_config(self) -> Dict[str, Any]:
        """Create Prometheus configuration."""
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'prometheus-config'
            },
            'data': {
                'prometheus.yml': '''
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'olfactory-transformer'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: olfactory-transformer
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: (.+)
      target_label: __address__
      replacement: ${1}:8080
'''
            }
        }
    
    def _create_service_monitor(self) -> Dict[str, Any]:
        """Create ServiceMonitor for Prometheus Operator."""
        return {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': f"{self.config.service_name}-monitor",
                'labels': {
                    'app': self.config.service_name
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': self.config.service_name
                    }
                },
                'endpoints': [{
                    'port': 'metrics',
                    'interval': '30s',
                    'path': self.config.metrics_endpoint
                }]
            }
        }
    
    def generate_ci_cd_pipeline(self, output_dir: Path) -> Path:
        """Generate CI/CD pipeline configuration."""
        pipeline_dir = Path(output_dir) / ".github" / "workflows"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub Actions workflow
        workflow = {
            'name': 'Deploy Olfactory Transformer',
            'on': {
                'push': {
                    'branches': ['main'],
                    'tags': ['v*']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.9'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
pip install -r requirements.txt
pip install pytest pytest-cov
'''
                        },
                        {
                            'name': 'Run tests',
                            'run': 'pytest tests/ --cov=olfactory_transformer --cov-report=xml'
                        }
                    ]
                },
                'build-and-deploy': {
                    'needs': 'test',
                    'runs-on': 'ubuntu-latest',
                    'if': 'github.event_name == \'push\' && github.ref == \'refs/heads/main\'',
                    'steps': [
                        {
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Build Docker image',
                            'run': f'docker build -t {self.config.service_name}:${{{{ github.sha }}}} .'
                        },
                        {
                            'name': 'Deploy to Kubernetes',
                            'run': '''
kubectl set image deployment/olfactory-transformer olfactory-transformer=olfactory-transformer:${{ github.sha }}
kubectl rollout status deployment/olfactory-transformer
'''
                        }
                    ]
                }
            }
        }
        
        workflow_file = pipeline_dir / "deploy.yml"
        if HAS_YAML:
            with open(workflow_file, 'w') as f:
                yaml.dump(workflow, f, default_flow_style=False)
        
        self.logger.info(f"Generated CI/CD pipeline: {workflow_file}")
        return workflow_file
    
    def create_production_checklist(self) -> Dict[str, List[str]]:
        """Generate production deployment checklist."""
        return {
            'Pre-deployment': [
                'Run comprehensive test suite',
                'Perform security scanning',
                'Validate model performance benchmarks',
                'Check resource requirements',
                'Verify monitoring and logging setup',
                'Test rollback procedures'
            ],
            'Deployment': [
                'Build and tag Docker images',
                'Apply Kubernetes manifests',
                'Verify pod startup and health checks',
                'Test service endpoints',
                'Validate auto-scaling configuration',
                'Check monitoring dashboards'
            ],
            'Post-deployment': [
                'Monitor application performance',
                'Verify logging and metrics collection',
                'Test production endpoints',
                'Validate auto-scaling behavior',
                'Document deployment version',
                'Update runbooks and documentation'
            ],
            'Maintenance': [
                'Schedule regular model updates',
                'Monitor resource usage trends',
                'Review and update scaling policies',
                'Backup model artifacts and configurations',
                'Security updates and patches',
                'Performance optimization reviews'
            ]
        }