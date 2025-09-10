# PS-06 Competition System - Deployment Guide

## Overview

This guide covers deployment options for the PS-06 Competition System, from development setups to production deployments across different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB
- GPU: Optional (CPU-only mode available)

**Recommended Requirements:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM
- Network: 1Gbps for model downloads

### Software Dependencies

- Python 3.9+
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.30+
- Git LFS

### Optional Dependencies

- Kubernetes 1.24+
- Nginx/Apache
- PostgreSQL 13+
- Redis 6+

## Development Deployment

### Quick Start

1. **Clone Repository**
```bash
git clone <repository-url>
cd ps06-system
```

2. **Setup Environment**
```bash
# Run automated setup
./scripts/setup_environment.sh --type development

# Or manual setup
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Download Models**
```bash
./scripts/download_models.sh
```

4. **Start Services**
```bash
# Start database and Redis (if using Docker)
docker-compose up -d postgres redis

# Start application
python -m src.api.main

# Start worker (in another terminal)
celery -A src.tasks.celery_app worker --loglevel=info
```

### Development Configuration

Edit `.env` file for development:

```env
# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true

# Database (use SQLite for development)
DATABASE_URL=sqlite:///dev.db

# Redis (optional for development)
REDIS_URL=redis://localhost:6379/0

# File storage
DATA_DIR=./data
MODELS_DIR=./models
```

## Docker Deployment

### Standard Docker Deployment

1. **Prepare Environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. **Build and Start**
```bash
# Build and start all services
docker-compose up --build

# Or start in background
docker-compose up -d
```

3. **Verify Deployment**
```bash
# Check service status
docker-compose ps

# Check logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/api/v1/health
```

### Production Docker Setup

1. **Use Production Compose File**
```bash
docker-compose -f docker-compose.yml -f configs/deployment/docker-compose.prod.yml up -d
```

2. **Environment Configuration**
```env
# Production settings
DEBUG=false
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
API_PORT=8000

# Secure database
DATABASE_URL=postgresql://ps06_user:secure_password@postgres:5432/ps06_db

# Redis with password
REDIS_URL=redis://:secure_password@redis:6379/0

# Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
```

### Docker Services

The Docker setup includes:

- **API Service**: Main FastAPI application
- **Worker Service**: Celery workers for processing
- **Scheduler Service**: Celery beat scheduler
- **PostgreSQL**: Database
- **Redis**: Message broker and cache
- **MinIO**: Object storage
- **Nginx**: Reverse proxy (production)
- **Flower**: Celery monitoring (optional)

### Docker Volumes

```yaml
volumes:
  postgres_data: # Database data
  redis_data: # Redis persistence
  minio_data: # Object storage
  models: # ML models
  logs: # Application logs
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.24+
- kubectl configured
- Helm 3.x (optional)
- Container registry access

### Basic K8s Deployment

1. **Create Namespace**
```bash
kubectl create namespace ps06
```

2. **Deploy Configuration**
```bash
# Apply all configurations
kubectl apply -f configs/kubernetes/production/ -n ps06

# Or use individual files
kubectl apply -f configs/kubernetes/production/configmap.yaml -n ps06
kubectl apply -f configs/kubernetes/production/secret.yaml -n ps06
kubectl apply -f configs/kubernetes/production/postgres.yaml -n ps06
kubectl apply -f configs/kubernetes/production/redis.yaml -n ps06
kubectl apply -f configs/kubernetes/production/api.yaml -n ps06
kubectl apply -f configs/kubernetes/production/worker.yaml -n ps06
```

3. **Verify Deployment**
```bash
# Check pods
kubectl get pods -n ps06

# Check services
kubectl get services -n ps06

# Check logs
kubectl logs -f deployment/api -n ps06
```

### Kubernetes Configuration Files

**ConfigMap** (`configmap.yaml`):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ps06-config
  namespace: ps06
data:
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  REDIS_URL: "redis://redis:6379/0"
  MODELS_DIR: "/app/models"
```

**Secret** (`secret.yaml`):
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ps06-secrets
  namespace: ps06
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  secret-key: <base64-encoded-secret-key>
```

**API Deployment** (`api.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: ps06
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: ps06-system:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ps06-config
        - secretRef:
            name: ps06-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Scaling and Auto-scaling

1. **Manual Scaling**
```bash
# Scale API pods
kubectl scale deployment api --replicas=5 -n ps06

# Scale workers
kubectl scale deployment worker --replicas=3 -n ps06
```

2. **Horizontal Pod Autoscaler**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: ps06
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
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

## Production Deployment

### Server Preparation

1. **System Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Firewall Configuration**
```bash
# Allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

3. **SSL/TLS Setup**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Production Configuration

1. **Environment Variables**
```env
# Security
SECRET_KEY=your-very-secure-secret-key-here
DEBUG=false
LOG_LEVEL=WARNING

# Database
DATABASE_URL=postgresql://ps06_user:secure_password@localhost:5432/ps06_db

# Redis
REDIS_URL=redis://:secure_password@localhost:6379/0

# File Limits
MAX_FILE_SIZE=500000000

# Performance
CELERY_WORKER_CONCURRENCY=4
DATABASE_POOL_SIZE=20

# Monitoring
SENTRY_DSN=https://your-sentry-dsn-here
```

2. **Nginx Configuration**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    client_max_body_size 500M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    location /api/v1/ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

3. **Systemd Service**
```ini
[Unit]
Description=PS-06 Competition System
After=network.target

[Service]
Type=exec
User=ps06
Group=ps06
WorkingDirectory=/opt/ps06-system
ExecStart=/opt/ps06-system/venv/bin/python -m src.api.main
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/ps06-system

[Install]
WantedBy=multi-user.target
```

### Backup Strategy

1. **Database Backup**
```bash
#!/bin/bash
# Backup script
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/ps06"

# PostgreSQL backup
pg_dump ps06_db > $BACKUP_DIR/db_backup_$DATE.sql

# Files backup
tar -czf $BACKUP_DIR/files_backup_$DATE.tar.gz /opt/ps06-system/data

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

2. **Automated Backup with Cron**
```bash
# Add to crontab
0 2 * * * /opt/ps06-system/scripts/backup.sh
```

## Monitoring and Logging

### Application Monitoring

1. **Health Checks**
```bash
# Basic health
curl -f http://localhost:8000/api/v1/health

# Detailed health
curl -f http://localhost:8000/api/v1/health/detailed

# Metrics
curl -f http://localhost:8000/api/v1/metrics
```

2. **Prometheus Integration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ps06-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 30s
```

### Logging Configuration

1. **Structured Logging**
```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/ps06/application.log",
            "maxBytes": 50000000,
            "backupCount": 10,
            "formatter": "json"
        }
    }
}
```

2. **Log Aggregation with ELK Stack**
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    
  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    
  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
```

### Performance Monitoring

1. **Grafana Dashboard**
```json
{
  "dashboard": {
    "title": "PS-06 System Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

2. **Alert Rules**
```yaml
# alerts.yml
groups:
- name: ps06.rules
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
    for: 5m
    annotations:
      summary: "High response time detected"
```

## Security Considerations

### Authentication and Authorization

1. **API Authentication**
```python
# Add to main.py
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# JWT configuration
jwt_authentication = JWTAuthentication(
    secret=settings.SECRET_KEY,
    lifetime_seconds=3600,
    tokenUrl="/auth/jwt/login"
)
```

2. **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/process")
@limiter.limit("10/minute")
async def process_audio(request: Request):
    pass
```

### Network Security

1. **Firewall Rules**
```bash
# UFW configuration
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

2. **SSL/TLS Configuration**
```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
add_header Strict-Transport-Security "max-age=63072000" always;
```

### Data Security

1. **Environment Variables**
```bash
# Never commit secrets to git
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
```

2. **Database Security**
```sql
-- Create dedicated user
CREATE USER ps06_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE ps06_db TO ps06_user;
GRANT USAGE ON SCHEMA public TO ps06_user;
GRANT CREATE ON SCHEMA public TO ps06_user;
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
```bash
# Check logs
docker-compose logs api
kubectl logs deployment/api -n ps06

# Check configuration
docker-compose config
kubectl describe pod <pod-name> -n ps06
```

2. **Database Connection Issues**
```bash
# Test database connection
psql -h localhost -U ps06_user -d ps06_db

# Check database logs
docker-compose logs postgres
```

3. **Model Loading Errors**
```bash
# Check model directory
ls -la models/
df -h  # Check disk space

# Download models manually
./scripts/download_models.sh --force
```

4. **Performance Issues**
```bash
# Check system resources
htop
nvidia-smi

# Check application metrics
curl http://localhost:8000/api/v1/metrics
```

### Debug Mode

Enable debug mode for troubleshooting:

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "Checking PS-06 System Health..."

# API Health
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✓ API is healthy"
else
    echo "✗ API is not responding"
    exit 1
fi

# Database Health
if docker-compose exec postgres pg_isready > /dev/null 2>&1; then
    echo "✓ Database is healthy"
else
    echo "✗ Database is not responding"
    exit 1
fi

# Redis Health
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is healthy"
else
    echo "✗ Redis is not responding"
    exit 1
fi

echo "All services are healthy!"
```

### Support and Maintenance

1. **Log Analysis**
```bash
# Search for errors
grep -i error /var/log/ps06/application.log

# Monitor real-time logs
tail -f /var/log/ps06/application.log | grep ERROR
```

2. **Performance Tuning**
```bash
# Monitor resource usage
iostat -x 1
vmstat 1
sar -u 1
```

3. **Update Procedure**
```bash
# Backup before update
./scripts/backup.sh

# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart services
docker-compose restart
```

For additional support, refer to:
- [API Documentation](api_documentation.md)
- [Model Integration Guide](model_integration.md)
- GitHub Issues: [repository-url]/issues