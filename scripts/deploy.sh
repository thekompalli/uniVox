#!/bin/bash

# PS-06 Competition System - Deployment Script
# Automated deployment for different environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${ENVIRONMENT:-staging}"
DEPLOY_TYPE="${DEPLOY_TYPE:-docker}"
VERSION="${VERSION:-latest}"
BACKUP="${BACKUP:-true}"
ROLLBACK="${ROLLBACK:-false}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"

# Deployment settings
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
PROJECT_NAME="${PROJECT_NAME:-ps06-system}"
NAMESPACE="${NAMESPACE:-ps06}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_header() {
    echo -e "${BLUE}"
    echo "================================================================="
    echo "PS-06 Competition System - Deployment Script"
    echo "================================================================="
    echo -e "${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy PS-06 Competition System

OPTIONS:
    -h, --help              Show this help message
    -e, --env ENV           Environment: development, staging, production (default: staging)
    -t, --type TYPE         Deployment type: docker, kubernetes, bare-metal (default: docker)
    -v, --version VER       Version/tag to deploy (default: latest)
    --no-backup             Skip backup before deployment
    --rollback              Rollback to previous version
    --dry-run               Show what would be deployed without executing
    --force                 Force deployment even with warnings

ENVIRONMENTS:
    development             Local development deployment
    staging                 Staging environment
    production              Production environment

DEPLOYMENT TYPES:
    docker                  Docker Compose deployment
    kubernetes              Kubernetes deployment
    bare-metal              Direct server deployment

EXAMPLES:
    $0                                      # Deploy staging with Docker
    $0 --env production --type kubernetes   # Deploy to production K8s
    $0 --rollback --env production         # Rollback production
    $0 --dry-run --env staging             # Preview staging deployment

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Docker registry URL
    KUBECONFIG             Kubernetes config file
    DEPLOY_KEY             Deployment SSH key
    SLACK_WEBHOOK          Slack notification webhook

EOF
}

check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check deployment type requirements
    case $DEPLOY_TYPE in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker not found"
                exit 1
            fi
            
            if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
                log_error "Docker Compose not found"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl not found"
                exit 1
            fi
            
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        bare-metal)
            if ! command -v systemctl &> /dev/null; then
                log_warning "systemctl not found, some features may not work"
            fi
            ;;
    esac
    
    # Check environment-specific requirements
    case $ENVIRONMENT in
        production)
            if [[ "$FORCE" != "true" ]]; then
                read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirm
                if [[ "$confirm" != "yes" ]]; then
                    log_info "Deployment cancelled"
                    exit 0
                fi
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

load_environment_config() {
    log_info "Loading environment configuration for $ENVIRONMENT..."
    
    # Load environment-specific config
    local env_file=".env.$ENVIRONMENT"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
        log_info "Loaded $env_file"
    else
        log_warning "$env_file not found, using default .env"
        if [[ -f ".env" ]]; then
            set -a
            source .env
            set +a
        fi
    fi
    
    # Override with environment-specific values
    case $ENVIRONMENT in
        production)
            export DEBUG=false
            export LOG_LEVEL=WARNING
            export CELERY_WORKER_CONCURRENCY=4
            ;;
        staging)
            export DEBUG=false
            export LOG_LEVEL=INFO
            export CELERY_WORKER_CONCURRENCY=2
            ;;
        development)
            export DEBUG=true
            export LOG_LEVEL=DEBUG
            export CELERY_WORKER_CONCURRENCY=1
            ;;
    esac
    
    log_success "Environment configuration loaded"
}

create_backup() {
    if [[ "$BACKUP" != "true" ]]; then
        log_info "Skipping backup"
        return 0
    fi
    
    log_info "Creating backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)_${ENVIRONMENT}_${VERSION}"
    mkdir -p "$backup_dir"
    
    case $DEPLOY_TYPE in
        docker)
            # Backup Docker volumes
            if docker-compose ps | grep -q "Up"; then
                log_info "Backing up Docker volumes..."
                
                # Database backup
                if docker-compose exec -T postgres pg_isready &> /dev/null; then
                    docker-compose exec -T postgres pg_dump -U "${POSTGRES_USER:-ps06_user}" "${POSTGRES_DB:-ps06_db}" > "$backup_dir/database.sql"
                fi
                
                # File data backup
                docker run --rm -v ps06_data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/data.tar.gz -C /data .
            fi
            ;;
        kubernetes)
            # Backup Kubernetes resources
            kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/k8s_resources.yaml"
            
            # Database backup if available
            if kubectl get pod -n "$NAMESPACE" -l app=postgres | grep -q Running; then
                kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dump -U "${POSTGRES_USER:-ps06_user}" "${POSTGRES_DB:-ps06_db}" > "$backup_dir/database.sql"
            fi
            ;;
    esac
    
    # Configuration backup
    cp -r configs/ "$backup_dir/" 2>/dev/null || true
    cp .env* "$backup_dir/" 2>/dev/null || true
    
    log_success "Backup created in $backup_dir"
    echo "$backup_dir" > .last_backup
}

build_images() {
    if [[ "$DEPLOY_TYPE" != "docker" && "$DEPLOY_TYPE" != "kubernetes" ]]; then
        return 0
    fi
    
    log_info "Building Docker images..."
    
    local build_args=()
    
    # Add build arguments
    build_args+=(--build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
    build_args+=(--build-arg "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')")
    build_args+=(--build-arg "VERSION=$VERSION")
    
    # Environment-specific build target
    case $ENVIRONMENT in
        production)
            build_args+=(--target runtime)
            ;;
        staging)
            build_args+=(--target runtime)
            ;;
        development)
            build_args+=(--target development)
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would build with: docker build ${build_args[*]} -t $PROJECT_NAME:$VERSION ."
        return 0
    fi
    
    # Build main image
    docker build "${build_args[@]}" -t "$PROJECT_NAME:$VERSION" .
    
    # Tag for registry if specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        docker tag "$PROJECT_NAME:$VERSION" "$DOCKER_REGISTRY/$PROJECT_NAME:$VERSION"
        docker tag "$PROJECT_NAME:$VERSION" "$DOCKER_REGISTRY/$PROJECT_NAME:latest"
        
        log_info "Pushing to registry..."
        docker push "$DOCKER_REGISTRY/$PROJECT_NAME:$VERSION"
        docker push "$DOCKER_REGISTRY/$PROJECT_NAME:latest"
    fi
    
    log_success "Docker images built and pushed"
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    # Select compose file based on environment
    local compose_files=("-f" "docker-compose.yml")
    
    case $ENVIRONMENT in
        production)
            if [[ -f "configs/deployment/docker-compose.prod.yml" ]]; then
                compose_files+=("-f" "configs/deployment/docker-compose.prod.yml")
            fi
            ;;
        staging)
            if [[ -f "docker-compose.staging.yml" ]]; then
                compose_files+=("-f" "docker-compose.staging.yml")
            fi
            ;;
        development)
            if [[ -f "docker-compose.dev.yml" ]]; then
                compose_files+=("-f" "docker-compose.dev.yml")
            fi
            ;;
    esac
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}_${ENVIRONMENT}"
    export VERSION="$VERSION"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would execute:"
        echo "docker-compose ${compose_files[*]} config"
        docker-compose "${compose_files[@]}" config
        return 0
    fi
    
    # Deploy
    log_info "Starting services..."
    docker-compose "${compose_files[@]}" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f "http://localhost:${API_PORT:-8000}/api/v1/health" &> /dev/null; then
            break
        fi
        
        ((attempt++))
        sleep 5
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to start within timeout"
            docker-compose "${compose_files[@]}" logs
            exit 1
        fi
    done
    
    log_success "Docker deployment completed"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Apply configurations
    local k8s_dir="configs/kubernetes/$ENVIRONMENT"
    
    if [[ ! -d "$k8s_dir" ]]; then
        log_error "Kubernetes configurations not found for $ENVIRONMENT"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would apply:"
        kubectl diff -f "$k8s_dir/" -n "$NAMESPACE" || true
        return 0
    fi
    
    # Apply configurations
    log_info "Applying Kubernetes configurations..."
    kubectl apply -f "$k8s_dir/" -n "$NAMESPACE"
    
    # Update image tags
    kubectl set image deployment/api api="$DOCKER_REGISTRY/$PROJECT_NAME:$VERSION" -n "$NAMESPACE"
    kubectl set image deployment/worker worker="$DOCKER_REGISTRY/$PROJECT_NAME:$VERSION" -n "$NAMESPACE"
    
    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    kubectl rollout status deployment/api -n "$NAMESPACE" --timeout=600s
    kubectl rollout status deployment/worker -n "$NAMESPACE" --timeout=600s
    
    log_success "Kubernetes deployment completed"
}

deploy_bare_metal() {
    log_info "Deploying to bare metal..."
    
    local deploy_dir="/opt/$PROJECT_NAME"
    local service_name="$PROJECT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy to $deploy_dir"
        return 0
    fi
    
    # Create deployment directory
    sudo mkdir -p "$deploy_dir"
    
    # Copy application files
    sudo rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' . "$deploy_dir/"
    
    # Install Python dependencies
    sudo "$deploy_dir/venv/bin/pip" install -r "$deploy_dir/requirements.txt"
    
    # Create systemd service
    sudo tee "/etc/systemd/system/$service_name.service" > /dev/null << EOF
[Unit]
Description=PS-06 Competition System
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=ps06
Group=ps06
WorkingDirectory=$deploy_dir
Environment=PYTHONPATH=$deploy_dir
ExecStart=$deploy_dir/venv/bin/python -m src.api.main
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    # Create worker service
    sudo tee "/etc/systemd/system/$service_name-worker.service" > /dev/null << EOF
[Unit]
Description=PS-06 Competition System Worker
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=ps06
Group=ps06
WorkingDirectory=$deploy_dir
Environment=PYTHONPATH=$deploy_dir
ExecStart=$deploy_dir/venv/bin/celery -A src.tasks.celery_app worker --loglevel=info
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and start services
    sudo systemctl daemon-reload
    sudo systemctl enable "$service_name" "$service_name-worker"
    sudo systemctl restart "$service_name" "$service_name-worker"
    
    log_success "Bare metal deployment completed"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    local health_endpoint=""
    
    case $DEPLOY_TYPE in
        docker)
            health_endpoint="http://localhost:${API_PORT:-8000}/api/v1/health"
            ;;
        kubernetes)
            # Port forward for testing
            kubectl port-forward -n "$NAMESPACE" service/api 8080:8000 &
            local port_forward_pid=$!
            sleep 5
            health_endpoint="http://localhost:8080/api/v1/health"
            ;;
        bare-metal)
            health_endpoint="http://localhost:${API_PORT:-8000}/api/v1/health"
            ;;
    esac
    
    # Test health endpoint
    local max_attempts=10
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f "$health_endpoint" &> /dev/null; then
            log_success "Health check passed"
            break
        fi
        
        ((attempt++))
        sleep 10
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Health check failed"
            exit 1
        fi
    done
    
    # Cleanup port forward if used
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]] && [[ -n "${port_forward_pid:-}" ]]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    # Run smoke tests
    if [[ -f "scripts/smoke_tests.sh" ]]; then
        log_info "Running smoke tests..."
        ./scripts/smoke_tests.sh "$ENVIRONMENT"
    fi
    
    log_success "Deployment verification completed"
}

rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [[ ! -f ".last_backup" ]]; then
        log_error "No backup information found for rollback"
        exit 1
    fi
    
    local backup_dir
    backup_dir=$(cat .last_backup)
    
    if [[ ! -d "$backup_dir" ]]; then
        log_error "Backup directory not found: $backup_dir"
        exit 1
    fi
    
    case $DEPLOY_TYPE in
        docker)
            log_info "Rolling back Docker deployment..."
            docker-compose down
            
            # Restore database if available
            if [[ -f "$backup_dir/database.sql" ]]; then
                log_info "Restoring database..."
                docker-compose up -d postgres
                sleep 10
                docker-compose exec -T postgres psql -U "${POSTGRES_USER:-ps06_user}" -d "${POSTGRES_DB:-ps06_db}" < "$backup_dir/database.sql"
            fi
            
            # Restore data
            if [[ -f "$backup_dir/data.tar.gz" ]]; then
                docker run --rm -v ps06_data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar xzf /backup/data.tar.gz -C /data
            fi
            ;;
        kubernetes)
            log_info "Rolling back Kubernetes deployment..."
            kubectl rollout undo deployment/api -n "$NAMESPACE"
            kubectl rollout undo deployment/worker -n "$NAMESPACE"
            ;;
    esac
    
    log_success "Rollback completed"
}

send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local color="good"
        if [[ "$status" == "failed" ]]; then
            color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"title\":\"PS-06 Deployment $status\",\"text\":\"$message\",\"fields\":[{\"title\":\"Environment\",\"value\":\"$ENVIRONMENT\",\"short\":true},{\"title\":\"Version\",\"value\":\"$VERSION\",\"short\":true}]}]}" \
            "$SLACK_WEBHOOK" || true
    fi
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOY_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            --no-backup)
                BACKUP=false
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    print_header
    
    # Validate arguments
    case $ENVIRONMENT in
        development|staging|production) ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    case $DEPLOY_TYPE in
        docker|kubernetes|bare-metal) ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
    
    log_info "Deployment Configuration:"
    log_info "  Environment: $ENVIRONMENT"
    log_info "  Type: $DEPLOY_TYPE"
    log_info "  Version: $VERSION"
    log_info "  Backup: $BACKUP"
    log_info "  Dry Run: $DRY_RUN"
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        send_notification "rolled back" "Deployment rolled back successfully"
        exit 0
    fi
    
    # Main deployment flow
    local start_time
    start_time=$(date +%s)
    
    # Trap for error handling
    trap 'send_notification "failed" "Deployment failed at step: $current_step"' ERR
    
    current_step="prerequisites"
    check_prerequisites
    
    current_step="configuration"
    load_environment_config
    
    current_step="backup"
    create_backup
    
    current_step="build"
    build_images
    
    current_step="deploy"
    case $DEPLOY_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        bare-metal)
            deploy_bare_metal
            ;;
    esac
    
    current_step="verification"
    verify_deployment
    
    # Calculate deployment time
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Deployment completed successfully in ${duration}s"
    send_notification "succeeded" "Deployment completed successfully in ${duration}s"
}

# Run main function
main "$@"