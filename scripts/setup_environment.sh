#!/bin/bash

# PS-06 Competition System - Environment Setup Script
# Sets up the complete development and production environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SETUP_TYPE="${SETUP_TYPE:-development}"
SKIP_DEPS="${SKIP_DEPS:-false}"
SKIP_MODELS="${SKIP_MODELS:-false}"
SKIP_DB="${SKIP_DB:-false}"
FORCE="${FORCE:-false}"

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
    echo "PS-06 Competition System - Environment Setup"
    echo "================================================================="
    echo -e "${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup PS-06 Competition System environment

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Setup type: development, production, docker (default: development)
    --skip-deps             Skip dependency installation
    --skip-models           Skip model downloads
    --skip-db               Skip database setup
    --force                 Force reinstallation of existing components

SETUP TYPES:
    development             Full development environment with dev tools
    production              Production environment with minimal dependencies
    docker                  Setup for Docker deployment
    testing                 Testing environment setup

EXAMPLES:
    $0                                  # Development setup
    $0 --type production               # Production setup
    $0 --skip-models --skip-db         # Quick setup without models/db

EOF
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null; then
            echo "centos"
        elif command -v pacman &> /dev/null; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

check_system_requirements() {
    log_info "Checking system requirements..."
    
    local os_type
    os_type=$(detect_os)
    log_info "Detected OS: $os_type"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        local python_version
        python_version=$(python3 --version | cut -d' ' -f2)
        log_info "Python version: $python_version"
        
        if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
            log_error "Python 3.9+ required, found $python_version"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        local total_mem
        total_mem=$(free -g | awk '/^Mem:/{print $2}')
        log_info "Available memory: ${total_mem}GB"
        
        if [[ $total_mem -lt 8 ]]; then
            log_warning "At least 8GB RAM recommended, found ${total_mem}GB"
        fi
    fi
    
    # Check available disk space
    local available_space
    available_space=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    log_info "Available disk space: ${available_space}GB"
    
    if [[ $available_space -lt 50 ]]; then
        log_warning "At least 50GB disk space recommended, found ${available_space}GB"
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        log_info "GPU detected: $gpu_info"
    else
        log_warning "No NVIDIA GPU detected (CPU-only mode)"
    fi
    
    log_success "System requirements check completed"
}

install_system_dependencies() {
    local os_type
    os_type=$(detect_os)
    
    log_info "Installing system dependencies for $os_type..."
    
    case $os_type in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                git-lfs \
                curl \
                wget \
                unzip \
                software-properties-common \
                ffmpeg \
                libsndfile1-dev \
                libsox-dev \
                sox \
                python3-dev \
                python3-pip \
                python3-venv \
                postgresql-client \
                redis-tools \
                htop \
                vim \
                tree
            ;;
        centos)
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                curl \
                wget \
                unzip \
                ffmpeg \
                sox \
                python3-devel \
                python3-pip \
                postgresql \
                redis \
                htop \
                vim \
                tree
            ;;
        macos)
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                cmake \
                git \
                git-lfs \
                ffmpeg \
                sox \
                postgresql \
                redis \
                htop \
                tree
            ;;
        *)
            log_warning "Unsupported OS: $os_type. Please install dependencies manually."
            ;;
    esac
    
    # Install Git LFS
    git lfs install
    
    log_success "System dependencies installed"
}

setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements based on setup type
    case $SETUP_TYPE in
        development)
            log_info "Installing development dependencies..."
            pip install -r requirements.txt
            if [[ -f "requirements-dev.txt" ]]; then
                pip install -r requirements-dev.txt
            fi
            # Install package in development mode
            pip install -e ".[dev]"
            ;;
        production)
            log_info "Installing production dependencies..."
            pip install -r requirements.txt
            pip install .
            ;;
        testing)
            log_info "Installing testing dependencies..."
            pip install -r requirements.txt
            if [[ -f "requirements-test.txt" ]]; then
                pip install -r requirements-test.txt
            fi
            pip install -e ".[dev,test]"
            ;;
        docker)
            log_info "Docker setup - skipping Python environment"
            return 0
            ;;
    esac
    
    log_success "Python environment setup completed"
}

setup_configuration() {
    log_info "Setting up configuration..."
    
    # Copy environment file
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            log_info "Created .env from .env.example"
        else
            log_warning ".env.example not found, creating minimal .env"
            cat > .env << EOF
# PS-06 Competition System Configuration
APP_NAME="PS-06 Competition System"
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://ps06_user:ps06_password@localhost:5432/ps06_db
REDIS_URL=redis://localhost:6379/0

# Models
MODELS_DIR=./models
DATA_DIR=./data
LOGS_DIR=./logs

# GPU
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
EOF
        fi
        
        log_warning "Please edit .env file with your specific configuration"
    else
        log_info ".env file already exists"
    fi
    
    # Create necessary directories
    mkdir -p data/{audio_input,processed,results,speaker_gallery}
    mkdir -p logs
    mkdir -p models
    mkdir -p temp
    
    # Set permissions
    chmod 755 data logs models temp
    
    log_success "Configuration setup completed"
}

setup_database() {
    if [[ "$SKIP_DB" == "true" ]]; then
        log_info "Skipping database setup"
        return 0
    fi
    
    log_info "Setting up database..."
    
    # Check if PostgreSQL is running
    if ! command -v psql &> /dev/null; then
        log_warning "PostgreSQL client not found. Please install PostgreSQL."
        return 1
    fi
    
    # Load environment
    if [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
    fi
    
    # Try to connect to database
    local db_host="${POSTGRES_HOST:-localhost}"
    local db_port="${POSTGRES_PORT:-5432}"
    local db_name="${POSTGRES_DB:-ps06_db}"
    local db_user="${POSTGRES_USER:-ps06_user}"
    
    if pg_isready -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" &> /dev/null; then
        log_info "Database connection successful"
    else
        log_warning "Cannot connect to database. Please ensure PostgreSQL is running."
        log_info "To start PostgreSQL:"
        
        local os_type
        os_type=$(detect_os)
        case $os_type in
            ubuntu)
                echo "  sudo systemctl start postgresql"
                ;;
            macos)
                echo "  brew services start postgresql"
                ;;
            *)
                echo "  Check your system's PostgreSQL documentation"
                ;;
        esac
        
        return 1
    fi
    
    # Run database migrations if available
    if [[ -f "alembic.ini" ]]; then
        log_info "Running database migrations..."
        alembic upgrade head
    fi
    
    log_success "Database setup completed"
}

setup_redis() {
    log_info "Setting up Redis..."
    
    # Check if Redis is running
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            log_info "Redis connection successful"
        else
            log_warning "Cannot connect to Redis. Please ensure Redis is running."
            
            local os_type
            os_type=$(detect_os)
            case $os_type in
                ubuntu)
                    echo "  sudo systemctl start redis-server"
                    ;;
                macos)
                    echo "  brew services start redis"
                    ;;
                *)
                    echo "  Check your system's Redis documentation"
                    ;;
            esac
        fi
    else
        log_warning "Redis client not found. Please install Redis."
    fi
    
    log_success "Redis setup completed"
}

download_models() {
    if [[ "$SKIP_MODELS" == "true" ]]; then
        log_info "Skipping model downloads"
        return 0
    fi
    
    log_info "Downloading models..."
    
    if [[ -f "scripts/download_models.sh" ]]; then
        chmod +x scripts/download_models.sh
        
        local download_args=""
        if [[ "$SETUP_TYPE" == "production" ]]; then
            download_args="--quiet"
        fi
        
        ./scripts/download_models.sh $download_args
    else
        log_warning "Model download script not found"
    fi
    
    log_success "Model download completed"
}

setup_development_tools() {
    if [[ "$SETUP_TYPE" != "development" ]]; then
        return 0
    fi
    
    log_info "Setting up development tools..."
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        log_info "Pre-commit hooks installed"
    fi
    
    # Setup IDE configurations
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
EOF
        log_info "VS Code configuration created"
    fi
    
    # Create development aliases
    cat > .dev_aliases << EOF
# PS-06 Development Aliases
alias ps06-server='python -m src.api.main'
alias ps06-worker='celery -A src.tasks.celery_app worker --loglevel=info'
alias ps06-test='pytest tests/ -v'
alias ps06-test-cov='pytest tests/ --cov=src --cov-report=html'
alias ps06-format='black src/ tests/ && isort src/ tests/'
alias ps06-lint='flake8 src/ tests/ && mypy src/'
alias ps06-logs='tail -f logs/ps06_system.log'

# Add to your shell profile:
# echo 'source $(pwd)/.dev_aliases' >> ~/.bashrc
EOF
    
    log_info "Development aliases created in .dev_aliases"
    log_info "Add 'source $(pwd)/.dev_aliases' to your shell profile"
    
    log_success "Development tools setup completed"
}

setup_docker() {
    if [[ "$SETUP_TYPE" != "docker" ]]; then
        return 0
    fi
    
    log_info "Setting up Docker environment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    
    # Build Docker images
    log_info "Building Docker images..."
    docker-compose build
    
    # Create Docker volumes
    docker-compose up --no-start
    
    log_success "Docker setup completed"
}

run_tests() {
    if [[ "$SETUP_TYPE" == "production" ]]; then
        return 0
    fi
    
    log_info "Running tests to verify setup..."
    
    if [[ -f "scripts/run_tests.sh" ]]; then
        chmod +x scripts/run_tests.sh
        ./scripts/run_tests.sh --quick
    else
        if command -v pytest &> /dev/null; then
            pytest tests/ -x --tb=short
        else
            log_warning "pytest not found, skipping tests"
        fi
    fi
    
    log_success "Tests completed"
}

print_completion_message() {
    echo
    log_success "PS-06 Competition System setup completed!"
    echo
    echo "Next steps:"
    echo "==========="
    
    case $SETUP_TYPE in
        development)
            echo "1. Activate virtual environment:"
            echo "   source venv/bin/activate"
            echo
            echo "2. Edit .env file with your configuration"
            echo
            echo "3. Start the development server:"
            echo "   python -m src.api.main"
            echo
            echo "4. Start a worker (in another terminal):"
            echo "   celery -A src.tasks.celery_app worker --loglevel=info"
            echo
            echo "5. Access the API documentation:"
            echo "   http://localhost:8000/docs"
            ;;
        production)
            echo "1. Edit .env file with production configuration"
            echo "2. Set up reverse proxy (nginx)"
            echo "3. Configure process manager (systemd, supervisor)"
            echo "4. Set up monitoring and logging"
            echo "5. Configure backups"
            ;;
        docker)
            echo "1. Edit .env file for Docker configuration"
            echo "2. Start services:"
            echo "   docker-compose up -d"
            echo "3. Check service status:"
            echo "   docker-compose ps"
            echo "4. View logs:"
            echo "   docker-compose logs -f"
            ;;
        testing)
            echo "1. Run full test suite:"
            echo "   ./scripts/run_tests.sh"
            echo "2. Generate coverage report:"
            echo "   pytest --cov=src --cov-report=html"
            ;;
    esac
    
    echo
    echo "Documentation: README.md"
    echo "Support: https://github.com/your-org/ps06-system/issues"
    echo
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -t|--type)
                SETUP_TYPE="$2"
                shift 2
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-db)
                SKIP_DB=true
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
    
    # Validate setup type
    case $SETUP_TYPE in
        development|production|docker|testing)
            log_info "Setup type: $SETUP_TYPE"
            ;;
        *)
            log_error "Invalid setup type: $SETUP_TYPE"
            print_usage
            exit 1
            ;;
    esac
    
    # Run setup steps
    check_system_requirements
    
    if [[ "$SKIP_DEPS" != "true" ]]; then
        install_system_dependencies
    fi
    
    if [[ "$SETUP_TYPE" != "docker" ]]; then
        setup_python_environment
    fi
    
    setup_configuration
    setup_database
    setup_redis
    download_models
    setup_development_tools
    setup_docker
    run_tests
    
    print_completion_message
}

# Run main function
main "$@"