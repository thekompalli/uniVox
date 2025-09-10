#!/bin/bash

# PS-06 Competition System - Model Download Script
# Downloads all required models for the competition system

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="${MODELS_DIR:-./models}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
QUIET="${QUIET:-false}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-4}"

# Model URLs and information
declare -A MODELS=(
    ["whisper_large_v3"]="openai/whisper-large-v3"
    ["pyannote_diarization"]="pyannote/speaker-diarization-3.1"
    ["pyannote_segmentation"]="pyannote/segmentation-3.0"
    ["wespeaker_voxceleb"]="pyannote/wespeaker-voxceleb-resnet34-LM"
    ["wav2vec2_xlsr"]="facebook/wav2vec2-large-xlsr-53"
    ["indictrans2_1b"]="ai4bharat/indictrans2-en-indic-1B"
    ["nllb_distilled"]="facebook/nllb-200-distilled-600M"
    ["silero_vad"]="silero/silero-vad"
    ["speechbrain_lang_id"]="speechbrain/lang-id-voxlingua107-ecapa"
)

# Model sizes (approximate, in GB)
declare -A MODEL_SIZES=(
    ["whisper_large_v3"]="3.1"
    ["pyannote_diarization"]="0.5"
    ["pyannote_segmentation"]="0.2"
    ["wespeaker_voxceleb"]="0.3"
    ["wav2vec2_xlsr"]="1.2"
    ["indictrans2_1b"]="4.2"
    ["nllb_distilled"]="2.4"
    ["silero_vad"]="0.1"
    ["speechbrain_lang_id"]="0.8"
)

# Functions
log_info() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warning() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${YELLOW}[WARNING]${NC} $1"
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_header() {
    echo -e "${BLUE}"
    echo "================================================================="
    echo "PS-06 Competition System - Model Download Script"
    echo "================================================================="
    echo -e "${NC}"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [MODEL_NAMES...]

Download models for PS-06 Competition System

OPTIONS:
    -h, --help              Show this help message
    -f, --force             Force re-download existing models
    -q, --quiet             Quiet mode (minimal output)
    -p, --parallel N        Number of parallel downloads (default: 4)
    -d, --dir DIR           Models directory (default: ./models)
    --list                  List available models
    --check                 Check which models are already downloaded
    --size                  Show estimated download sizes

MODEL_NAMES:
    Specific models to download. If none specified, downloads all models.
    Available models: ${!MODELS[@]}

EXAMPLES:
    $0                                  # Download all models
    $0 whisper_large_v3 pyannote_diarization  # Download specific models
    $0 --force --parallel 2            # Force download with 2 parallel downloads
    $0 --check                          # Check downloaded models

ENVIRONMENT VARIABLES:
    MODELS_DIR              Models directory (default: ./models)
    HUGGINGFACE_TOKEN       HuggingFace token for private models
    FORCE_DOWNLOAD          Force re-download (true/false)
    QUIET                   Quiet mode (true/false)
    PARALLEL_DOWNLOADS      Number of parallel downloads

EOF
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Git LFS
    if ! command -v git-lfs &> /dev/null; then
        log_warning "Git LFS not found. Installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        elif command -v brew &> /dev/null; then
            brew install git-lfs
        else
            log_error "Please install git-lfs manually"
            exit 1
        fi
        git lfs install
    fi
    
    # Check HuggingFace CLI
    if ! python3 -c "import huggingface_hub" &> /dev/null; then
        log_info "Installing huggingface_hub..."
        pip3 install huggingface_hub
    fi
    
    # Check disk space
    available_space=$(df "$MODELS_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
    required_space=$((20 * 1024 * 1024))  # 20GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        log_warning "Low disk space. At least 20GB recommended for all models."
    fi
    
    log_success "Requirements check passed"
}

create_directories() {
    log_info "Creating model directories..."
    
    mkdir -p "$MODELS_DIR"/{whisper,pyannote,wespeaker,wav2vec2,indictrans2,nllb,silero,speechbrain,triton}
    
    # Create subdirectories for organization
    mkdir -p "$MODELS_DIR"/whisper/{large-v3,medium,small}
    mkdir -p "$MODELS_DIR"/pyannote/{diarization,segmentation,embedding}
    mkdir -p "$MODELS_DIR"/indictrans2/{en-indic,indic-en}
    mkdir -p "$MODELS_DIR"/triton/{whisper,pyannote,wespeaker}
    
    log_success "Directories created"
}

download_huggingface_model() {
    local model_name="$1"
    local model_path="$2"
    local target_dir="$3"
    
    log_info "Downloading $model_name from HuggingFace..."
    
    # Set HuggingFace token if available
    local hf_token_arg=""
    if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
        hf_token_arg="--token $HUGGINGFACE_TOKEN"
    fi
    
    # Download using huggingface_hub
    python3 -c "
import os
from huggingface_hub import snapshot_download
import sys

try:
    token = os.getenv('HUGGINGFACE_TOKEN')
    snapshot_download(
        repo_id='$model_path',
        local_dir='$target_dir',
        token=token if token else None,
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print('Successfully downloaded $model_name')
except Exception as e:
    print(f'Error downloading $model_name: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Downloaded $model_name"
        
        # Create a marker file
        echo "$(date)" > "$target_dir/.download_complete"
        echo "$model_path" >> "$target_dir/.download_complete"
    else
        log_error "Failed to download $model_name"
        return 1
    fi
}

download_model() {
    local model_name="$1"
    local model_path="${MODELS[${model_name}]}"
    
    # Determine target directory based on model
    case "$model_name" in
        whisper_*)
            target_dir="$MODELS_DIR/whisper/${model_name#whisper_}"
            ;;
        pyannote_*)
            target_dir="$MODELS_DIR/pyannote/${model_name#pyannote_}"
            ;;
        wespeaker_*)
            target_dir="$MODELS_DIR/wespeaker/${model_name#wespeaker_}"
            ;;
        wav2vec2_*)
            target_dir="$MODELS_DIR/wav2vec2/${model_name#wav2vec2_}"
            ;;
        indictrans2_*)
            target_dir="$MODELS_DIR/indictrans2/${model_name#indictrans2_}"
            ;;
        nllb_*)
            target_dir="$MODELS_DIR/nllb/${model_name#nllb_}"
            ;;
        silero_*)
            target_dir="$MODELS_DIR/silero/${model_name#silero_}"
            ;;
        speechbrain_*)
            target_dir="$MODELS_DIR/speechbrain/${model_name#speechbrain_}"
            ;;
        *)
            target_dir="$MODELS_DIR/$model_name"
            ;;
    esac
    
    # Check if already downloaded
    if [[ -f "$target_dir/.download_complete" && "$FORCE_DOWNLOAD" != "true" ]]; then
        log_info "$model_name already downloaded (use --force to re-download)"
        return 0
    fi
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Download the model
    download_huggingface_model "$model_name" "$model_path" "$target_dir"
}

download_special_models() {
    log_info "Downloading special models..."
    
    # Silero VAD (direct download)
    local silero_dir="$MODELS_DIR/silero/vad"
    if [[ ! -f "$silero_dir/.download_complete" || "$FORCE_DOWNLOAD" == "true" ]]; then
        mkdir -p "$silero_dir"
        python3 -c "
import torch
import os

# Download Silero VAD
model, utils = torch.hub.load(repo_or_dir='silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

# Save model
torch.save(model.state_dict(), '$silero_dir/silero_vad.pth')
print('Silero VAD downloaded successfully')

# Save utils
import pickle
with open('$silero_dir/vad_utils.pkl', 'wb') as f:
    pickle.dump(utils, f)

# Create completion marker
with open('$silero_dir/.download_complete', 'w') as f:
    f.write('$(date)\nsilero/silero-vad\n')
"
        log_success "Downloaded Silero VAD"
    fi
}

list_models() {
    echo "Available models:"
    echo "=================="
    for model in "${!MODELS[@]}"; do
        size="${MODEL_SIZES[$model]:-Unknown}"
        echo "  $model (${MODELS[$model]}) - ~${size}GB"
    done
}

check_downloaded_models() {
    echo "Model Download Status:"
    echo "======================"
    
    local total_size=0
    for model in "${!MODELS[@]}"; do
        case "$model" in
            whisper_*) target_dir="$MODELS_DIR/whisper/${model#whisper_}" ;;
            pyannote_*) target_dir="$MODELS_DIR/pyannote/${model#pyannote_}" ;;
            wespeaker_*) target_dir="$MODELS_DIR/wespeaker/${model#wespeaker_}" ;;
            wav2vec2_*) target_dir="$MODELS_DIR/wav2vec2/${model#wav2vec2_}" ;;
            indictrans2_*) target_dir="$MODELS_DIR/indictrans2/${model#indictrans2_}" ;;
            nllb_*) target_dir="$MODELS_DIR/nllb/${model#nllb_}" ;;
            silero_*) target_dir="$MODELS_DIR/silero/${model#silero_}" ;;
            speechbrain_*) target_dir="$MODELS_DIR/speechbrain/${model#speechbrain_}" ;;
            *) target_dir="$MODELS_DIR/$model" ;;
        esac
        
        if [[ -f "$target_dir/.download_complete" ]]; then
            echo -e "  ✅ $model (${MODEL_SIZES[$model]:-?}GB)"
            total_size=$(echo "$total_size + ${MODEL_SIZES[$model]:-0}" | bc -l 2>/dev/null || echo "$total_size")
        else
            echo -e "  ❌ $model (${MODEL_SIZES[$model]:-?}GB)"
        fi
    done
    
    echo "======================"
    echo "Total downloaded size: ~${total_size}GB"
}

show_sizes() {
    echo "Model Download Sizes:"
    echo "===================="
    
    local total=0
    for model in "${!MODELS[@]}"; do
        size="${MODEL_SIZES[$model]:-0}"
        echo "  $model: ~${size}GB"
        total=$(echo "$total + $size" | bc -l 2>/dev/null || echo "$total")
    done
    
    echo "===================="
    echo "Total size: ~${total}GB"
}

main() {
    local models_to_download=()
    local show_list=false
    local show_check=false
    local show_size=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -f|--force)
                FORCE_DOWNLOAD=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -p|--parallel)
                PARALLEL_DOWNLOADS="$2"
                shift 2
                ;;
            -d|--dir)
                MODELS_DIR="$2"
                shift 2
                ;;
            --list)
                show_list=true
                shift
                ;;
            --check)
                show_check=true
                shift
                ;;
            --size)
                show_size=true
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                models_to_download+=("$1")
                shift
                ;;
        esac
    done
    
    print_header
    
    # Handle special commands
    if [[ "$show_list" == "true" ]]; then
        list_models
        exit 0
    fi
    
    if [[ "$show_check" == "true" ]]; then
        check_downloaded_models
        exit 0
    fi
    
    if [[ "$show_size" == "true" ]]; then
        show_sizes
        exit 0
    fi
    
    # Check requirements
    check_requirements
    
    # Create directories
    create_directories
    
    # Determine which models to download
    if [[ ${#models_to_download[@]} -eq 0 ]]; then
        models_to_download=("${!MODELS[@]}")
        log_info "No specific models specified, downloading all models"
    fi
    
    # Validate model names
    for model in "${models_to_download[@]}"; do
        if [[ ! "${MODELS[$model]+isset}" ]]; then
            log_error "Unknown model: $model"
            log_info "Available models: ${!MODELS[*]}"
            exit 1
        fi
    done
    
    # Show download plan
    log_info "Models to download: ${models_to_download[*]}"
    log_info "Download directory: $MODELS_DIR"
    log_info "Parallel downloads: $PARALLEL_DOWNLOADS"
    log_info "Force re-download: $FORCE_DOWNLOAD"
    
    # Download models
    log_info "Starting model downloads..."
    
    local failed_downloads=()
    local successful_downloads=()
    
    # Sequential download for now (can be parallelized later)
    for model in "${models_to_download[@]}"; do
        if download_model "$model"; then
            successful_downloads+=("$model")
        else
            failed_downloads+=("$model")
        fi
    done
    
    # Download special models
    download_special_models
    
    # Summary
    echo
    log_info "Download Summary:"
    log_success "Successfully downloaded: ${#successful_downloads[@]} models"
    if [[ ${#failed_downloads[@]} -gt 0 ]]; then
        log_error "Failed downloads: ${failed_downloads[*]}"
    fi
    
    # Create model registry file
    cat > "$MODELS_DIR/model_registry.json" << EOF
{
    "updated": "$(date -Iseconds)",
    "models": {
$(for model in "${successful_downloads[@]}"; do
    echo "        \"$model\": {"
    echo "            \"path\": \"${MODELS[$model]}\","
    echo "            \"size_gb\": \"${MODEL_SIZES[$model]:-unknown}\","
    echo "            \"downloaded\": true"
    echo "        }$([ "$model" != "${successful_downloads[-1]}" ] && echo ",")"
done)
    }
}
EOF
    
    log_success "Model registry created at $MODELS_DIR/model_registry.json"
    
    if [[ ${#failed_downloads[@]} -eq 0 ]]; then
        log_success "All models downloaded successfully!"
        exit 0
    else
        log_error "Some models failed to download. Check the logs above."
        exit 1
    fi
}

# Run main function
main "$@"