#!/bin/bash

set -e
set -o pipefail

# Parse command line arguments
CPU_ONLY=false

for arg in "$@"; do
    case $arg in
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./scripts/install_pack.sh [--cpu]"
            echo ""
            echo "Options:"
            echo "  --cpu       Install CPU-only version of PyTorch"
            echo "  (no flag)   Install GPU version with CUDA 11.8 (default)"
            echo ""
            echo "Examples:"
            echo "  ./scripts/install_pack.sh          # Install GPU version"
            echo "  ./scripts/install_pack.sh --cpu    # Install CPU version"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
if [ "$CPU_ONLY" = true ]; then
    echo "Installing main package (CPU-only version)..."
    pip install -e .
else
    echo "Installing main package (GPU version with CUDA 11.8)..."
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

    pip install -e .
fi

echo ""
echo "Installing Depth-Anything..."
cd src/seg_cell_tower/third_party_models/Depth-Anything
pip install -e .
cd ../../../..

echo ""
echo "Installing GroundingDINO..."
cd src/seg_cell_tower/third_party_models/GroundingDINO
pip install -e .
cd ../../../..

echo ""
echo "✅ Installation complete!"