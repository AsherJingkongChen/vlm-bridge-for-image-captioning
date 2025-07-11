#!/bin/bash
# Control script for local machine
# This script creates SSH tunnel for TensorBoard and manages checkpoints

set -e  # Exit on error

echo "=================================================="
echo "VLM Bridge Training Controller (Local)"
echo "=================================================="

# Check if host is provided
if [ -z "$1" ]; then
    echo "Usage: $0 user@vast-instance.com [command]"
    echo ""
    echo "Commands:"
    echo "  monitor   - Create TensorBoard tunnel"
    echo "  download  - Download checkpoints"
    echo ""
    exit 1
fi

VAST_HOST=$1
COMMAND=$2

# Test SSH connection
echo "🔗 Testing connection to $VAST_HOST..."
if ! ssh -q -o ConnectTimeout=5 $VAST_HOST "echo 'Connected'"; then
    echo "❌ Failed to connect to $VAST_HOST"
    echo "Please check your SSH connection"
    exit 1
fi

case $COMMAND in
    monitor)
        echo "📊 Creating TensorBoard tunnel..."
        echo ""
        echo "Local port 6006 → Remote port 6006"
        echo ""
        echo "=================================================="
        echo "TensorBoard will be available at:"
        echo "  http://localhost:6006"
        echo "=================================================="
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        echo ""
        
        # Create SSH tunnel with auto-reconnect
        while true; do
            ssh -N -L 6006:localhost:6006 $VAST_HOST || {
                echo ""
                echo "🔄 Connection lost. Reconnecting in 5 seconds..."
                sleep 5
            }
        done
        ;;
        
    download)
        echo "📥 Downloading checkpoints..."
        echo ""
        
        # List available checkpoints
        echo "Available checkpoints:"
        ssh $VAST_HOST "cd vlm-bridge && find checkpoints -name '*.pth' -type f" || {
            echo "No checkpoints found or unable to list"
            exit 1
        }
        
        echo ""
        echo "Which checkpoint to download?"
        echo "1) best_model.pth (best validation loss)"
        echo "2) latest_checkpoint.pth (most recent)"
        echo "3) best_model_weights_only.pth (weights only, smaller)"
        echo "4) All checkpoints"
        echo ""
        read -p "Enter choice (1-4): " choice
        
        # Create local checkpoint directory
        mkdir -p checkpoints
        
        case $choice in
            1)
                echo "Downloading best_model.pth..."
                scp $VAST_HOST:vlm-bridge/checkpoints/*/best_model.pth checkpoints/ || echo "File not found"
                ;;
            2)
                echo "Downloading latest_checkpoint.pth..."
                scp $VAST_HOST:vlm-bridge/checkpoints/*/latest_checkpoint.pth checkpoints/ || echo "File not found"
                ;;
            3)
                echo "Downloading best_model_weights_only.pth..."
                scp $VAST_HOST:vlm-bridge/checkpoints/*/best_model_weights_only.pth checkpoints/ || echo "File not found"
                ;;
            4)
                echo "Downloading all checkpoints..."
                rsync -avz --progress $VAST_HOST:vlm-bridge/checkpoints/ checkpoints/
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac
        
        echo ""
        echo "✅ Download complete!"
        echo "Checkpoints saved to: ./checkpoints/"
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        echo "Use 'monitor' or 'download'"
        exit 1
        ;;
esac