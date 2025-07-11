#!/bin/bash
# Control script for local machine
# This script creates SSH tunnel for TensorBoard and manages checkpoints

set -e  # Exit on error

echo "=================================================="
echo "VLM Bridge Training Controller (Local)"
echo "=================================================="

# Parse arguments - command first approach
COMMAND=""
VAST_HOST=""
SSH_KEY=""
SSH_PORT=""

# First argument must be command
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> <user@host> [-i ssh_key_path] [-p port]"
    echo ""
    echo "Commands:"
    echo "  monitor        - Create TensorBoard tunnel"
    echo "  download-best  - Download best model checkpoint"
    echo "  download-latest - Download latest checkpoint"
    echo "  download-weights - Download weights-only file"
    echo "  download-all   - Download all checkpoints"
    echo ""
    echo "Options:"
    echo "  -i, --identity  SSH private key path (required)"
    echo "  -p, --port      SSH port number (required)"
    echo ""
    echo "Examples:"
    echo "  $0 monitor root@ssh1.vast.ai -p 12345 -i ~/.ssh/id_ed25519"
    echo "  $0 download-best root@ssh2.vast.ai -p 23456 -i ~/.ssh/id_ed25519"
    echo "  $0 download-all root@ssh3.vast.ai -p 34567 -i ~/.ssh/id_ed25519"
    echo ""
    exit 1
fi

COMMAND="$1"
shift

# Second argument must be host
if [ $# -eq 0 ]; then
    echo "❌ Error: Missing host argument"
    echo "Usage: $0 $COMMAND <user@host> [-i ssh_key_path] [-p port]"
    exit 1
fi

VAST_HOST="$1"
shift

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--identity)
            if [ -z "$2" ]; then
                echo "❌ Error: SSH key path required after -i"
                exit 1
            fi
            SSH_KEY="$2"
            shift 2
            ;;
        -p|--port)
            if [ -z "$2" ]; then
                echo "❌ Error: Port number required after -p"
                exit 1
            fi
            SSH_PORT="$2"
            shift 2
            ;;
        *)
            echo "❌ Error: Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$SSH_KEY" ]; then
    echo "❌ Error: SSH key required (-i option)"
    exit 1
fi

if [ -z "$SSH_PORT" ]; then
    echo "❌ Error: SSH port required (-p option)"
    exit 1
fi

# Validate SSH key file exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key file not found: $SSH_KEY"
    exit 1
fi

# Build SSH command options
SSH_OPTS="-q -o ConnectTimeout=5 -i $SSH_KEY -p $SSH_PORT"

# Test SSH connection
echo "🔗 Testing connection to $VAST_HOST..."
echo "Using SSH key: $SSH_KEY"
echo "Using port: $SSH_PORT"

if ! ssh $SSH_OPTS $VAST_HOST "echo 'Connected successfully'"; then
    echo "❌ Failed to connect to $VAST_HOST"
    echo "Please check your SSH connection, key file, and port number"
    exit 1
fi

# Execute specific command
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
            ssh $SSH_OPTS -N -L 6006:localhost:6006 $VAST_HOST || {
                echo ""
                echo "🔄 Connection lost. Reconnecting in 5 seconds..."
                sleep 5
            }
        done
        ;;
        
    download-best)
        echo "📥 Downloading best model checkpoint..."
        mkdir -p checkpoints
        
        # List available checkpoints first
        echo "Checking available checkpoints..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find checkpoints -name 'best_model.pth' -type f"
        
        echo "Downloading best_model.pth..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/best_model.pth \
            checkpoints/ || {
            echo "❌ Failed to download best_model.pth"
            echo "Make sure the file exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: ./checkpoints/best_model.pth"
        ;;
        
    download-latest)
        echo "📥 Downloading latest checkpoint..."
        mkdir -p checkpoints
        
        echo "Checking available checkpoints..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find checkpoints -name 'latest_checkpoint.pth' -type f"
        
        echo "Downloading latest_checkpoint.pth..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/latest_checkpoint.pth \
            checkpoints/ || {
            echo "❌ Failed to download latest_checkpoint.pth"
            echo "Make sure the file exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: ./checkpoints/latest_checkpoint.pth"
        ;;
        
    download-weights)
        echo "📥 Downloading weights-only file..."
        mkdir -p checkpoints
        
        echo "Checking available checkpoints..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find checkpoints -name 'best_model_weights_only.pth' -type f"
        
        echo "Downloading best_model_weights_only.pth..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/best_model_weights_only.pth \
            checkpoints/ || {
            echo "❌ Failed to download best_model_weights_only.pth"
            echo "Make sure the file exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: ./checkpoints/best_model_weights_only.pth"
        ;;
        
    download-all)
        echo "📥 Downloading all checkpoints..."
        mkdir -p checkpoints
        
        echo "Listing all available checkpoints..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find checkpoints -name '*.pth' -type f"
        
        echo ""
        echo "Downloading entire checkpoints directory..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/ \
            checkpoints/ || {
            echo "❌ Failed to download checkpoints"
            echo "Make sure the checkpoints directory exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: All checkpoints saved to ./checkpoints/"
        ;;
        
    *)
        echo "❌ Invalid command: $COMMAND"
        echo "Valid commands are: monitor, download-best, download-latest, download-weights, download-all"
        exit 1
        ;;
esac