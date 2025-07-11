#!/bin/bash
# Control script for local machine
# This script creates SSH tunnel for TensorBoard and manages checkpoints

set -e  # Exit on error

echo "=================================================="
echo "VLM Bridge Training Controller (Local)"
echo "=================================================="

# Parse arguments
VAST_HOST=""
COMMAND=""
SSH_KEY=""
SSH_PORT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--identity)
            SSH_KEY="$2"
            shift 2
            ;;
        -p|--port)
            SSH_PORT="$2"
            shift 2
            ;;
        *)
            if [ -z "$VAST_HOST" ]; then
                VAST_HOST="$1"
            elif [ -z "$COMMAND" ]; then
                COMMAND="$1"
            fi
            shift
            ;;
    esac
done

# Check if host and command are provided
if [ -z "$VAST_HOST" ] || [ -z "$COMMAND" ]; then
    echo "Usage: $0 <user@host> <command> [-i ssh_key_path] [-p port]"
    echo ""
    echo "Commands:"
    echo "  monitor   - Create TensorBoard tunnel"
    echo "  download  - Download checkpoints"
    echo ""
    echo "Options:"
    echo "  -i, --identity  SSH private key path"
    echo "  -p, --port      SSH port number"
    echo ""
    echo "Examples:"
    echo "  $0 root@ssh1.vast.ai download -p 12345"
    echo "  $0 root@ssh2.vast.ai download -p 23456 -i ~/.ssh/id_ed25519"
    echo "  $0 root@ssh3.vast.ai monitor  -p 34567 -i ~/.ssh/id_ed25519"
    echo ""
    exit 1
fi

# Build SSH command options
SSH_OPTS="-q -o ConnectTimeout=5"
if [ -n "$SSH_KEY" ]; then
    if [ ! -f "$SSH_KEY" ]; then
        echo "❌ SSH key file not found: $SSH_KEY"
        exit 1
    fi
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi
if [ -n "$SSH_PORT" ]; then
    SSH_OPTS="$SSH_OPTS -p $SSH_PORT"
fi

# Test SSH connection
echo "🔗 Testing connection to $VAST_HOST..."
if [ -n "$SSH_KEY" ]; then
    echo "Using SSH key: $SSH_KEY"
fi
if ! ssh $SSH_OPTS $VAST_HOST "echo 'Connected'"; then
    echo "❌ Failed to connect to $VAST_HOST"
    echo "Please check your SSH connection and key file"
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
            ssh $SSH_OPTS -N -L 6006:localhost:6006 $VAST_HOST || {
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
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find checkpoints -name '*.pth' -type f" || {
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
        
        # Build SCP/rsync options from SSH options
        SCP_OPTS=""
        RSYNC_SSH_CMD="ssh"
        if [ -n "$SSH_KEY" ]; then
            SCP_OPTS="$SCP_OPTS -i $SSH_KEY"
            RSYNC_SSH_CMD="$RSYNC_SSH_CMD -i $SSH_KEY"
        fi
        if [ -n "$SSH_PORT" ]; then
            SCP_OPTS="$SCP_OPTS -P $SSH_PORT"
            RSYNC_SSH_CMD="$RSYNC_SSH_CMD -p $SSH_PORT"
        fi

        case $choice in
            1)
                echo "Downloading best_model.pth..."
                scp $SCP_OPTS $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/*/best_model.pth checkpoints/ || echo "File not found"
                ;;
            2)
                echo "Downloading latest_checkpoint.pth..."
                scp $SCP_OPTS $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/*/latest_checkpoint.pth checkpoints/ || echo "File not found"
                ;;
            3)
                echo "Downloading best_model_weights_only.pth..."
                scp $SCP_OPTS $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/*/best_model_weights_only.pth checkpoints/ || echo "File not found"
                ;;
            4)
                echo "Downloading all checkpoints..."
                rsync -avz --progress -e "$RSYNC_SSH_CMD" $VAST_HOST:vlm-bridge-for-image-captioning/checkpoints/ checkpoints/
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
        echo "❌ Invalid command: $COMMAND"
        echo "Valid commands are: monitor, download"
        exit 1
        ;;
esac