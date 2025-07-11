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
REMOTE_FILE=""
LOCAL_DIR=""

# First argument must be command
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> <user@host> [options]"
    echo ""
    echo "Commands:"
    echo "  monitor        - Create TensorBoard tunnel"
    echo "  download       - Download file/directory from remote"
    echo ""
    echo "Options:"
    echo "  -i, --identity  SSH private key path (required)"
    echo "  -p, --port      SSH port number (required)"
    echo "  -f, --file      Remote file/directory path (for download)"
    echo "  -d, --dir       Local target directory (default: checkpoints/experiment)"
    echo ""
    echo "Examples:"
    echo "  $0 monitor root@ssh1.vast.ai -p 12345 -i ~/.ssh/id_ed25519"
    echo "  $0 download root@host -f checkpoints/experiment/best_model.pth -p 12345 -i ~/.ssh/key"
    echo "  $0 download root@host -f checkpoints/experiment/ -d models/backup -p 12345 -i ~/.ssh/key"
    echo "  $0 download root@host -f checkpoints/experiment/best_model.pth -p 23456 -i ~/.ssh/key"
    echo ""
    exit 1
fi

COMMAND="$1"
shift

# Second argument must be host
if [ $# -eq 0 ]; then
    echo "❌ Error: Missing host argument"
    echo "Usage: $0 $COMMAND <user@host> [options]"
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
        -f|--file)
            if [ -z "$2" ]; then
                echo "❌ Error: Remote file path required after -f"
                exit 1
            fi
            REMOTE_FILE="$2"
            shift 2
            ;;
        -d|--dir)
            if [ -z "$2" ]; then
                echo "❌ Error: Local directory path required after -d"
                exit 1
            fi
            LOCAL_DIR="$2"
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

# Generic download function
download_file() {
    local remote_path="$1"
    local local_dir="$2"
    
    # Set default local directory if not specified
    if [ -z "$local_dir" ]; then
        local_dir="checkpoints/experiment"
    fi
    
    # Ensure local directory exists
    mkdir -p "$local_dir"
    
    # Check if remote path ends with / (directory download)
    if [[ "$remote_path" == */ ]]; then
        echo "📁 Downloading directory: $remote_path"
        echo "Listing available files..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find $remote_path -name '*.pth' -type f 2>/dev/null || echo 'No .pth files found'"
        
        echo ""
        echo "Downloading entire directory..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            "$VAST_HOST:vlm-bridge-for-image-captioning/$remote_path" \
            "$local_dir/" || {
            echo "❌ Failed to download directory: $remote_path"
            echo "Make sure the directory exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: Directory saved to ./$local_dir/"
    else
        # Single file download
        local filename=$(basename "$remote_path")
        echo "📄 Downloading file: $remote_path"
        
        echo "Checking if file exists..."
        ssh $SSH_OPTS $VAST_HOST "cd vlm-bridge-for-image-captioning && find $remote_path -type f 2>/dev/null || echo 'File not found'"
        
        echo "Downloading $filename..."
        rsync -avz --progress -e "ssh $SSH_OPTS" \
            "$VAST_HOST:vlm-bridge-for-image-captioning/$remote_path" \
            "$local_dir/" || {
            echo "❌ Failed to download $filename"
            echo "Make sure the file exists on remote server"
            exit 1
        }
        
        echo "✅ Download complete: ./$local_dir/$filename"
    fi
}

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
        
    download)
        # Validate required arguments for download command
        if [ -z "$REMOTE_FILE" ]; then
            echo "❌ Error: Remote file path required for download command (-f option)"
            echo "Usage: $0 download <user@host> -f <remote_path> [-d local_dir] [-p port] [-i key]"
            exit 1
        fi
        
        download_file "$REMOTE_FILE" "$LOCAL_DIR"
        ;;
        
    *)
        echo "❌ Invalid command: $COMMAND"
        echo "Valid commands are: monitor, download"
        exit 1
        ;;
esac