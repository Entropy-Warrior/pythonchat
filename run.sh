#!/bin/bash

# Create storage structure
STORAGE_DIR="./storage"
DIRS=("config" "history")  # Removed "models" since we use it from the image

for dir in "${DIRS[@]}"; do
    mkdir -p "$STORAGE_DIR/$dir"
done

# Check if image exists locally and pull only if needed
IMAGE_NAME="ghcr.io/entropy-warrior/pythonchat:latest"
if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "Image not found locally. Pulling..."
    docker pull $IMAGE_NAME
else
    # Check if newer version is available
    echo "Checking for updates..."
    if ! docker pull $IMAGE_NAME | grep -q "Image is up to date"; then
        echo "Updated to latest version"
    else
        echo "Already using latest version"
    fi
fi

# Run container with persistent storage
docker run -it \
    --rm \
    --name pychat \
    -v "$PWD/$STORAGE_DIR/config:/app/storage/config" \
    -v "$PWD/$STORAGE_DIR/history:/app/storage/history" \
    $IMAGE_NAME