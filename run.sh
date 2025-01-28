#!/bin/bash

# Create storage structure
STORAGE_DIR="./storage"
DIRS=("models" "config" "history")

for dir in "${DIRS[@]}"; do
    mkdir -p "$STORAGE_DIR/$dir"
done

# Run container with persistent storage
docker run -it \
    --rm \
    --name pychat \
    -v "$PWD/$STORAGE_DIR/models:/app/storage/models" \
    -v "$PWD/$STORAGE_DIR/config:/app/storage/config" \
    -v "$PWD/$STORAGE_DIR/history:/app/storage/history" \
    pythonchat