#!/bin/sh

set -e

DATASET="cifar-10"
DATA_DIR="data/$DATASET"
ZIP="$DATA_DIR/cifar-10-binary.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

mkdir -p "$DATA_DIR"

if [ ! -f "$ZIP" ]; then
    echo "Downloading CIFAR-10..."

    if command -v curl >/dev/null 2>&1; then
        curl -L "$URL" -o "$ZIP"
    elif command -v wget >/dev/null 2>&1; then
        wget "$URL" -O "$ZIP"
    else
        echo "Error: curl or wget is required."
        exit 1
    fi
else
    echo "CIFAR-10 archive already exists, skipping download."
fi

echo "Extracting CIFAR-10..."

tar -xzf "$ZIP" -C "$DATA_DIR" --strip-components=1

echo "CIFAR-10 downloaded successfully!"