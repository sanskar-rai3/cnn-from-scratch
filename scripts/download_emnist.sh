#!/bin/sh

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "Datasets:"
    echo "  balanced"
    echo "  byclass"
    echo "  bymerge"
    echo "  digits"
    echo "  letters"
    echo "  mnist"
    exit 1
fi

DATASET="$1"
DATA_DIR="data/$DATASET"

case "$DATASET" in
    balanced|byclass|bymerge|digits|letters|mnist)
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'"
        exit 1
        ;;
esac

mkdir -p "$DATA_DIR"

ZIP="data/emnist.zip"
URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

if [ ! -f "$ZIP" ]; then
    echo "Downloading EMNIST archive..."

    if command -v curl >/dev/null 2>&1; then
        curl -L "$URL" -o "$ZIP"
    elif command -v wget >/dev/null 2>&1; then
        wget "$URL" -O "$ZIP"
    else
        echo "Error: curl or wget is required."
        exit 1
    fi
else
    echo "EMNIST archive already exists, skipping download."
fi

PREFIX="emnist-$DATASET"

FILES="
gzip/${PREFIX}-train-images-idx3-ubyte.gz
gzip/${PREFIX}-train-labels-idx1-ubyte.gz
gzip/${PREFIX}-test-images-idx3-ubyte.gz
gzip/${PREFIX}-test-labels-idx1-ubyte.gz
"

if [ "$DATASET" != "mnist" ]; then
    FILES="$FILES
gzip/${PREFIX}-mapping.txt"
fi

echo "Extracting EMNIST $DATASET..."

unzip -j "$ZIP" $FILES -d "$DATA_DIR"

for file in "$DATA_DIR"/*.gz
do
    gzip -d "$file"
done

echo "EMNIST $DATASET downloaded successfully!"
