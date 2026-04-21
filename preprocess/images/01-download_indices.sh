#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
download_dir="${DOWNLOAD_DIR:-$REPO_ROOT/data/all_indices}"

mkdir -p "$download_dir"

sources_to_download=("3" "8")

for s in "${sources_to_download[@]}"
do 
    echo "$s"
    aws s3 cp --no-sign-request "s3://cellpainting-gallery/cpg0016-jump/source_$s/workspace/load_data_csv/" "$download_dir/source_$s" --exclude "*" --include "*/*/load_data.csv" --recursive
done
