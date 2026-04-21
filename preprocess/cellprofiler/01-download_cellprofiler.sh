#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
download_dir="${DOWNLOAD_DIR:-$REPO_ROOT/data/cellprofiler_features/aws}"

mkdir -p "$download_dir"

for s in $@
do 
    echo $s
    mkdir $download_dir/source_$s
    aws s3 cp  --no-sign-request s3://cellpainting-gallery/cpg0016-jump/source_$s/workspace/profiles/ $download_dir/source_$s  --exclude "*" --include "*/*/*.parquet" --recursive 
done
