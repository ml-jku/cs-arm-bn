#!/bin/bash

# Script Name: download_jump_metadata.sh
#
# Description: This script retrieves the metadata for the JUMP dataset.
#
# Usage: ./download_jump_metadata.sh [METADATA_DIR] [--force]
#
# The resulting metadata will be stored in $METADATA_DIR.
#
# Output Files:
#   - compound.csv.gz: the table that lists all the compound perturbations in the JUMP.
#   - crispr.csv.gz: the table that lists all the CRISPR perturbations in the JUMP.
#   - orf.csv.gz: the table that lists all the ORF perturbations in the JUMP.
#   - plate.csv.gz: the table that lists all the plates in the JUMP. This is the most important file as it allows to create the path to the other files.
#   - well.csv.gz: the table that links the wells in the JUMP with the perturbation and plate.
#   - microscope_config.csv and microscope_filter.csv: the tables that list the microscope configuration and filters used in the JUMP. Not used in this project.
#   - the compressed files all have an uncompressed version in the same directory.
#
# Author: Gabriel Watkinson
# Date: 2023-05-31

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

METADATA_DIR="${METADATA_DIR:-$REPO_ROOT/data/metadata}"
FORCE=""
CUSTOM_DIR=""

for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE="--force"
            ;;
        *)
            if [ -z "$CUSTOM_DIR" ]; then
                CUSTOM_DIR="$arg"
            else
                echo "Unexpected argument: $arg"
                echo "Usage: $0 [METADATA_DIR] [--force]"
                exit 1
            fi
            ;;
    esac
done

if [ -n "$CUSTOM_DIR" ]; then
    METADATA_DIR="$CUSTOM_DIR"
fi

# Normalize to an absolute path so later `cd` operations do not break it.
if [[ "$METADATA_DIR" != /* ]]; then
    METADATA_DIR="$PWD/$METADATA_DIR"
fi

# Check that files don't already exist.
if [ "$FORCE" != "--force" ] && [ -f "$METADATA_DIR/plate.csv" ] && [ -f "$METADATA_DIR/well.csv" ] && [ -f "$METADATA_DIR/crispr.csv" ] && [ -f "$METADATA_DIR/orf.csv" ] && [ -f "$METADATA_DIR/microscope_config.csv" ] && [ -f "$METADATA_DIR/microscope_filter.csv" ]; then
    echo "Metadata files already exist. Use --force to overwrite."
    exit 1
fi

# Create the directory where the metadata will be stored if needed.
echo "Creating metadata directory: $METADATA_DIR ..."
mkdir -p "$METADATA_DIR"

TMP_DIR="$METADATA_DIR/tmp"
rm -rf "$TMP_DIR"

# Clone the repository containing the metadata for the JUMP dataset.
echo "Cloning metadata repository https://github.com/jump-cellpainting/datasets.git ..."
git clone https://github.com/jump-cellpainting/datasets.git "$TMP_DIR"
cd "$TMP_DIR"


# Keep only the interesting folder.
echo "Moving metadata folder..."
mv "$TMP_DIR/metadata/"* "$METADATA_DIR"


# Decrompress the files.
echo "Decompressing metadata files..."
for file in "$METADATA_DIR"/*.csv.gz; do
    [ -e "$file" ] || continue
    gunzip -c "$file" > "${file%.gz}"
done

# Remove the temporary folder.
echo "Removing temporary folders..."
rm -rf "$TMP_DIR"
rm -f "$METADATA_DIR/LICENCE"
rm -f "$METADATA_DIR/LICENSE"

echo "Metadata retrieval complete."
