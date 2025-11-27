#!/bin/bash

CONFIG_DIR="examples/config_downsampled"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./preprocessor.sh --load_config "$file"
done

echo "All configurations processed."
