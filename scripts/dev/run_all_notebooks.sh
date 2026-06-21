#!/bin/sh

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 notebook1.ipynb [notebook2.ipynb ...]" >&2
    exit 2
fi

failed=0

for nb in "$@"; do
    echo "==============================="
    echo "Clearing outputs: $nb"

    if ! jupyter nbconvert --clear-output --inplace "$nb"; then
        echo "ERROR: Failed to clear outputs: $nb" >&2
        failed=1
        continue
    fi

    echo "Executing: $nb"

    if ! jupyter nbconvert \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=600 \
        "$nb"; then

        echo "ERROR: Notebook failed during execution: $nb" >&2
        failed=1
    else
        # tqdm writes the progress bar as many separate \r-prefixed stream
        # outputs; coalesce them into a single final line so it renders
        # correctly in static viewers (GitHub, mkdocs, the wasm viewer).
        python -c "import sys, nbformat
from nbconvert.preprocessors import CoalesceStreamsPreprocessor
nb = nbformat.read(sys.argv[1], as_version=4)
nb, _ = CoalesceStreamsPreprocessor().preprocess(nb, {})
nbformat.write(nb, sys.argv[1])" "$nb"
        echo "SUCCESS: $nb"
    fi
    echo

done

exit "$failed"