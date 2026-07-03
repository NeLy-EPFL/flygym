#!/bin/bash
# Package a lazily-loaded asset directory (see
# src/flygym/utils/assets_lazy_loading.py) into the <name>.tar + <name>.checksum
# pair expected on the S3 bucket. The two output files are written next to the
# input folder, named after it, ready to be uploaded as-is.
#
# Usage: scripts/dev/make_tar_for_lazy_loaded_assets.sh <path/to/asset_dir>
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path/to/asset_dir>" >&2
    exit 1
fi

src_dir="$1"
if [ ! -d "$src_dir" ]; then
    echo "Error: '$src_dir' is not a directory." >&2
    exit 1
fi

# Strip a trailing slash so basename gives the asset's name, not "".
src_dir="${src_dir%/}"
name="$(basename "$src_dir")"
out_dir="$(dirname "$src_dir")"
tar_path="$out_dir/$name.tar"
checksum_path="$out_dir/$name.checksum"

# Archive the directory's *contents*, not the directory itself, so archive
# members are bare file names (lazy_load_asset_dir extracts straight into the
# cache dir and expects files there directly, e.g. `mesh_dir / "a.stl"`).
tar -cf "$tar_path" -C "$src_dir" .
sha256sum "$tar_path" | cut -d' ' -f1 > "$checksum_path"

echo "Wrote $tar_path ($(du -h "$tar_path" | cut -f1)) and $checksum_path ($(cat "$checksum_path"))"
