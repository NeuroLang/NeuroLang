#!/usr/bin/env bash
# Print a hash of the files under the directory $1 (their paths and
# contents), or "none" when it does not exist: a cache key that changes
# exactly when they do.
set -euo pipefail
if [ ! -d "$1" ]; then echo none; exit 0; fi
cd "$1"
find . -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-16
