#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

. util/setupFormatTools.sh

# Get all files that are modified, untracked or added (status in first column) then extract filenames (2nd column)
formatSrcs "$(git status --porcelain | grep -E "^ ?(M|\?\?|A) " | gawk '{print $2}')"

