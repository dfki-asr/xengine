#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

. util/setupFormatTools.sh

formatSrcs "$(find examples src test util -type f)"

