if [ ! -d .git ]; then
  echo "Must be run from root of git repo!"
  exit 1
fi

function formatSrcs {
  srcs="${1}"

  CLANG_FMT_CMD=util/clang-format
  PY_FMT_CMD="python3 external/yapf/yapf"

  # Run in parallel and wait
  echo "${srcs}" | grep -E ".*\.[chi](pp)?$"  | xargs --verbose -r ${CLANG_FMT_CMD} -style=file -i &
  echo "${srcs}" | grep -E ".*\.py(\.cmake)?$" | PYTHONPATH="$PWD/external/yapf:${PYTHONPATH:-}" xargs --verbose -r ${PY_FMT_CMD} -i -p &
  wait
}
