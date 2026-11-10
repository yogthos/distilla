#!/bin/sh
# Build distilla's native pieces into native/. Run from the project root
# (the `jolt native` task does).
#
#   native/libduckdb.{so,dylib}         DuckDB, the parquet reader and writer
#                                       (prebuilt release, DUCKDB_VERSION)
#   native/llama.cpp/                   llama.cpp at LLAMA_REV, built as shared
#                                       libraries
#   native/libdistilla_llama.{so,dylib} the shim over llama.cpp the jolt side
#                                       binds (distilla_llama.c) and the
#                                       decoder's training graph on ggml
#                                       (distilla_train.c)
#   native/libdistilla_ops.{so,dylib}   the optimizer kernels (distilla_ops.c)
#
# DISTILLA_GPU=cuda|metal|vulkan|hip|none picks llama.cpp's backend: metal by
# default on macOS, none elsewhere. DISTILLA_SKIP_LLAMA=1 builds only DuckDB
# and the optimizer kernels, enough for the tests without the llama.cpp ones.
# DISTILLA_NATIVE=OFF builds llama.cpp for the generic CPU instead of this
# one (CI, where a cached build may run on another machine).
set -e

DUCKDB_VERSION=${DUCKDB_VERSION:-v1.5.6}
LLAMA_REV=26d52b0f9543f441fc2bfa383557c992a4fbf4b9 # b12239
LLAMA_SRC=${LLAMA_CPP_SRC:-https://github.com/ggml-org/llama.cpp}

OS=$(uname -s)
ARCH=$(uname -m)
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
if [ "$OS" = Darwin ]; then EXT=dylib; else EXT=so; fi

# --- DuckDB ----------------------------------------------------------------
if [ ! -f "native/libduckdb.$EXT" ]; then
  case "$OS-$ARCH" in
    Linux-x86_64)  asset=libduckdb-linux-amd64.zip ;;
    Linux-aarch64) asset=libduckdb-linux-arm64.zip ;;
    Darwin-*)      asset=libduckdb-osx-universal.zip ;;
    *) echo "no prebuilt DuckDB for $OS-$ARCH; put libduckdb.$EXT in native/" >&2; exit 1 ;;
  esac
  tmp=native/duckdb-download
  rm -rf "$tmp" && mkdir -p "$tmp"
  curl -sSfL -o "$tmp/$asset" "https://github.com/duckdb/duckdb/releases/download/$DUCKDB_VERSION/$asset"
  (cd "$tmp" && unzip -q "$asset")
  cp "$tmp/libduckdb.$EXT" "native/libduckdb.$EXT"
  cp "$tmp/duckdb.h" native/duckdb.h
  rm -rf "$tmp"
fi

# --- the optimizer kernels (no dependencies) ------------------------------------
cc -O3 -fPIC -shared -std=c11 native/distilla_ops.c -lm -o "native/libdistilla_ops.$EXT"

[ "${DISTILLA_SKIP_LLAMA:-0}" = 1 ] && exit 0

# --- llama.cpp ---------------------------------------------------------------
L=native/llama.cpp
if [ ! -f "$L/.git/HEAD" ] || [ "$(cat "$L/.git/HEAD")" != "$LLAMA_REV" ]; then
  rm -rf "$L"
  mkdir -p "$L"
  git -C "$L" init --quiet
  git -C "$L" fetch --quiet --depth 1 "$LLAMA_SRC" "$LLAMA_REV"
  git -C "$L" checkout --quiet FETCH_HEAD
fi

GPU=${DISTILLA_GPU:-}
if [ -z "$GPU" ]; then
  if [ "$OS" = Darwin ]; then GPU=metal; else GPU=none; fi
fi
set -- -DGGML_METAL=OFF
case "$GPU" in
  metal)  set -- -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON ;;
  cuda)   set -- "$@" -DGGML_CUDA=ON ;;
  vulkan) set -- "$@" -DGGML_VULKAN=ON ;;
  hip)    set -- "$@" -DGGML_HIP=ON ;;
  none)   ;;
  *) echo "unknown DISTILLA_GPU: $GPU (cuda, metal, vulkan, hip or none)" >&2; exit 1 ;;
esac

B=$L/build-$GPU
if [ ! -f "$B/bin/libllama.$EXT" ]; then
  cmake -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE="${DISTILLA_NATIVE:-ON}" "$@"
  cmake --build "$B" --config Release -j "$JOBS" --target llama ggml
fi

# the libraries land in bin/ on Linux and macOS alike
LIBDIR=$B/bin
[ -f "$LIBDIR/libllama.$EXT" ] || LIBDIR=$B/src
if [ "$OS" = Darwin ]; then
  RPATH="-Wl,-rpath,@loader_path/llama.cpp/build-$GPU/bin"
else
  RPATH="-Wl,-rpath,\$ORIGIN/llama.cpp/build-$GPU/bin"
fi
cc -O2 -fPIC -shared -std=c11 native/distilla_llama.c native/distilla_train.c \
  -I"$L/include" -I"$L/ggml/include" \
  -L"$LIBDIR" -lllama -lggml -lggml-base -lggml-cpu -lm $RPATH -o "native/libdistilla_llama.$EXT"
echo "built native/libdistilla_llama.$EXT against llama.cpp $LLAMA_REV ($GPU)"
