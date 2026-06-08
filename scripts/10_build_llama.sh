#!/usr/bin/env bash
# Run ON the Raspberry Pi. Clones and builds llama.cpp (server + bench + cli)
# with native ARM (NEON) optimization.
#
# Compiles with -j3 (not -j4) to leave memory headroom on the 4 GB board: each
# parallel g++ job for the ggml/llama core can exceed 1 GB, and four at once
# risks the OOM-killer aborting the build.
set -euo pipefail
cd "$HOME"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
echo "llama.cpp @ $(git rev-parse --short HEAD)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j3 --target llama-server llama-bench llama-cli
ls -lh build/bin/llama-server build/bin/llama-bench build/bin/llama-cli
