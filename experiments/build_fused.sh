#!/usr/bin/env bash
# Build an isolated patched engine; existing llama.cpp installations are untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
source_engine=${SOURCE_ENGINE:-/home/everett/llama.cpp-build}
target_engine=${ENGINE:-$PWD/engine}
if [ ! -d "$target_engine/.git" ]; then
    git clone --no-hardlinks "$source_engine" "$target_engine"
    git -C "$target_engine" apply "$PWD/patches/0008-fused-cpu-moe-rows.patch"
fi
cuda_root=${CUDA_HOME:-$HOME/miniconda3/envs/cudabuild}
cmake -S "$target_engine" -B "$target_engine/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_CUDA_NCCL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF
cmake --build "$target_engine/build" --target llama-server -j"${BUILD_JOBS:-4}"
mkdir -p cpp/build
g++ -O2 -std=c++17 cpp/test-moe-rows.cpp -I"$target_engine/ggml/include" \
    -L"$target_engine/build/bin" -lggml-cpu -lggml-base \
    -Wl,-rpath,"$target_engine/build/bin" -o cpp/build/test-moe-rows
OMP_WAIT_POLICY=active cpp/build/test-moe-rows

g++ -O2 -std=c++17 cpp/moe-logits.cpp -I"$target_engine/common" \
    -I"$target_engine/include" -I"$target_engine/ggml/include" \
    -L"$target_engine/build/bin" -lllama-common -lllama -lggml \
    -Wl,-rpath,"$target_engine/build/bin" -o cpp/build/moe-logits
