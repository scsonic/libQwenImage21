#!/bin/bash
# Host MNNConvert (+ qwen_image21_demo for CPU testing on the host).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party/MNN/build_host" && cd "$ROOT/third_party/MNN/build_host"
cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_LLM=ON -DMNN_BUILD_DIFFUSION=ON \
    -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_SEP_BUILD=OFF
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" MNNConvert qwen_image21_demo
