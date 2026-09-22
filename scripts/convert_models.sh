#!/bin/bash
# Rebuild the MNN model directory from the public sources (you normally just download the result:
#   hf download evankuo/Qwen-Image-2.1-MNN --local-dir models/qwen_image21).
#
# Needs: python env with export/requirements.txt, and MNNConvert (scripts/build_mnnconvert_mac.sh).
# Downloads ~10.8 GB: GGUF Q4_K DiT (4.2 GB), Qwen-Image-2.1 VAE (1.35 GB), Qwen3-VL-8B MNN (5.1 GB).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
export MNNCONVERT="${MNNCONVERT:-$ROOT/third_party/MNN/build_host/MNNConvert}"
SRC="$ROOT/models/src"
OUT="$ROOT/models/qwen_image21"
mkdir -p "$SRC" "$OUT/text_encoder"

hf download leejet/Qwen-Image-2.1-GGUF qwen_image_2.1-Q4_K.gguf --local-dir "$SRC/gguf"
hf download Qwen/Qwen-Image-2.1 vae/config.json vae/diffusion_pytorch_model.safetensors --local-dir "$SRC/qwen-image-2.1"
hf download taobao-mnn/Qwen3-VL-8B-Instruct-MNN --local-dir "$SRC/qwen3-vl-8b-mnn"

# DiT: Q4_K -> MNN int4 (lossless, block 32), small layers int8.
# --fuse 0: MNN's fused Attention op gives wrong results for the prefix / step masks.
"$PY" "$ROOT/export/export_mnn.py" --gguf "$SRC/gguf/qwen_image_2.1-Q4_K.gguf" --out "$OUT" --aux_bits 8 --fuse 0
# VAE decoder (residual stream rescaled for fp16) + encoder for image editing, fp16 weights, dynamic size
"$PY" "$ROOT/export/export_vae.py" --vae "$SRC/qwen-image-2.1/vae" --out "$OUT" --fp16 --parts decoder,encoder
rm -rf "$OUT/onnx" "$OUT/vae_check.pt" "$OUT/order.txt"

# Text encoder: stock Qwen3-VL-8B MNN files + configs that return the pre-norm last hidden state
for f in llm.mnn llm.mnn.weight embeddings_int4.bin tokenizer.txt llm_config.json visual.mnn visual.mnn.weight; do
    cp "$SRC/qwen3-vl-8b-mnn/$f" "$OUT/text_encoder/$f"
done
"$PY" "$ROOT/export/make_te_config.py" "$OUT/text_encoder"
du -sh "$OUT"
