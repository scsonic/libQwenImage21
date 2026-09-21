#!/bin/bash
# Push a model directory to the demo app (and the CLI demo if built).
#   scripts/push_models.sh [model_dir] [adb serial]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/models/qwen_image21}"
A="adb ${2:+-s $2}"
DST=/sdcard/Android/data/com.scsonic.qwenimage21.demo/files/qwen_image21
$A shell mkdir -p $DST/text_encoder
cd "$SRC"
for f in $(find . -type f ! -name "*.part" ! -path "./.cache/*" | sed 's|^\./||' | sort); do
    [ "$f" = "README.md" ] || [ "$f" = "LICENSE" ] || [ "$f" = ".gitattributes" ] && continue
    lsz=$(wc -c < "$f" | tr -d ' ')
    rsz=$($A shell "stat -c %s $DST/$f 2>/dev/null" | tr -d '\r' || true)
    if [ "$lsz" = "$rsz" ]; then echo "skip $f"; continue; fi
    echo "push $f ($((lsz / 1000000)) MB)"
    $A push "$f" "$DST/$f" > /dev/null
done
CLI="$ROOT/third_party/MNN/build_android_cli"
if [ -x "$CLI/qwen_image21_demo" ]; then
    $A shell mkdir -p /data/local/tmp/qwen
    $A push "$CLI/libMNN.so" "$CLI/qwen_image21_demo" /data/local/tmp/qwen/ > /dev/null
fi
echo "done -> $DST"
