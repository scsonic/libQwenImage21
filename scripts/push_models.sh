#!/bin/bash
# Push a model directory to the demo app (and the CLI demo if built).
#   scripts/push_models.sh [model_dir] [adb serial]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/models/qwen_image21}"
A="adb ${2:+-s $2}"
DST=/sdcard/Android/data/com.scsonic.qwenimage21.demo/files/qwen_image21
$A shell mkdir -p $DST/text_encoder
# adb shell creates these owned by shell:ext_data_rw, mode rwxrws---. On devices where the app process
# isn't in the ext_data_rw group (observed on a Snapdragon 8 Gen 2 test board), "other" gets zero access
# and the app can't even traverse the directory, so every file looks missing though it's right there.
$A shell chmod 755 $DST $DST/text_encoder
cd "$SRC"
for f in $(find -L . -type f ! -name "*.part" ! -path "./.cache/*" | sed 's|^\./||' | sort); do
    [ "$f" = "README.md" ] || [ "$f" = "LICENSE" ] || [ "$f" = ".gitattributes" ] && continue
    lsz=$(wc -c < "$f" | tr -d ' ')
    rsz=$($A shell "stat -c %s $DST/$f 2>/dev/null" | tr -d '\r' || true)
    if [ "$lsz" = "$rsz" ]; then echo "skip $f"; continue; fi
    echo "push $f ($((lsz / 1000000)) MB)"
    $A shell "mkdir -p \"\$(dirname '$DST/$f')\""
    # Not `adb push`: on some devices (seen on a Snapdragon 8 Gen 2 test board) it fails with
    # "remote fchown failed: Operation not permitted" for anything under Android/data and leaves no file
    # at all. Piping through `cat` writes the file as the shell user with no chown call.
    cat "$f" | $A shell "cat > '$DST/$f'"
    $A shell "chmod 644 '$DST/$f'"
    rsz=$($A shell "stat -c %s $DST/$f 2>/dev/null" | tr -d '\r' || true)
    [ "$lsz" = "$rsz" ] || { echo "incomplete transfer: $f ($rsz of $lsz bytes)"; exit 1; }
done
CLI="$ROOT/third_party/MNN/build_android_cli"
if [ -x "$CLI/qwen_image21_demo" ]; then
    $A shell mkdir -p /data/local/tmp/qwen
    $A push "$CLI/libMNN.so" "$CLI/qwen_image21_demo" /data/local/tmp/qwen/ > /dev/null
fi
echo "done -> $DST"
