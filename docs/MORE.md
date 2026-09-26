# Command line, building from source, repository layout

[← back to the main README](../README.md)

## Command line (adb)

```bash
ANDROID_NDK=/path/to/ndk scripts/build_libmnn_android.sh     # also builds the CLI
scripts/push_models.sh models/qwen_image21
adb shell "cd /data/local/tmp/qwen && LD_LIBRARY_PATH=. ./qwen_image21_demo \
    /sdcard/Android/data/com.scsonic.qwenimage21.demo/files/qwen_image21 /sdcard/out.png \
    'A red apple on a wooden table' 20 42 opencl 0 1 576x448 low 4 1"
# image edit: append the input image
adb shell "cd /data/local/tmp/qwen && LD_LIBRARY_PATH=. ./qwen_image21_demo <model_dir> /sdcard/edit.png \
    'Make it winter, snow on the ground' 20 42 opencl 0 1 512 low 4 1 /sdcard/input.jpg"
```

`<model_dir> <out.png> <prompt> [steps=20] [seed=42] [opencl|cpu] [memory_mode=0] [te_on_cpu=1] [size=512|WxH]
[precision=low|normal|high] [threads=4] [vae_on_cpu=0] [input_image] [turbo=0] [input_image2] [ref_area_scale=1.0]`.
In edit mode `size` is the pixel budget only (output keeps the *last* reference's ratio). `turbo=1` loads
`dit_turbo.mnn` and forces `steps` to 6 — see [docs/TURBO.md](TURBO.md). `input_image2` adds a 2nd reference
(optional) and `ref_area_scale` shrinks each reference's area (e.g. `0.5`) — see [docs/MULTI_REF.md](MULTI_REF.md).
`QWEN_IMAGE21_DUMP=<dir>` dumps intermediate tensors (`export/compare_dump.py`).

## Building from source

| Step | Command |
|---|---|
| libMNN.so for the AAR (+ CLI) | `ANDROID_NDK=… scripts/build_libmnn_android.sh` |
| APK / AAR | `./gradlew :demo:assembleDebug :qwenimage21:assembleRelease` |
| Host MNNConvert | `scripts/build_mnnconvert_mac.sh` |
| Re-convert the models | `pip install -r export/requirements.txt && scripts/convert_models.sh` |

A prebuilt `libMNN.so` and headers are checked in under `qwenimage21/src/main/`, so the APK/AAR build needs no NDK
work and no `third_party/MNN` checkout (that submodule is only for rebuilding `libMNN.so` itself). CI builds it this
way on every push — see [`.github/workflows/build.yml`](../.github/workflows/build.yml).

Verification scripts in `export/`: `test_equiv.py` (re-implementation vs. diffusers), `test_edit_equiv.py` /
`test_edit2_equiv.py` (1- and 2-reference edit layout vs. diffusers), `test_mnn.py` (MNN vs. torch),
`compare_dump.py` (on-device dumps vs. torch).

## Repository layout

| Path | |
|---|---|
| `qwenimage21/` | Android library: `QwenImage21`, `ModelDownloader`, JNI, prebuilt `libMNN.so` + headers |
| `demo/` | Demo app |
| `export/` | Model conversion + verification (Python) |
| `scripts/` | Build, convert and push scripts |
| `third_party/MNN` | MNN fork with the Qwen-Image-2.1 pipeline (submodule) |
