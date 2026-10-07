# Timings, command line, building from source, repository layout

[← back to the main README](../README.md)

## Timings

| 20 steps | text encoder | K/V prefix | DiT | VAE | total |
|---|---|---|---|---|---|
| Text to image, 448×576 | 22 s | 2.7 s (P=58) | 20×19.1 s | 18 s | **451 s** |
| Edit → 352×448 (Fast) | 67 s (+vision) | 17 s (P=672) | 20×12.6 s | 13 s | **348 s** |
| Edit → 448×576 (Standard) | 80 s | 26 s (P=1064) | 20×21.4 s | 22 s | **556 s** |

| Turbo, 6 steps | text encoder + prefix | DiT | VAE | total |
|---|---|---|---|---|
| Text to image, 448×576–512×512 | ~15 s | 6×~22 s ≈ 134 s | ~18 s | **~216–235 s** |
| Edit, 1 reference → 352×448 | ~20 s (+vision, P≈680) | 6×~18 s ≈ 108 s | ~13 s | **~196–200 s** |
| Edit, 2 references (half scale), ~448×576–672×384 | ~70 s (+2× vision, +2× VAE-encode, P~1100) | 6×~27 s ≈ 161 s | ~22 s | **257–275 s** |

An edit step is slower than a text-to-image step at the same size because attention also runs over the condition
image's tokens (P+N keys instead of N); a 2nd reference adds more of the same.

| 20 steps, text to image | Standard (448×576) | Fast (512×288) | Tiny (320×320) |
|---|---|---|---|
| DiT step | 19.1 s | 10.8 s | 7.4 s |
| VAE decode | 18 s, 4.3 GB | 10 s, 2.6 GB | 7 s, 2.0 GB |
| Total | 451 s | 289 s | 217 s |

The text encoder (13–16 s) is a fixed cost at every size; the smaller tiers mainly cut the VAE decode peak, the
stage most likely to be killed on a phone. A smoke test at Pico (128×128) ran ~1.9 s/DiT step on Turbo and
produced a correct if low-detail image — the pipeline has no lower-size floor beyond the 32px-multiple
requirement, but Micro/Nano/Pico aren't quality-evaluated the way Standard/Fast/Tiny are.

## Why MNN and OpenCL?

**MNN** is the only Android runtime found that covers the whole pipeline — int4 LLM engine for the 8B text
encoder, general graph runtime for the DiT/VAE, one OpenCL backend for all three — and its converter is open
enough to hand-build the lossless GGUF→int4 re-pack this project uses.

**OpenCL over Vulkan:** OpenCL is currently the fastest general GPU path on Android. Vulkan compute is assumed
**1.5×+ slower** here (experience with MNN on Qualcomm, not a benchmark of this model) — significant at ~20 s/step.
Vulkan's edge is portability on GPUs without a usable OpenCL driver; MNN has a Vulkan backend if you want to
measure it (`MNN_FORWARD_VULKAN`).

**Not the NPU (yet):** a QNN/HTP graph is tied to a chip generation, so it needs a separate build per SoC, and
only top-tier NPUs clearly beat the same-generation GPU. The text encoder (a plain int4 prefill, 13–16 s of the
total) is the realistic NPU candidate, but the vision tower it needs for image editing can't run there today, so
it would have to fall back to CPU whenever an input image is involved.

So: OpenCL for the DiT, CPU wherever the GPU runs out of memory (currently the VAE for non-default settings),
NPU for the text encoder as future work.

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
