# Turbo mode (experimental): 6 steps instead of 20–40

[Viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) is a LoRA that distills Qwen-Image-2.1
down to a fixed 6-step schedule with no CFG. It's applied **unmerged**: the int4 base weights in `dit.mnn` are
untouched, and `dit_turbo.mnn` is a separate model file that adds the LoRA's rank-128 correction as a small extra
fp16 branch alongside each targeted layer (attention, image-MLP, modulation, timestep embedder) — the same way
diffusers and the LoRA's own ComfyUI node apply it, chosen over merging the correction into the int4 weights because
the LoRA's own measurements show merging into even bf16 (let alone int4) measurably hurts fidelity. `txt_in.mnn`,
`img_in.mnn`, `vae_decoder.mnn`, `vae_encoder.mnn` and the text encoder are all identical to the base pipeline and
shared with it — only `dit_turbo.mnn` is new (+ ~700 MB over `dit.mnn`'s 4.5 GB, for the LoRA's own weights).

[← back to the main README](../README.md)

## Text to image

<p>
<img src="turbo_t2i_studio.png" width="30%"/>
<img src="turbo_t2i_kimono.png" width="30%"/>
<img src="turbo_t2i_harajuku.png" width="30%"/>
</p>

*Studio portrait (448×576, 234 s) · furisode kimono under cherry blossoms (448×576, 219 s) · Harajuku street
fashion (512×512, 231 s). All on a Snapdragon 8 Gen 2, OpenCL, 6 steps.*

## Image editing

The left portrait above (`turbo_t2i_studio.png`) is the input; only the prompt changes.

<p>
<img src="turbo_edit_yukata.png" width="30%"/>
<img src="turbo_edit_seifuku.png" width="30%"/>
<img src="turbo_edit_office.png" width="30%"/>
</p>

*Summer festival yukata (352×448, 200 s) · school uniform (352×448, 196 s) · office wear (352×448, 198 s). Same
person, same settings otherwise as [Sizes → Fast](../README.md#sizes) in the main README.*

## Measured on a Snapdragon 8 Gen 2 (OpenCL), this batch

| | text encoder + prefix | DiT (6 steps) | VAE | total |
|---|---|---|---|---|
| Text to image, 448×576 | ~15 s | 6 × ~22.3 s ≈ 134 s | ~18 s | **~220–235 s** |
| Image edit → 352×448 | ~20 s (prefix P≈680) | 6 × ~18 s ≈ 108 s | ~13 s | **~196–200 s** |

A turbo DiT step (~22 s at 448×576) is slower than a base-model step at the same size (~19 s) — the extra fp16 LoRA
branch costs roughly the 10–25% diffusers/ComfyUI itself measures — but going from 20 steps to 6 still cuts total
time by about half for text-to-image and edit alike, since the fixed text-encoder/VAE cost is a larger share of a
now-much-shorter run. See [Why MNN and OpenCL?](../README.md#why-mnn-and-opencl) for the base pipeline's numbers.

## Using it

Not wired into the demo app's UI yet. From the CLI, append `1` as a 15th argument:

```bash
adb shell "cd /data/local/tmp/qwen && LD_LIBRARY_PATH=. ./qwen_image21_demo \
    <model_dir> /sdcard/out.png 'A red apple on a wooden table' 20 42 opencl 0 1 448x576 low 4 1 '' 1"
#                                                                                              ^turbo=1
```

`setTurbo(true)` forces the step count to 6 regardless of what's passed in and loads `dit_turbo.mnn` instead of
`dit.mnn`; everything else (size, edit mode, memory checks) works the same as the base model. Needs
`dit_turbo.mnn` + `.weight` in the model directory (not yet in the default download — see the model repo for now).

**Status:** validated on-device (numeric check against the PyTorch reference, correlation 0.9999 on the DiT's
velocity output; the samples above). Not yet the default — that's the plan once it's wired into the app and README
claims are updated to match.
