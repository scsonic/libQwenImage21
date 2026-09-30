# Tiny VAE (experimental): TAEQI2.1, ~215x faster decode, ~30 MB instead of ~660 MB

[TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1) is a tiny distilled autoencoder by
[madebyollin](https://github.com/madebyollin/taesd) — a handful of convolution layers trained to match
Qwen-Image-2.1's real VAE at its own "latent API" (16x spatial compression, 64 latent channels, RGBA). Because it
was distilled to match the real VAE's DiT-facing latent directly (not the VAE's own pre-normalization latent), it's
a **drop-in replacement**: same tensor names, shapes and value ranges as `vae_decoder.mnn` / `vae_encoder.mnn`, so
nothing else in the pipeline changes.

[← back to the main README](../README.md)

## Why this matters on a phone

The real VAE decoder is ~0.5 GB and needs ~4.3 GB of runtime memory at 512² — enough that it exhausted GPU memory
and rebooted the test phone (see [Limitations](../README.md#limitations)), so it's always run on CPU today, at
~19.5 s/decode. TAEQI2.1 is ~15 MB per direction (~30 MB total, vs ~660 MB for the real encoder+decoder), runs
**on the GPU** with no memory concerns, and decodes in well under a tenth of a second once the shader cache is
warm.

## Measured on-device (Snapdragon 8 Gen 2, Adreno 740)

Four prompts, each run twice — real VAE (forced onto CPU, as today) vs. TAEQI2.1 (on GPU) — same seed, same Turbo
6-step DiT, same everything else. `qwen_image21_demo` CLI, `a8a03486`.

| Prompt | Real VAE decode | Real VAE RAM | Real total | Tiny VAE decode | Tiny VAE RAM | Tiny total | Result |
|---|---|---|---|---|---|---|---|
| "A red apple on a wooden table, simple studio lighting" | 19.63 s | 4326 MB | 220.9 s | 0.57 s¹ | 187 MB | 202.8 s | [real](vae_cmp_p1_real.png) · [tiny](vae_cmp_p1_tiny.png) |
| "A cat sitting on a windowsill at sunset" | 19.54 s | 4326 MB | 226.0 s | 0.09 s | 187 MB | 202.1 s | [real](vae_cmp_p2_real.png) · [tiny](vae_cmp_p2_tiny.png) |
| "A futuristic city skyline at night with neon lights" | 19.54 s | 4326 MB | 225.9 s | 0.09 s | 187 MB | 203.5 s | [real](vae_cmp_p3_real.png) · [tiny](vae_cmp_p3_tiny.png) |
| "A bowl of ramen noodles on a wooden table, steam rising" | 19.54 s | 4326 MB | 229.4 s | 0.09 s | 187 MB | 207.5 s | [real](vae_cmp_p4_real.png) · [tiny](vae_cmp_p4_tiny.png) |
| **Average** | **19.56 s** | **4326 MB** | **225.6 s** | **0.09–0.57 s** | **187 MB** | **204.0 s** | |

¹ First run after install pays a one-time OpenCL shader-compile cost (`.mnn_cl_cache` didn't exist yet); every run
after that was 0.09 s.

**Decode: ~217x faster, ~23x less RAM** (4326 MB → 187 MB, a ~4.1 GB saving). **End-to-end: ~21.6 s faster per
generation** even though decode is a small slice of a 6-step Turbo run's total time — most of the ~204 s left is
the DiT steps and model loading, unchanged either way. The saving is much larger, proportionally, against the base
20-40 step model (same fixed ~19.5 s decode either way) and against edit mode's VAE *encode* step, which this
model also replaces (`vae_encoder_tiny.mnn`).

<p>
<img src="vae_cmp_p1_tiny.png" width="24%"/>
<img src="vae_cmp_p2_tiny.png" width="24%"/>
<img src="vae_cmp_p3_tiny.png" width="24%"/>
<img src="vae_cmp_p4_tiny.png" width="24%"/>
</p>

*All four Tiny-VAE outputs above. Side by side with the real-VAE version of each, the differences are in fine
grain/texture, not composition, sharpness or color — see the linked pairs in the table.*

## Using it

**CLI:** append `1` as the 17th argument (and `vae_on_cpu=0`, since TAEQI2.1 is meant to run on GPU):

```bash
adb shell "cd /data/local/tmp/qwen && LD_LIBRARY_PATH=. ./qwen_image21_demo \
    <model_dir> /sdcard/out.png 'A red apple on a wooden table' 20 42 opencl 0 1 512x512 low 4 0 '' 0 '' 1.0 1"
#                                                                                          ^vae_on_cpu=0        ^tiny_vae=1
```

Needs `vae_decoder_tiny.mnn` and `vae_encoder_tiny.mnn` in the model directory (not part of the default download
yet — see [Files](../README.md#files) once this ships in the app).

## How it was made

`export/export_vae_tiny.py` rebuilds TAEQI2.1's exact architecture (copied from
[taesd.py](https://github.com/madebyollin/taesd/blob/main/taesd.py)'s `F16Encoder`/`F16Decoder`, MIT-licensed) in
plain PyTorch, loads `taeqi2_1.safetensors` with the same diffusers-key remap the model card's own wrapper uses,
then wraps it in the same RGBA-range conversions ([-1,1] in/out) as `export_vae.py`'s real-VAE wrappers so the
exported ONNX/MNN graphs are call-compatible with the existing runtime. Verified two ways before touching the
phone: (1) the state dict loads `strict=True` into the reimplemented architecture (confirms it's byte-for-byte the
published one); (2) decoding the *real* VAE encoder's actual latents with the MNN-converted tiny decoder reproduces
the input photo essentially unchanged (encoder max abs error 0.0013, decoder 0.0028, against a ±4.5 / [-1,1] range
— fp16-export-level noise, not an architecture mismatch).

## Status

Verified numerically (see above) and on-device (this page). Not yet wired into the demo app's UI or the default
download — this page is the standalone CLI results; app integration is the next step.
