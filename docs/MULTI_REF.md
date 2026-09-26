# Multi-reference editing: 2 condition images instead of 1

Qwen-Image-2.1 natively supports up to 10 reference images per edit (order fixes `<image1>`, `<image2>`, ... in the
prompt). This runtime supports **1 or 2**: each reference becomes its own attention block in the joint prefix —
bidirectional within itself, causal across blocks and to the target — exactly matching
`QwenImage21Transformer2DModel.build_token_metadata` / `QwenImage21Rope.forward` in diffusers, generalized from the
single-reference layout in `export/qwen_image21_mnn.py:edit_layout`. No new model files: it's a runtime change only,
verified against the actual diffusers transformer (random weights, `export/test_edit2_equiv.py`) before ever
touching the phone.

[← back to the main README](../README.md)

## Shrinking references to save RAM/time

A reference's own resolution doesn't have to match the output: each one is independently resized (own aspect ratio
kept) to a configurable fraction of the output's pixel area before it's encoded. The *output*'s own size/aspect is
unaffected — it's always taken from the **last** reference, at full area, matching Qwen-Image-2.1's own convention
when no explicit `height`/`width` is given.

- **Full** (1.0×, the original single-reference behaviour): each reference costs as much as the output itself. Two
  references at Full roughly **doubles** the prefix length, and with it the DiT prefix pass, the K/V cache, and
  (a little) each denoising step.
- **Half** (0.5×): each reference's *area* — not side length — is halved, so two references together cost about as
  much as **one** reference at Full does today. This is the safer default with two references on a memory-tight
  phone.

This isn't a hypothetical: on the 16 GB test phone, two Full-scale 512²-area references pushed the prefix to ~2100
tokens and the run was killed by the system outright (`lowmemorykiller`, "device is not responding") before it
could even report an `OUT_OF_MEMORY` exception. The same edit with both references at Half completed normally.

## Example

<p>
<img src="turbo_t2i_studio.png" width="30%"/>
<img src="sample_lighthouse_576x448.png" width="36%"/>
<img src="turbo_edit_2ref_lighthouse.png" width="30%"/>
</p>

*Reference 1 (a generated portrait) + reference 2 (a photo of a lighthouse at sunset) → "Place the woman from image
1 standing on the rocky shore in front of the lighthouse from image 2, sunset sky, keep her face and outfit
unchanged." Turbo, 6 steps, both references at Half scale, 576×448 output: **256.6 s** on a Snapdragon 8 Gen 2 —
about 25–50 s more than a comparable single-reference turbo edit, mostly a second vision-encoder + VAE-encoder pass
and a longer (but still Half-scale) prefix.*

## Using it

**Demo app:** the Image Edit tab has a **+ 2nd reference (optional)** button under the first image, with its own
preview and a **Clear** button; leaving it empty behaves exactly as before. A **Shrink each reference to half
size** checkbox controls Full vs. Half (applies to 1 reference too, though the RAM saving only matters with 2).

**Library:**

```java
Bitmap b = qi.edit("Combine the two references as described",
                   inputImage1, inputImage2,          // inputImage2 may be null
                   QwenImage21.Size.Tier.STANDARD, 20, 42, outPng, listener);
```

`QwenImage21.Options.refSize` is `RefSize.FULL` (default) or `RefSize.HALF`.

**CLI:** two more trailing arguments after `turbo`:

```bash
adb shell "cd /data/local/tmp/qwen && LD_LIBRARY_PATH=. ./qwen_image21_demo \
    <model_dir> /sdcard/out.png 'Combine the two references as described' 6 42 opencl 0 1 512x512 low 4 1 \
    /sdcard/ref1.png 1 /sdcard/ref2.png 0.5"
#                                       ^input_image2         ^ref_area_scale
```

`ref_area_scale` (default `1.0`) applies to every reference passed, including a single one.

## Status

Verified two ways: (1) the RoPE/mask/prefix construction (`export/qwen_image21_mnn.py:edit_layout_n`) matches the
actual diffusers `QwenImage21Transformer2DModel` to float precision on random weights, including an exact reduction
to the original single-reference `edit_layout` when given one image; (2) on-device, real trained weights, the
sample above. Not yet the default UI state for a second reference (it stays opt-in) since it hasn't seen the same
breadth of real-world use as single-reference editing.
