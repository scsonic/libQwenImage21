# 2-bit DiT (optional): smaller download, not faster

[leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) also publishes a **Q2_K**
quantization of the DiT, alongside the Q4_K one this project already uses for `dit.mnn`. `dit_2bit.mnn` is the
same lossless-repack approach applied to Q2_K: every Q2_K sub-block (`w = d·sc·q − dmin·m`, q in `[0,3]`, 16
weights per sub-block — half of Q4_K's 32) maps exactly onto MNN's asymmetric int2, so the weights are copied
without re-quantization. `export/q2k.py` does the unpacking; verified bit-exact against the `gguf` Python
package's own Q2_K dequantizer, and against real tensors from the actual GGUF file.

[← back to the main README](../README.md)

## The honest result: smaller, not faster

The original motivation was speed — half the bits should mean less memory traffic. It doesn't play out that way
on this runtime: **`dit_2bit.mnn` is 3.6 GB instead of 4.5 GB (20% smaller, not 50% — Q2_K's native block size is
16 instead of Q4_K's 32, so the per-block scale/min overhead eats into the saving), but each DiT step is slightly
*slower*, not faster**, because MNN's OpenCL low-memory int2 kernel takes a different, less-optimized code path
than int4's (int4 gets an image-backed fast path; int2/int3 are buffer-only — see `ConvBufLowMemoryExecution.cpp`).
If your goal is speed, use [Turbo](TURBO.md) (6 fixed steps) instead or in addition; if your goal is a smaller
download/install, 2-bit delivers that honestly.

| | DiT step | Total (text-to-image, 512×512) | File size |
|---|---|---|---|
| int4 (`dit.mnn`) | ~19.1 s | ~451 s (448×576, 20 steps) | 4.5 GB |
| **int2 (`dit_2bit.mnn`)** | ~20.5 s | **495.9 s** (20 steps) | **3.6 GB** |
| int4 + Turbo (`dit_turbo.mnn`) | ~22 s | ~216–235 s (6 steps) | 5.2 GB |
| **int2 + Turbo (`dit_2bit_turbo.mnn`)** | ~24.3 s | **242.1 s** (6 steps) | **4.3 GB** |

*Snapdragon 8 Gen 2, OpenCL, same prompt/seed. int4 row is the existing README/TURBO.md measurement (448×576);
int2 rows measured this session at 512×512 — see the images below for both.*

<p>
<img src="twobit_t2i_coffee.png" width="30%"/>
<img src="twobit_turbo_t2i_coffee.png" width="30%"/>
</p>

*"A cozy coffee shop on a rainy evening, warm light, a sign that reads QWEN", seed 42. Left: int2, 20 steps
(495.9 s). Right: int2 + Turbo, 6 steps (242.1 s). Both coherent and detailed — at 20+ steps int2's quality holds
up well against int4.*

**Step count matters more for int2 than int4.** An early sanity check at only 8 steps (no Turbo) produced a
visibly malformed image — not soft/blurry, but structurally wrong (duplicated/melted shapes) — while the exact
same prompt/seed at 20 steps was clean. This looks like quantization noise that a short schedule doesn't give the
model enough steps to correct, rather than a hard quality ceiling. If you use the base (non-Turbo) int2 model,
stay at something like the app's default 20 steps rather than dropping it to save time.

**int2 + Turbo runs cleanly but is visibly softer than int4 + Turbo.** Turbo's fixed 6-step schedule is a
different, distilled schedule (not the same interpolation cut short), and it does produce a coherent image with
int2 — it isn't broken. But directly comparing int2+Turbo against int4+Turbo on the *same* prompt/seed/size shows
a real, consistent quality gap: int4+Turbo is sharper (crisp object edges, fine texture like wood grain or apple
skin speckling), int2+Turbo is softer with somewhat warped geometry, more so in scenes with lots of straight lines
(a café's window grid) than in simple single-object scenes (an apple, where the gap is subtle). The likely
explanation is the same one as the step-count finding above: Turbo's 6 steps don't give int2 as much room to
correct its quantization noise as the base model's 20 do, and int4 starts from much less quantization error to
begin with.

<p>
<img src="twobit_turbo_coffee_2bit.png" width="24%"/>
<img src="twobit_turbo_coffee_4bit.png" width="24%"/>
<img src="twobit_turbo_apple_2bit.png" width="24%"/>
<img src="twobit_turbo_apple_4bit.png" width="24%"/>
</p>

*Same prompt/seed/size (512×512, 6 Turbo steps) for each pair. Coffee shop: int2 (1st) vs int4 (2nd) — the
window grid and furniture are noticeably crisper in int4. Apple: int2 (3rd) vs int4 (4th) — both correct, int4 has
finer wood-grain and skin-speckle detail. If Turbo speed matters more than the last bit of sharpness, int2+Turbo
is usable; if you want Turbo's speed *and* int4's sharpness, that's just int4+Turbo (5.2 GB) — there's no way to
get both the 2-bit size and int4 sharpness today.*

## Two MNN bugs found and fixed along the way

Both are in the runtime fork (`MNN/`), not in this repo's export scripts — GGUF Q2_K with MNN's own int2 format
via an external weight file, at block size 16, had apparently never been exercised through MNN's OpenCL
low-memory path before:

- **Use-after-free crash on load.** `ConvBufLowMemoryExecution::getInfoFromOpLowMemory()` got the decoded weight
  buffer from a local `shared_ptr<ConvolutionCommon::Int8Common>` and stashed a raw pointer to it
  (`mFilterDataPtr`) for a caller to read *after* the function returned — by which point, for the 2/3-bit path
  specifically (4-bit and 8-bit alias an already-persistent buffer instead), the local `shared_ptr` had gone out
  of scope and freed it. Fixed by keeping the `shared_ptr` alive in a member
  (`ConvBufLowMemoryExecution.hpp`/`.cpp`).
- **Wrong dequant offset → NaN after the crash fix.** The int2/int3 OpenCL unpack kernels multiply the *raw*
  stored code (0..3 for int2) by `scale` and add `offset`, so `offset` has to already have the centering term
  folded in (`offset = zero − (1 << (bits−1)) · scale)` — the kernel's own comment says as much. The C++ transfer
  loop only did this folding for `mNumQuantBit == 4`; 2-bit and 3-bit got the raw `zero` instead, a per-weight
  error that compounded through 32 transformer blocks into `NaN`/`Inf`. Fixed by extending the existing int4
  special case to int2/int3 too.

## Using it

Export (after `scripts/build_models.sh`'s usual Q4_K-based `dit.mnn`/`dit_turbo.mnn`):

```bash
python export/export_mnn.py --gguf models/gguf/qwen_image_2.1-Q2_K.gguf --out models/qwen_image21 \
    --aux_bits 8 --block 64 --fuse 0 --only dit --dit_name dit_2bit.mnn
python export/export_mnn.py --gguf models/gguf/qwen_image_2.1-Q2_K.gguf --out models/qwen_image21 \
    --aux_bits 8 --block 64 --fuse 0 --only dit --dit_name dit_2bit_turbo.mnn \
    --lora models/lora/viggle-turbo-r128.safetensors
```

Library: `Options.dit2Bit = true` (independent of `Options.turbo` — the two combine to select one of
`dit.mnn` / `dit_turbo.mnn` / `dit_2bit.mnn` / `dit_2bit_turbo.mnn`). CLI: 18th arg to `qwen_image21_demo`. The
demo app has a **2-bit DiT** checkbox next to **Turbo LoRA**, and matching download checkboxes.

## Status

Verified end-to-end on the test phone (Snapdragon 8 Gen 2, OpenCL): both `dit_2bit.mnn` and
`dit_2bit_turbo.mnn` load and generate correctly after the two MNN fixes above. Not yet evaluated on image
editing (only text-to-image) or on other GPUs/vendors, where the now-fixed OpenCL low-memory int2 path may
behave differently.
