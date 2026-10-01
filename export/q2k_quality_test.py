# Quality sanity check: run the full DiT denoising loop in plain PyTorch with Q4_K-dequantized weights vs
# Q2_K-dequantized weights (same real text embedding + noise, reused from an earlier dump), decode both with
# the real VAE, and save side by side. No MNN involved -- this only asks "is Q2_K's precision loss acceptable
# for this model", independent of whether the MNN int2 kernel is correct (already verified separately).
import os, sys, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_image21_mnn as Q
from export_mnn import GGUFWeights

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--dump", default=os.path.join(HERE, "../outputs/dump"))
ap.add_argument("--out", default=os.path.join(HERE, "../outputs/q2k_quality"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ld = lambda n, shape: torch.from_numpy(np.fromfile(os.path.join(a.dump, n + ".f32"), np.float32).reshape(shape))
text = ld("text_hidden", (1, -1, 4096))
noise = ld("noise", (1, -1, 64))
L, N = text.shape[1], noise.shape[1]
H = W = int(N ** 0.5)
print(f"L={L} N={N} H=W={H}")

cos, sin = Q.rope_tables(L, H, W)
sigmas = Q.sigmas_schedule(a.steps)


def run(gguf_path, tag):
    t0 = time.time()
    W_ = GGUFWeights(gguf_path)
    with torch.no_grad():
        txt_in = Q.TxtIn(W_, W_.param).eval()
        img_in = Q.ImgIn(W_).eval()
        dit = Q.DiT(W_, W_.param).eval()
        txt_h = txt_in(text)
        kv = list(dit(txt_h, torch.tensor([0.0]), cos[:L], sin[:L], Q.prefix_mask(L),
                      *[torch.zeros(2, 1, 32, 128) for _ in range(32)])[1:])
        lat = noise.clone()
        for i in range(a.steps):
            sigma = sigmas[i]
            v = dit(img_in(lat), torch.tensor([sigma]), cos[L:], sin[L:],
                    torch.zeros(1, 1, N, L + N), *kv)[0]
            lat = lat + (sigmas[i + 1] - sigma) * v
            print(f"  [{tag}] step {i+1}/{a.steps} sigma={sigma:.4f} {time.time()-t0:.1f}s", flush=True)
    np.save(os.path.join(a.out, f"lat_{tag}.npy"), lat.numpy())
    print(f"[{tag}] done in {time.time()-t0:.1f}s")
    return lat


lat_q2k = run(os.path.join(HERE, "../models/gguf/qwen_image_2.1-Q2_K.gguf"), "2bit")
lat_q4k = run(os.path.join(HERE, "../models/gguf/qwen_image_2.1-Q4_K.gguf"), "4bit")

# decode both with the real VAE
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(HERE, "../models/Qwen-Image-2.1/vae"),
                                                torch_dtype=torch.float32).eval()
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1).float()
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1).float()


def decode(lat, tag):
    img = lat.float().reshape(1, H, W, 64).permute(0, 3, 1, 2)
    with torch.no_grad():
        px = vae.decode((img * std + mean).unsqueeze(2), return_dict=False)[0][:, :, 0]
    px = (px.clamp(-1, 1) + 1) / 2 * 255
    arr = px[0].permute(1, 2, 0).byte().numpy()
    from PIL import Image
    Image.fromarray(arr[..., :3]).save(os.path.join(a.out, f"{tag}_{a.steps}step.png"))
    print("saved", tag)


decode(lat_q2k, "2bit")
decode(lat_q4k, "4bit")

d = (lat_q2k - lat_q4k).abs()
print(f"latent diff: max {d.max():.4f} mean {d.mean():.4f}  (4bit latent max {lat_q4k.abs().max():.4f})")
