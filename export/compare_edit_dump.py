# Compare the C++ edit pipeline dumps with the torch reference (VAE encoder + DiT with the edit layout).
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen_image21_mnn as Q
from export_mnn import GGUFWeights
from PIL import Image
gguf, D, img_path, vae_dir = sys.argv[1:5]
ld = lambda n: np.fromfile(os.path.join(D, n + ".f32"), np.float32)
def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return f"rel {np.linalg.norm(a-b)/np.linalg.norm(b):.4f} corr {np.corrcoef(a,b)[0,1]:.5f}"
is_pad = [bool(x) for x in ld("edit_is_pad")]
L = len(is_pad)
text = torch.from_numpy(ld("edit_text_hidden").reshape(1, L, 4096))
H = W = 32
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
im = np.array(Image.open(img_path).convert("RGBA")).astype(np.float32)
im[..., 3] = 255
x = torch.from_numpy(im).permute(2, 0, 1)[None] / 127.5 - 1
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
with torch.no_grad():
    lat = (vae.encode(x.unsqueeze(2)).latent_dist.mode()[:, :, 0] - mean) / std
cond_ref = lat.reshape(1, 64, -1).transpose(1, 2)
cond = torch.from_numpy(ld("edit_cond").reshape(1, -1, 64))
print("vae encode", rel(cond, cond_ref))
W8 = GGUFWeights(gguf); cache = {}
wp = lambda n: cache.setdefault(n, W8(n).to(torch.bfloat16))
dit, txt_in, img_in = Q.DiT(wp, W8.param), Q.TxtIn(W8, W8.param), Q.ImgIn(W8)
t1, t2, cos, sin, mask = Q.edit_layout(is_pad, H, W, H, W)
with torch.no_grad():
    th = txt_in(text)
    keep = torch.tensor([not p for p in is_pad]); thk = th[:, keep]
    prefix = torch.cat([thk[:, :t1], img_in(cond), thk[:, t1:]], 1)
    P = prefix.shape[1]
    _, kv = dit(prefix, torch.tensor([0.0]), cos[:P], sin[:P], torch.zeros(32, 2, 1, 32, 128), mask)
    mkv = ld("prefix_kv").reshape(kv.shape)
    print("prefix kv", rel(mkv, kv), " layer0", rel(mkv[0], kv[0]), " layer31", rel(mkv[31], kv[31]))
    noise = torch.from_numpy(ld("noise").reshape(1, -1, 64))
    v0, _ = dit(img_in(noise), torch.tensor([1.0]), cos[P:], sin[P:], kv, torch.zeros(1, 1, H * W, P + H * W))
    print("v0", rel(ld("v0"), v0))
    print("t1", t1, "t2", t2, "text max", text.abs().max().item(), "pad hidden max", text[0, torch.tensor(is_pad)].abs().max().item())
