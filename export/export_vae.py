# Export the Qwen-Image-2.1 VAE decoder (single image) to MNN.
# input  : latent [1, 64, h, w]   (normalized DiT latents, unpacked to NCHW)
# output : image  [1, 4, 16h, 16w] RGBA in [-1, 1]
# usage: python export_vae.py --vae models/Qwen-Image-2.1/vae --out models/mnn [--random]
import argparse, json, os, subprocess, sys
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21, QwenImage21AttentionBlock, QwenImage21Resample, QwenImage21ResidualBlock)

MNNCONVERT = os.environ.get("MNNCONVERT", "mnnconvert")


@torch.no_grad()
def rescale_residual_stream(decoder, s):
    """Divide the decoder's residual stream by `s` so it fits fp16 (it peaks at ~3.5e5 otherwise).

    Every residual branch starts with a scale-invariant RMS norm and norm_out precedes conv_out, so scaling the
    stream only requires scaling what writes into it: conv_in, each branch's last conv, the attention proj, and the
    biases of the linear convs on the stream (shortcut / upsample). The decoded image is unchanged.
    """
    def scale(conv, weight=True):
        if weight:
            conv.weight.div_(s)
        if conv.bias is not None:
            conv.bias.div_(s)
    scale(decoder.conv_in)
    for m in decoder.modules():
        if isinstance(m, QwenImage21ResidualBlock):
            scale(m.conv2)
            if not isinstance(m.conv_shortcut, nn.Identity):
                scale(m.conv_shortcut, weight=False)
        elif isinstance(m, QwenImage21AttentionBlock):
            scale(m.proj)
        elif isinstance(m, QwenImage21Resample) and m.mode.startswith("upsample"):
            scale(m.resample[1], weight=False)


def _fp16_safe_rms_norm(self, x):
    # normalize(x) == normalize(x / max|x|); the pre-division keeps the sum of squares <= C in fp16
    amax = x.abs().amax(dim=1, keepdim=True) + 1e-6
    n = torch.nn.functional.normalize(x / amax, dim=1)
    return n * self.scale * self.gamma + self.bias


class VaeDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        cfg = vae.config
        self.register_buffer("mean", torch.tensor(cfg.latents_mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.latents_std).view(1, -1, 1, 1))

    def forward(self, latent):
        z = (latent * self.std + self.mean).unsqueeze(2)
        self.vae.clear_cache()
        x = self.vae.post_quant_conv(z) if getattr(self.vae, "post_quant_conv", None) is not None else z
        out = self.vae.decoder(x, feat_cache=self.vae._feat_map, feat_idx=[0], first_chunk=True)
        self.vae.clear_cache()
        return torch.clamp(out[:, :, 0], -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="store conv weights as fp16")
    ap.add_argument("--scale", type=float, default=256.0, help="residual stream divisor for fp16 headroom")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.random:
        cfg = json.load(open(os.path.join(a.vae, "config.json")))
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        vae = AutoencoderKLQwenImage21(**cfg)
    else:
        vae = AutoencoderKLQwenImage21.from_pretrained(a.vae, torch_dtype=torch.float32)
    vae.eval()
    # nearest-exact == nearest for integer x2 scaling, and ONNX only knows the latter
    for m in vae.modules():
        if isinstance(m, nn.Upsample) and m.mode == "nearest-exact":
            m.mode = "nearest"
    dec = VaeDecoder(vae).eval()
    lh = a.size // 16
    lat = torch.randn(1, 64, lh, lh)
    with torch.no_grad():
        ref = vae.decode((lat * dec.std + dec.mean).unsqueeze(2), return_dict=False)[0][:, :, 0]
    peak = []
    hooks = [m.register_forward_hook(lambda m, i, o: peak.append(o.abs().max().item()))
             for m in vae.decoder.modules() if isinstance(m, (nn.Conv2d,))]
    rescale_residual_stream(vae.decoder, a.scale)
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21RMS_norm
    QwenImage21RMS_norm.forward = _fp16_safe_rms_norm
    with torch.no_grad():
        mine = dec(lat)
    for h in hooks:
        h.remove()
    print("wrapper vs vae.decode max err", (ref - mine).abs().max().item(), tuple(mine.shape),
          "peak activation", max(peak))
    torch.save({"lat": lat, "img": mine}, os.path.join(a.out, "vae_check.pt"))
    onnx_path = os.path.join(a.out, "onnx", "vae_decoder.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(dec, (lat,), onnx_path, input_names=["latent"], output_names=["image"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    mnn_path = os.path.join(a.out, "vae_decoder.mnn")
    args = [MNNCONVERT, "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", mnn_path, "--bizCode", "qwen_image21"]
    # fp16 weights are kept inside the .mnn: the fp16 + --saveExternalData combination fails to load.
    args.append("--fp16" if a.fp16 else "--saveExternalData")
    print(" ".join(args))
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-1500:])


if __name__ == "__main__":
    main()
