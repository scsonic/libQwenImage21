# Export TAEQI2.1 (https://huggingface.co/madebyollin/taeqi2_1) -- a tiny distilled VAE for Qwen-Image-2.1 by
# madebyollin (github.com/madebyollin/taesd) -- to MNN, with the *same I/O contract as export_vae.py's real VAE*
# so it's a drop-in replacement at runtime:
#   decoder: latent [1, 64, h, w] (DiT-facing latent, unnormalized -- see below) -> image [1, 4, 16h, 16w] RGBA [-1, 1]
#   encoder: image  [1, 4, H, W] RGBA in [-1, 1] -> latent [1, 64, H/16, W/16]
#
# Unlike the real VAE, TAEQI2.1 has latents_mean=0 / latents_std=1 (it was distilled to match the DiT-facing latent
# directly, not the VAE's own pre-normalization latent), so there's no mean/std buffer to carry -- only the RGBA
# value-range conversion the real wrappers also do ([-1,1] <-> [0,1]).
#
# Architecture copied verbatim from github.com/madebyollin/taesd/blob/main/taesd.py (MIT), F16Encoder/F16Decoder.
# usage: python export_vae_tiny.py --weights taeqi2_1.safetensors --out models/mnn
import argparse, os, subprocess
import torch
import torch.nn as nn
import safetensors.torch as stt

MNNCONVERT = os.environ.get("MNNCONVERT", "mnnconvert")


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def f16_encoder(latent_channels=64, image_channels=4):
    return nn.Sequential(
        nn.PixelUnshuffle(2), conv(image_channels * 4, 64), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 128, stride=2, bias=False), Block(128, 128), Block(128, 128), Block(128, 128),
        conv(128, 256, stride=2, bias=False), Block(256, 256), Block(256, 256), Block(256, 256),
        conv(256, latent_channels),
    )


def f16_decoder(latent_channels=64, image_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels * 4), nn.PixelShuffle(2),
    )


def convert_diffusers_sd_to_taesd(sd):
    """Same remap as the HF model card's wrapper: '{enc|dec}.layers.N.suffix' -> '{enc|dec}.N[+1 for dec].suffix'
    (the decoder's own Sequential starts with a parameter-less Clamp() at index 0, hence the +1)."""
    out = {}
    for k, v in sd.items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 1 if encdec == "decoder" else 0
        out[".".join([encdec, str(int(index) + offset), *suffix])] = v
    return out


class TinyVaeDecoder(nn.Module):
    """latent [1, 64, h, w] -> image [1, 4, 16h, 16w] RGBA in [-1, 1] -- same contract as export_vae.py:VaeDecoder."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent):
        return (self.decoder(latent) * 2 - 1).clamp(-1, 1)


class TinyVaeEncoder(nn.Module):
    """image [1, 4, H, W] RGBA in [-1, 1] -> latent [1, 64, H/16, W/16] -- same contract as export_vae.py:VaeEncoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        return self.encoder(image * 0.5 + 0.5)


def convert(onnx_path, mnn_path, fp16):
    args = [MNNCONVERT, "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", mnn_path, "--bizCode", "qwen_image21"]
    args.append("--fp16" if fp16 else "--saveExternalData")
    print(" ".join(args))
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to taeqi2_1.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--parts", default="decoder,encoder")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)

    sd = convert_diffusers_sd_to_taesd(stt.load_file(a.weights))
    enc_sd = {k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")}
    dec_sd = {k[len("decoder."):]: v.float() for k, v in sd.items() if k.startswith("decoder.")}
    encoder, decoder = f16_encoder(), f16_decoder()
    encoder.load_state_dict(enc_sd)
    decoder.load_state_dict(dec_sd)
    encoder.eval()
    decoder.eval()

    lh = a.size // 16
    lat = torch.randn(1, 64, lh, lh)
    img = torch.rand(1, 4, a.size, a.size) * 2 - 1
    dec, enc = TinyVaeDecoder(decoder).eval(), TinyVaeEncoder(encoder).eval()
    parts = a.parts.split(",")
    with torch.no_grad():
        if "decoder" in parts:
            out = dec(lat)
            print("decoder output range", out.min().item(), out.max().item(), "shape", out.shape)
            p = os.path.join(a.out, "onnx", "vae_decoder_tiny.onnx")
            torch.onnx.export(dec, (lat,), p, input_names=["latent"], output_names=["image"],
                              dynamic_axes={"latent": {2: "h", 3: "w"}, "image": {2: "H", 3: "W"}},
                              opset_version=17, dynamo=False, do_constant_folding=True)
            convert(p, os.path.join(a.out, "vae_decoder_tiny.mnn"), a.fp16)
        if "encoder" in parts:
            out = enc(img)
            print("encoder output range", out.min().item(), out.max().item(), "shape", out.shape)
            p = os.path.join(a.out, "onnx", "vae_encoder_tiny.onnx")
            torch.onnx.export(enc, (img,), p, input_names=["image"], output_names=["latent"],
                              dynamic_axes={"image": {2: "H", 3: "W"}, "latent": {2: "h", 3: "w"}},
                              opset_version=17, dynamo=False, do_constant_folding=True)
            convert(p, os.path.join(a.out, "vae_encoder_tiny.mnn"), a.fp16)


if __name__ == "__main__":
    main()
