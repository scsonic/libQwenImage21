# Qwen-Image-2.1 DiT re-implemented for MNN export.
#
# The diffusers transformer is split into pieces that suit an on-device runtime:
#   txt_in.mnn  : text features  [1, L, 4096] -> [1, L, 4096]
#   img_in.mnn  : packed latents [1, N, 64]   -> [1, N, 4096]
#   dit.mnn     : 32 single-stream blocks + norm_out/proj_out
#                 inputs : hidden [1,N,4096], timestep [1], rope_cos/rope_sin [N,64],
#                          attn_mask [1,1,N,P+N], past_kv_0..31 [2,P,32,128]
#                 outputs: out [1,N,64] (velocity), present_kv_0..31 [2,N,32,128]
#
# Prefix pass (once per prompt): hidden = txt_in(text), timestep = 0, P = 1 dummy key masked out,
#   causal mask; read `present_kv` -> text KV cache.
# Step pass (every denoising step): hidden = img_in(latents), timestep = sigma, past_kv = text KV cache,
#   zero mask; read `out`.
# This is exactly the diffusers `kv_cache_mode="cached"` path, because text never attends to the image.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 4096
HEADS = 32
HEAD_DIM = 128
LAYERS = 32
MLP = DIM * 3
IN_CH = 64
EPS = 1e-6


class Linear(nn.Module):
    """Linear that is either real (reference run) or a FakeLinear placeholder (MNN export)."""

    export_mode = False

    def __init__(self, name, ic, oc, weight_provider):
        super().__init__()
        self.name = name
        self.in_features = ic
        self.out_features = oc
        self.bias = None
        self._provider = weight_provider

    @property
    def weight(self):
        # MNNConverter.build_weight reads `linear.weight.data`; load lazily so only one layer lives in RAM.
        w = self._provider(self.name)
        assert tuple(w.shape) == (self.out_features, self.in_features), (self.name, w.shape)
        return nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        if Linear.export_mode:
            from utils.custom_op import FakeLinearOp
            return FakeLinearOp.apply(x, self.in_features, self.out_features, False, self.name)
        return F.linear(x, self._provider(self.name).to(x.dtype))


class LoRALinear(nn.Module):
    """A base Linear (FakeLinear-exported, quantized as usual) plus an additive LoRA branch: two small real
    Linears (never quantized -> fp16 in build_weight's bits==16 path) that are exported like any other
    Q.Linear, so no new ONNX/MNN machinery is needed. forward = base(x) + B(A(x))."""

    def __init__(self, name, ic, oc, wp, lora):
        super().__init__()
        self.base = Linear(name, ic, oc, wp)
        r = lora.rank(name)
        self.lora_a = Linear(f"{name}.lora_A", ic, r, lora)
        self.lora_b = Linear(f"{name}.lora_B", r, oc, lora)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x))


def linear(name, ic, oc, wp, lora=None):
    """Linear, or LoRALinear if `lora` targets this layer name."""
    if lora is not None and lora.has(name):
        return LoRALinear(name, ic, oc, wp, lora)
    return Linear(name, ic, oc, wp)


def rms_norm(x, w, eps=EPS):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def layer_norm(x, eps=EPS):
    return F.layer_norm(x, (DIM,), eps=eps)


def apply_rope(x, cos, sin):
    # x: [1, N, H, D]; interleaved complex pairs (x[2i], x[2i+1]); cos/sin: [N, D/2]
    h, d = HEADS, HEAD_DIM
    x = x.reshape(1, -1, h, d // 2, 2)
    xr, xi = x[..., 0], x[..., 1]
    c = cos.reshape(1, -1, 1, d // 2)
    s = sin.reshape(1, -1, 1, d // 2)
    out = torch.stack([xr * c - xi * s, xr * s + xi * c], dim=-1)
    return out.reshape(1, -1, h, d)


class Block(nn.Module):
    def __init__(self, i, wp, params, lora=None):
        super().__init__()
        p = f"transformer_blocks.{i}"
        self.to_q = linear(f"{p}.attn.to_q", DIM, DIM, wp, lora)
        self.to_k = linear(f"{p}.attn.to_k", DIM, DIM, wp, lora)
        self.to_v = linear(f"{p}.attn.to_v", DIM, DIM, wp, lora)
        self.to_out = linear(f"{p}.attn.to_out.0", DIM, DIM, wp, lora)
        self.gate = linear(f"{p}.img_mlp.gate_layer", DIM, MLP, wp, lora)
        self.proj = linear(f"{p}.img_mlp.proj", DIM, MLP, wp, lora)
        self.out = linear(f"{p}.img_mlp.out", MLP, DIM, wp, lora)
        self.norm_q = nn.Parameter(params(f"{p}.attn.norm_q.weight"), requires_grad=False)
        self.norm_k = nn.Parameter(params(f"{p}.attn.norm_k.weight"), requires_grad=False)

    def forward(self, h, scale1, gate1, scale2, gate2, cos, sin, past_k, past_v, mask):
        x = layer_norm(h) * (1 + scale1)
        q = self.to_q(x).reshape(1, -1, HEADS, HEAD_DIM)
        k = self.to_k(x).reshape(1, -1, HEADS, HEAD_DIM)
        v = self.to_v(x).reshape(1, -1, HEADS, HEAD_DIM)
        q = apply_rope(rms_norm(q, self.norm_q), cos, sin)
        k = apply_rope(rms_norm(k, self.norm_k), cos, sin)
        present_k, present_v = k, v
        k_all = torch.cat([past_k, k], dim=1).permute(0, 2, 3, 1)  # [1,H,D,P+N]
        v_all = torch.cat([past_v, v], dim=1).permute(0, 2, 1, 3)  # [1,H,P+N,D]
        qh = q.permute(0, 2, 1, 3)  # [1,H,N,D]
        scores = torch.matmul(qh, k_all) * (1.0 / math.sqrt(HEAD_DIM)) + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), v_all)  # [1,H,N,D]
        attn = attn.permute(0, 2, 1, 3).reshape(1, -1, DIM)
        h = h + torch.tanh(gate1) * self.to_out(attn)
        x = layer_norm(h) * (1 + scale2)
        h = h + torch.tanh(gate2) * self.out(F.silu(self.gate(x)) * self.proj(x))
        return h, present_k, present_v


class DiT(nn.Module):
    def __init__(self, wp, params, layers=LAYERS, lora=None):
        super().__init__()
        self.t_lin1 = linear("time_text_embed.timestep_embedder.linear_1", 256, DIM, wp, lora)
        self.t_lin2 = linear("time_text_embed.timestep_embedder.linear_2", DIM, DIM, wp, lora)
        self.modulation = linear("modulation.1", DIM, 4 * DIM, wp, lora)
        self.norm_out = Linear("norm_out.linear", DIM, DIM, wp)
        self.proj_out = Linear("proj_out", DIM, IN_CH, wp)
        self.blocks = nn.ModuleList([Block(i, wp, params, lora) for i in range(layers)])
        half = 128
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half))

    def temb(self, timestep):
        args = (1000.0 * timestep).reshape(1, 1) * self.freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.t_lin2(F.silu(self.t_lin1(emb.reshape(1, 1, 256))))  # [1,1,DIM]

    def forward(self, hidden, timestep, rope_cos, rope_sin, attn_mask, *past_kv):
        # One [2,P,HEADS,HEAD_DIM] cache per layer rather than one [LAYERS,2,P,...] tensor: a single buffer would be
        # P MiB, and OpenCL refuses one larger than CL_DEVICE_MAX_MEM_ALLOC_SIZE (1 GiB on Adreno 740).
        temb = self.temb(timestep)
        mod = self.modulation(F.silu(temb))  # [1,1,4*DIM]
        scale1, gate1, scale2, gate2 = torch.split(mod, DIM, dim=-1)
        h = hidden
        presents = []
        for i, blk in enumerate(self.blocks):
            h, pk, pv = blk(h, scale1, gate1, scale2, gate2, rope_cos, rope_sin,
                            past_kv[i][0:1], past_kv[i][1:2], attn_mask)
            presents.append(torch.stack([pk[0], pv[0]], dim=0))
        scale = self.norm_out(F.silu(temb))
        out = self.proj_out(layer_norm(h) * (1 + scale))
        return (out, *presents)


class TxtIn(nn.Module):
    def __init__(self, wp, params):
        super().__init__()
        self.norm_w = nn.Parameter(params("txt_in.text_norm.weight") + 1.0, requires_grad=False)
        self.in_layer = Linear("txt_in.in_layer", DIM, DIM, wp)
        self.out_layer = Linear("txt_in.out_layer", DIM, DIM, wp)

    def forward(self, txt):
        x = rms_norm(txt, self.norm_w)
        return self.out_layer(F.gelu(self.in_layer(x), approximate="tanh"))


class ImgIn(nn.Module):
    def __init__(self, wp):
        super().__init__()
        self.img_in = Linear("img_in", IN_CH, DIM, wp)

    def forward(self, lat):
        return self.img_in(lat)


# ---------------------------------------------------------------- host-side helpers (mirrored in C++)

def rope_tables(text_len, h_tokens=32, w_tokens=32, axes=(16, 56, 56), theta=10000.0):
    """cos/sin [text_len + h*w, 64] following QwenImage21Rope for a text-to-image layout."""
    def freqs(pos, dim):
        inv = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        return torch.outer(pos.float(), inv)

    ti = torch.arange(text_len)
    hh = torch.arange(-(h_tokens - h_tokens // 2), h_tokens // 2).repeat_interleave(w_tokens)
    ww = torch.arange(-(w_tokens - w_tokens // 2), w_tokens // 2).repeat(h_tokens)
    fi = torch.full((h_tokens * w_tokens,), text_len)
    frame = torch.cat([ti, fi])
    hpos = torch.cat([ti, hh])
    wpos = torch.cat([ti, ww])
    ang = torch.cat([freqs(frame, axes[0]), freqs(hpos, axes[1]), freqs(wpos, axes[2])], dim=-1)
    return torch.cos(ang), torch.sin(ang)


def prefix_mask(n):
    m = torch.full((n, n + 1), -30000.0)
    m[:, 1:] = torch.triu(torch.full((n, n), -30000.0), diagonal=1)
    return m.reshape(1, 1, n, n + 1)


def sigmas_schedule(steps, image_seq_len=1024, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9,
                    shift_terminal=0.02):
    import numpy as np
    s = np.linspace(1.0, 1.0 / steps, steps)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    mu = image_seq_len * m + base_shift - m * base_seq
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    one_minus = 1.0 - s
    scale = one_minus[-1] / (1.0 - shift_terminal)
    s = 1.0 - one_minus / scale
    return list(s) + [0.0]


def edit_layout(is_pad, hc, wc, H, W, axes=(16, 56, 56), theta=10000.0):
    """Image-conditioned (edit) layout, mirroring QwenImage21Rope / block-causal mask for one condition image.

    is_pad: bool list over the text-encoder tokens (after dropping the system prompt), True at <|image_pad|>
            (hc*wc/4 slots, each standing for 2x2 latent tokens).
    Returns (t1, t2, cos, sin, prefix_mask) where the joint prefix is [t1 text][hc*wc condition latents][t2 text],
    cos/sin cover prefix + target (H*W) tokens, and prefix_mask is [1,1,P,P+1] with a leading dummy key.
    """
    first = is_pad.index(True)
    nslots = sum(is_pad)
    assert nslots * 4 == hc * wc and all(is_pad[first:first + nslots])
    t1 = first
    t2 = len(is_pad) - first - nslots
    nc = hc * wc
    P = t1 + nc + t2

    def grid(h, w):
        hh = torch.arange(-(h - h // 2), h // 2).repeat_interleave(w)
        ww = torch.arange(-(w - w // 2), w // 2).repeat(h)
        return hh, ww

    ch, cw = grid(hc, wc)
    th, tw = grid(H, W)
    f_t1 = torch.arange(t1)
    f_c = torch.full((nc,), t1)
    start2 = t1 + max(hc, wc)
    f_t2 = torch.arange(start2, start2 + t2)
    f_tg = torch.full((H * W,), start2 + t2)
    frame = torch.cat([f_t1, f_c, f_t2, f_tg])
    hpos = torch.cat([f_t1, ch, f_t2, th])
    wpos = torch.cat([f_t1, cw, f_t2, tw])

    def freqs(pos, dim):
        inv = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        return torch.outer(pos.float(), inv)

    ang = torch.cat([freqs(frame, axes[0]), freqs(hpos, axes[1]), freqs(wpos, axes[2])], dim=-1)
    block = torch.cat([torch.full((t1,), -1), torch.zeros(nc, dtype=torch.long), torch.full((t2,), -1)])
    q = torch.arange(P)[:, None]
    k = torch.arange(P)[None, :]
    allowed = (k <= q) | ((block[:, None] == block[None, :]) & (block[:, None] >= 0))
    m = torch.full((P, P + 1), -30000.0)
    m[:, 1:][allowed] = 0.0
    return t1, t2, torch.cos(ang), torch.sin(ang), m.reshape(1, 1, P, P + 1)
