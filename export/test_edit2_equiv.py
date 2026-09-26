# Edit (two condition images) path of export/qwen_image21_mnn.py:edit_layout_n vs. the diffusers transformer,
# random weights. Generalizes test_edit_equiv.py to check the 2-reference block-causal mask and RoPE frame
# bookkeeping (each reference its own attention block, bidirectional inside, causal across).
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
import qwen_image21_mnn as Q
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel, QwenImage21KVCache

torch.manual_seed(0)
LAYERS = 2
ref = QwenImage21Transformer2DModel(num_layers=LAYERS).eval()
with torch.no_grad():
    for n, p in ref.named_parameters():
        p.copy_(torch.randn_like(p) * (0.05 if p.dim() > 1 else 0.3))
sd = {k: v.detach() for k, v in ref.state_dict().items()}
wp, params = (lambda n: sd[n + ".weight"]), (lambda n: sd[n])
dit, txt_in, img_in = Q.DiT(wp, params, layers=LAYERS).eval(), Q.TxtIn(wp, params).eval(), Q.ImgIn(wp).eval()

# Two references of different sizes (as condSize() would give two different-aspect / different-scale images),
# a short text gap between them (<|vision_end|><|vision_start|>), text before and after, and the target image.
hc1, wc1 = 12, 16
hc2, wc2 = 10, 10
H, W = 16, 12
t0, tgap, t2 = 5, 2, 9
is_pad = [False] * t0 + [True] * (hc1 * wc1 // 4) + [False] * tgap + [True] * (hc2 * wc2 // 4) + [False] * t2
text = torch.randn(1, len(is_pad), 4096)
cond1, cond2 = torch.randn(1, hc1 * wc1, 64), torch.randn(1, hc2 * wc2, 64)
lat = torch.randn(1, H * W, 64)
img_mask = torch.tensor([is_pad + [True] * (H * W // 4)])
sigma = 0.6
with torch.no_grad():
    cache = QwenImage21KVCache(LAYERS)
    out_ref = ref(hidden_states=torch.cat([cond1, cond2, lat], 1), encoder_hidden_states=text,
                  timestep=torch.tensor([sigma]), img_shapes=[[(1, hc1, wc1), (1, hc2, wc2), (1, H, W)]],
                  img_mask=img_mask, kv_cache=cache, kv_cache_mode="extract", return_dict=False)[0][:, -H * W:]

    P, cos, sin, mask = Q.edit_layout_n(is_pad, [(hc1, wc1), (hc2, wc2)], H, W)
    th = txt_in(text)
    keep = torch.tensor([not p for p in is_pad])
    thk = th[:, keep]  # t0 + tgap + t2 rows, in order
    prefix = torch.cat([thk[:, :t0], img_in(cond1), thk[:, t0:t0 + tgap], img_in(cond2), thk[:, t0 + tgap:]], 1)
    assert prefix.shape[1] == P, (prefix.shape[1], P)
    kv = list(dit(prefix, torch.tensor([0.0]), cos[:P], sin[:P], mask,
                  *[torch.zeros(2, 1, 32, 128) for _ in range(LAYERS)])[1:])
    out = dit(img_in(lat), torch.tensor([sigma]), cos[P:], sin[P:], torch.zeros(1, 1, H * W, P + H * W), *kv)[0]
    print("step out err", (out - out_ref).abs().max().item(), "ref", out_ref.abs().max().item())

# Sanity: edit_layout_n with a single image must reproduce edit_layout's own (cos, sin, mask) exactly.
hc, wc = 12, 16
is_pad1 = [False] * 5 + [True] * (hc * wc // 4) + [False] * 9
t1_, t2_, cos1, sin1, mask1 = Q.edit_layout(is_pad1, hc, wc, H, W)
P1, cos1n, sin1n, mask1n = Q.edit_layout_n(is_pad1, [(hc, wc)], H, W)
assert P1 == t1_ + hc * wc + t2_
print("single-image reduction: cos", (cos1 - cos1n).abs().max().item(), "sin", (sin1 - sin1n).abs().max().item(),
      "mask", (mask1 - mask1n).abs().max().item())
