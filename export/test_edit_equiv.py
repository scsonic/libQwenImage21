# Edit (one condition image) path of export/qwen_image21_mnn.py vs. the diffusers transformer, random weights.
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
import qwen_image21_mnn as Q
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel, QwenImage21KVCache

torch.manual_seed(0)
LAYERS = 2
ref = QwenImage21Transformer2DModel(num_layers=LAYERS).eval()
with torch.no_grad():
    for n, p in ref.named_parameters():
        p.copy_(torch.randn_like(p) * (0.05 if p.dim() > 1 else 0.3))   # large enough that blocks matter
sd = {k: v.detach() for k, v in ref.state_dict().items()}
wp, params = (lambda n: sd[n + ".weight"]), (lambda n: sd[n])
dit, txt_in, img_in = Q.DiT(wp, params, layers=LAYERS).eval(), Q.TxtIn(wp, params).eval(), Q.ImgIn(wp).eval()

hc, wc, H, W = 12, 16, 16, 12
t1, t2 = 5, 9
is_pad = [False] * t1 + [True] * (hc * wc // 4) + [False] * t2
text = torch.randn(1, len(is_pad), 4096)
cond, lat = torch.randn(1, hc * wc, 64), torch.randn(1, H * W, 64)
img_mask = torch.tensor([is_pad + [True] * (H * W // 4)])
sigma = 0.6
with torch.no_grad():
    cache = QwenImage21KVCache(LAYERS)
    out_ref = ref(hidden_states=torch.cat([cond, lat], 1), encoder_hidden_states=text, timestep=torch.tensor([sigma]),
                  img_shapes=[[(1, hc, wc), (1, H, W)]], img_mask=img_mask, kv_cache=cache, kv_cache_mode="extract",
                  return_dict=False)[0][:, -H * W:]
    t1_, t2_, cos, sin, mask = Q.edit_layout(is_pad, hc, wc, H, W)
    th = txt_in(text)
    keep = torch.tensor([not p for p in is_pad])
    thk = th[:, keep]
    prefix = torch.cat([thk[:, :t1], img_in(cond), thk[:, t1:]], 1)
    P = prefix.shape[1]
    kv = list(dit(prefix, torch.tensor([0.0]), cos[:P], sin[:P], mask,
                  *[torch.zeros(2, 1, 32, 128) for _ in range(LAYERS)])[1:])
    print("prefix K err", (kv[1, 0:1] - cache.get_layer(1).k).abs().max().item(), "ref", cache.get_layer(1).k.abs().max().item())
    out = dit(img_in(lat), torch.tensor([sigma]), cos[P:], sin[P:], torch.zeros(1, 1, H * W, P + H * W), *kv)[0]
    print("step out err", (out - out_ref).abs().max().item(), "ref", out_ref.abs().max().item())
