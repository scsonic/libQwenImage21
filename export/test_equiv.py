# Check export/qwen_image21_mnn.py against the diffusers transformer (random weights, few layers).
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
import qwen_image21_mnn as Q
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel, QwenImage21KVCache

torch.manual_seed(0)
LAYERS = 2
ref = QwenImage21Transformer2DModel(num_layers=LAYERS).eval()
with torch.no_grad():
    for n, p in ref.named_parameters():
        p.copy_(torch.randn_like(p) * (0.02 if p.dim() > 1 else 0.1))
sd = {k: v.detach() for k, v in ref.state_dict().items()}
wp = lambda name: sd[name + ".weight"]
params = lambda name: sd[name]
mine = Q.DiT(wp, params, layers=LAYERS).eval()
txt_in = Q.TxtIn(wp, params).eval()
img_in = Q.ImgIn(wp).eval()

L, H, W = 23, 32, 32
N = H * W
text = torch.randn(1, L, 4096)
lat = torch.randn(1, N, 64)
sigma = 0.7
img_mask = torch.cat([torch.zeros(1, L, dtype=torch.bool), torch.ones(1, N // 4, dtype=torch.bool)], 1)

with torch.no_grad():
    cache = QwenImage21KVCache(LAYERS)
    out_ref = ref(hidden_states=lat, encoder_hidden_states=text, timestep=torch.tensor([sigma]),
                  img_shapes=[[(1, H, W)]], img_mask=img_mask, kv_cache=cache, kv_cache_mode="extract",
                  return_dict=False)[0][:, -N:]
    cos, sin = Q.rope_tables(L, H, W)
    # prefix
    _, present = mine(txt_in(text), torch.tensor([0.0]), cos[:L], sin[:L],
                      torch.zeros(LAYERS, 2, 1, 32, 128), Q.prefix_mask(L))
    kref = cache.get_layer(1).k
    print("prefix K max err", (present[1, 0:1] - kref).abs().max().item(), "ref max", kref.abs().max().item())
    out, _ = mine(img_in(lat), torch.tensor([sigma]), cos[L:], sin[L:], present, torch.zeros(1, 1, N, L + N))
    print("step out max err", (out - out_ref).abs().max().item(), "ref max", out_ref.abs().max().item())

# scheduler vs diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
import numpy as np, json
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "../models/Qwen-Image-2.1/scheduler/scheduler_config.json")))
cfg.pop("_class_name"); cfg.pop("_diffusers_version")
sch = FlowMatchEulerDiscreteScheduler(**cfg)
steps = 20
m = (0.9 - 0.5) / (8192 - 256); mu = 1024 * m + 0.5 - m * 256
sch.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
print("sigma err", np.abs(np.array(sch.sigmas.tolist()) - np.array(Q.sigmas_schedule(steps))).max())
print("sigmas", [round(float(s), 4) for s in sch.sigmas])
