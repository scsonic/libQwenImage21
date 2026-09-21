# Per-block activation ranges of the DiT (fp32 torch reference, weights streamed from GGUF), to check fp16 headroom.
# usage: python act_stats.py <gguf> <text_hidden.npy [1,L,4096]> [sigma]
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen_image21_mnn as Q
from export_mnn import GGUFWeights

W = GGUFWeights(sys.argv[1])
text = torch.from_numpy(np.load(sys.argv[2])).float()
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
L, N = text.shape[1], 1024
torch.manual_seed(0)
lat = torch.randn(1, N, 64)
cos, sin = Q.rope_tables(L)
dit = Q.DiT(W, W.param)
txt_in, img_in = Q.TxtIn(W, W.param), Q.ImgIn(W)


def run(h, t, c, s, past, mask, tag):
    temb = dit.temb(torch.tensor([t]))
    mod = dit.modulation(torch.nn.functional.silu(temb))
    s1, g1, s2, g2 = torch.split(mod, Q.DIM, dim=-1)
    kv = []
    for i, blk in enumerate(dit.blocks):
        h, pk, pv = blk(h, s1, g1, s2, g2, c, s, past[i, 0:1], past[i, 1:2], mask)
        kv.append(torch.stack([pk[0], pv[0]]))
        print(f"{tag} block {i:2d}: |h|max {h.abs().max().item():10.1f}  rms {h.pow(2).mean().sqrt().item():8.2f}", flush=True)
    return h, torch.stack(kv)


with torch.no_grad():
    _, kv = run(txt_in(text), 0.0, cos[:L], sin[:L], torch.zeros(Q.LAYERS, 2, 1, 32, 128), Q.prefix_mask(L), "text")
    run(img_in(lat), sigma, cos[L:], sin[L:], kv, torch.zeros(1, 1, N, L + N), "image")
