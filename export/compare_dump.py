# Compare C++ pipeline dumps (QWEN_IMAGE21_DUMP) with the torch reference DiT (Q4_K weights dequantized).
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen_image21_mnn as Q
from export_mnn import GGUFWeights
D = sys.argv[2]
ld = lambda n, shape: np.fromfile(os.path.join(D, n + ".f32"), np.float32).reshape(shape)
W = GGUFWeights(sys.argv[1])
cache = {}
def wp(name):
    if name not in cache:
        cache[name] = W(name).to(torch.bfloat16)
    return cache[name]
def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return f"rel {np.linalg.norm(a-b)/np.linalg.norm(b):.4f} corr {np.corrcoef(a,b)[0,1]:.5f} |ref|max {np.abs(b).max():.3f} |mnn|max {np.abs(a).max():.3f}"
text = torch.from_numpy(ld("text_hidden", (1, -1, 4096)))
L, N = text.shape[1], 1024
cos, sin = Q.rope_tables(L)
with torch.no_grad():
    txt_h = Q.TxtIn(W, W.param)(text)
    print("txt_h    ", rel(ld("txt_h", (1, L, 4096)), txt_h))
    dit = Q.DiT(wp, W.param)
    kv = list(dit(txt_h, torch.tensor([0.0]), cos[:L], sin[:L], Q.prefix_mask(L),
                  *[torch.zeros(2, 1, 32, 128) for _ in range(32)])[1:])
    mkv = [ld(f"prefix_kv_{l}", tuple(kv[l].shape)) for l in range(32)]
    for l in (0, 15, 31):
        print(f"prefix_kv layer {l} K", rel(mkv[l][0], kv[l][0]), " V", rel(mkv[l][1], kv[l][1]))
    noise = torch.from_numpy(ld("noise", (1, N, 64)))
    v0 = dit(Q.ImgIn(W)(noise), torch.tensor([1.0]), cos[L:], sin[L:], torch.zeros(1, 1, N, L + N), *kv)[0]
    print("v0       ", rel(ld("v0", (1, N, 64)), v0))
    # same step but from MNN's own KV: isolates the step pass
    v0b = dit(Q.ImgIn(W)(noise), torch.tensor([1.0]), cos[L:], sin[L:], torch.zeros(1, 1, N, L + N),
              *[torch.from_numpy(k) for k in mkv])[0]
    print("v0 (mnn kv)", rel(ld("v0", (1, N, 64)), v0b))
    np.save(os.path.join(D, "ref_kv.npy"), np.stack([k.numpy() for k in kv]))
