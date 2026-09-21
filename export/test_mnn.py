# Run exported MNN pieces with pymnn and compare against the torch reference implementation.
# usage: python test_mnn.py <mnn_dir> [--random_sd random_sd.pt | --gguf file.gguf] [--layers N]
import argparse, os, sys, time
import numpy as np
import torch
import MNN
import MNN.expr as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen_image21_mnn as Q


def load(path, inputs, outputs, backend="CPU", precision="high"):
    rt = MNN.nn.create_runtime_manager(({"backend": backend, "precision": precision, "numThread": 8},))
    return MNN.nn.load_module_from_file(path, inputs, outputs, runtime_manager=rt, shape_mutable=True, rearrange=True)


def var(t):
    a = np.ascontiguousarray(t.detach().float().numpy())
    return F.const(a, list(a.shape), F.NCHW)


def to_np(v):
    return np.array(F.convert(v, F.NCHW).read(), dtype=np.float32)


def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12), np.corrcoef(a, b)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--random_sd")
    ap.add_argument("--gguf")
    ap.add_argument("--layers", type=int, default=Q.LAYERS)
    ap.add_argument("--L", type=int, default=23)
    ap.add_argument("--precision", default="high")
    a = ap.parse_args()
    if a.random_sd:
        sd = torch.load(a.random_sd)
        wp, params = (lambda n: sd[n + ".weight"]), (lambda n: sd[n])
    else:
        from export_mnn import GGUFWeights
        W = GGUFWeights(a.gguf)
        wp, params = W, W.param

    torch.manual_seed(1)
    L, N = a.L, 1024
    text = torch.randn(1, L, 4096)
    lat = torch.randn(1, N, 64)
    cos, sin = Q.rope_tables(L)
    sigma = 0.7

    with torch.no_grad():
        ref_txt = Q.TxtIn(wp, params)(text)
        ref_img = Q.ImgIn(wp)(lat)
        dit = Q.DiT(wp, params, layers=a.layers)
        _, ref_kv = dit(ref_txt, torch.tensor([0.0]), cos[:L], sin[:L],
                        torch.zeros(a.layers, 2, 1, 32, 128), Q.prefix_mask(L))
        ref_out, _ = dit(ref_img, torch.tensor([sigma]), cos[L:], sin[L:], ref_kv, torch.zeros(1, 1, N, L + N))

    m = load(os.path.join(a.dir, "txt_in.mnn"), ["txt"], ["txt_h"], precision=a.precision)
    txt_h = to_np(m.forward([var(text)])[0])
    print("txt_in  rel/corr", rel(txt_h, ref_txt))
    m = load(os.path.join(a.dir, "img_in.mnn"), ["lat"], ["img_h"], precision=a.precision)
    img_h = to_np(m.forward([var(lat)])[0])
    print("img_in  rel/corr", rel(img_h, ref_img))

    names = ["hidden", "timestep", "rope_cos", "rope_sin", "past_kv", "attn_mask"]
    pre = load(os.path.join(a.dir, "dit.mnn"), names, ["present_kv"], precision=a.precision)
    t0 = time.time()
    kv = pre.forward([var(ref_txt), var(torch.tensor([0.0])), var(cos[:L]), var(sin[:L]),
                      var(torch.zeros(a.layers, 2, 1, 32, 128)), var(Q.prefix_mask(L))])[0]
    kv = to_np(kv).reshape(ref_kv.shape)
    print(f"prefix  rel/corr {rel(kv, ref_kv)}  {time.time()-t0:.2f}s")
    step = load(os.path.join(a.dir, "dit.mnn"), names, ["out"], precision=a.precision)
    t0 = time.time()
    out = step.forward([var(ref_img), var(torch.tensor([sigma])), var(cos[L:]), var(sin[L:]),
                        var(ref_kv), var(torch.zeros(1, 1, N, L + N))])[0]
    out = to_np(out)
    print(f"step    rel/corr {rel(out, ref_out)}  {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
