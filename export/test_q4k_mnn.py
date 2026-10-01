# One Q4_K linear through the MNN runtime vs. float matmul with the dequantized weight.
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen_image21_mnn as Q, q4k, export_mnn as E
from test_mnn import load, var, to_np, rel
S = sys.argv[1]
raw = np.load(os.path.join(S, "toq_raw.npy")); ref = np.load(os.path.join(S, "toq_ref.npy"))
oc, ic = ref.shape
name = "transformer_blocks.0.attn.to_q"
class P:
    def __call__(self, n): return torch.from_numpy(ref)
    def prequant(self, n): return q4k.q4k_decode(raw, oc, ic) + (q4k.BITS, q4k.BLOCK)
class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.l = Q.Linear(name, ic, oc, P())
    def forward(self, x): return self.l(x)
m = M()
out = os.path.join(S, "q4k_test"); os.makedirs(out, exist_ok=True)
E.export_onnx(m, (torch.randn(1, 8, ic),), os.path.join(out, "l.onnx"), ["x"], ["y"], {"x": {1: "N"}, "y": {1: "N"}})
E.to_mnn(os.path.join(out, "l.onnx"), os.path.join(out, "l.mnn"), {name: m.l}, lambda n: 4, 64)
x = torch.randn(1, 37, ic)
y = to_np(load(os.path.join(out, "l.mnn"), ["x"], ["y"]).forward([var(x)])[0])
yr = (x @ torch.from_numpy(ref).T).numpy()
print("q4k linear MNN vs float rel/corr", rel(y, yr))
