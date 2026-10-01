# One synthetic Q2_K-shaped linear (random q/scale/zero, bits=2 block=16) through the MNN runtime vs. float
# matmul with the dequantized weight. No real GGUF file needed -- this only derisks whether MNN's int2 kernel
# computes the documented affine form correctly, independent of the (already-verified) GGUF bit-unpacking math
# in q2k.py.
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen_image21_mnn as Q, q2k, export_mnn as E
from test_mnn import load, var, to_np, rel

oc, ic = 64, 256  # small, but a multiple of q2k.BLOCK and a plausible Linear shape
rng = np.random.default_rng(0)
q = rng.integers(0, 4, size=(oc, ic)).astype(np.uint8)
nblk = ic // q2k.BLOCK
scale = rng.uniform(0.01, 0.2, size=(oc, nblk)).astype(np.float32)
zero = rng.uniform(-0.3, 0.3, size=(oc, nblk)).astype(np.float32)
ref = q2k.dequant(q, scale, zero)  # the float weight MNN should reconstruct

name = "transformer_blocks.0.attn.to_q"
class P:
    def __call__(self, n): return torch.from_numpy(ref)
    def prequant(self, n): return (q, scale, zero, q2k.BITS, q2k.BLOCK)
class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.l = Q.Linear(name, ic, oc, P())
    def forward(self, x): return self.l(x)
m = M()
out = os.path.join(HERE, "q2k_test"); os.makedirs(out, exist_ok=True)
E.export_onnx(m, (torch.randn(1, 8, ic),), os.path.join(out, "l.onnx"), ["x"], ["y"], {"x": {1: "N"}, "y": {1: "N"}})
E.to_mnn(os.path.join(out, "l.onnx"), os.path.join(out, "l.mnn"), {name: m.l}, lambda n: 2, 16)
x = torch.randn(1, 37, ic)
y = to_np(load(os.path.join(out, "l.mnn"), ["x"], ["y"]).forward([var(x)])[0])
yr = (x @ torch.from_numpy(ref).T).numpy()
print("q2k linear MNN vs float rel/corr", rel(y, yr))
