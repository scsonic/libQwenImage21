# Export Qwen-Image-2.1 DiT pieces to MNN with block-wise int4/int8 weights.
#
# Linear layers are exported to ONNX as `FakeLinear` custom ops (no weights), converted with MNNConvert,
# and then rebuilt into quantized Convolution ops whose weights are streamed layer by layer into the
# external .weight file (same scheme as MNN's llmexport). Peak RAM stays at one layer's weights.
#
# usage: python export_mnn.py --gguf models/gguf/qwen_image_2.1-Q8_0.gguf --out models/mnn
import argparse, copy, json, os, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../ref/llmexport"))
import qwen_image21_mnn as Q
from utils.torch_utils import quant as torch_quant

MNNCONVERT = os.environ.get("MNNCONVERT", "mnnconvert")


# ---------------------------------------------------------------- weights

class GGUFWeights:
    """Lazy dequantizing reader. Keys follow diffusers names; `img_mlp.gate_up` is split into gate/proj."""

    def __init__(self, path, gate_first=True):
        import gguf
        from gguf import quants
        self._quants = quants
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.gate_first = gate_first

    def _load(self, name):
        t = self.tensors[name]
        data = self._quants.dequantize(t.data, t.tensor_type)
        shape = [int(x) for x in reversed(t.shape.tolist())]
        return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32).reshape(shape))

    def _q4k(self, key):
        import q4k
        t = self.tensors[key]
        if int(t.tensor_type) != 12:  # GGML_TYPE_Q4_K
            return None
        ic, oc = [int(x) for x in t.shape.tolist()]
        return q4k.q4k_decode(np.asarray(t.data).reshape(oc, -1), oc, ic)

    def prequant(self, name):
        """Q4_K linears -> (q, scale, zero) at block 32, copied losslessly into MNN int4; None otherwise."""
        if name.endswith(".img_mlp.gate_layer") or name.endswith(".img_mlp.proj"):
            r = self._q4k(name.rsplit(".", 1)[0] + ".gate_up.weight")
            if r is None:
                return None
            half = r[0].shape[0] // 2
            first = name.endswith("gate_layer") == self.gate_first
            sl = slice(0, half) if first else slice(half, None)
            return tuple(np.ascontiguousarray(x[sl]) for x in r)
        return self._q4k(name + ".weight")

    def __call__(self, name):
        # Linear weight provider: name without ".weight"
        if name.endswith(".img_mlp.gate_layer") or name.endswith(".img_mlp.proj"):
            base = name.rsplit(".", 1)[0]
            w = self._load(base + ".gate_up.weight")
            a, b = w.chunk(2, dim=0)
            is_gate = name.endswith("gate_layer")
            return (a if is_gate == self.gate_first else b).contiguous()
        return self._load(name + ".weight")

    def param(self, name):
        return self._load(name)


# ---------------------------------------------------------------- MNN conversion helpers

def run(args):
    print(" ".join(args), flush=True)
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError("mnnconvert failed")
    return r.stdout


class Rebuilder:
    def __init__(self, weight_ops, bits_for, block, sym=False, hqq=False, scale16=True):
        self.scale16 = scale16
        self.weight_ops = weight_ops
        self.bits_for = bits_for
        self.block = block
        self.sym = sym
        self.hqq = hqq

    def write(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(data, list):
            data = np.array(data).astype(np.float32)
        return self.f.write(data.tobytes())

    def write_header(self, ic, oc, bits):
        n = self.f.write(b"\x02")
        dt = np.int32 if (oc > 65535 or ic > 65535) else np.int16
        n2 = self.write(np.array([oc, ic]).astype(dt))
        off = 1 << (bits - 1)
        wm = list(range(-off, off))
        if len(wm) == 256:
            wm.insert(0, 0)
        else:
            wm.insert(0, len(wm))
        n3 = self.write(np.array(wm, dtype=np.int8))
        return n + n2 + n3, dt == np.int32

    def build_weight(self, lin, bits):
        ic, oc = lin.in_features, lin.out_features
        pre = getattr(lin._provider, "prequant", None)
        pre = pre(lin.name) if pre is not None else None
        self.last_prequant = pre is not None
        if pre is not None:
            import q4k
            packed, alpha = q4k.mnn_pack(*pre)
            hl, int32 = self.write_header(ic, oc, 4)
            wl = self.write(packed) + hl
            al = self.write(alpha.astype(np.float16) if self.scale16 else alpha)
            ext = [self.offset, wl, al, 0, 0]
            self.offset += wl + al
            return ext, int32, 4, 32
        w = lin.weight.data.float()
        block = self.block if ic % self.block == 0 else 0
        if bits == 16:
            wl = self.write(w.flatten().half())
            ext = [self.offset, wl, 0, 0, 0]
            self.offset += wl
            return ext, False, 16, block
        q, alpha = torch_quant(w, bits, block, self.sym, False, self.hqq)
        hl, int32 = self.write_header(ic, oc, bits)
        wl = self.write(q) + hl
        al = self.write(alpha.float().half() if self.scale16 else alpha.float())
        ext = [self.offset, wl, al, 0, 0]
        self.offset += wl + al
        return ext, int32, bits, block

    def tensor(self, graph, name):
        graph["tensorName"].append(name)
        return [len(graph["tensorName"]) - 1]

    def rebuild_linear(self, op, graph):
        attrs = {a["key"]: a for a in op["main"]["attr"]}
        name = attrs["name"]["s"]
        ic, oc = attrs["in_features"]["i"], attrs["out_features"]["i"]
        lin = self.weight_ops[name]
        assert lin.in_features == ic and lin.out_features == oc
        t0 = time.time()
        ext, int32, bits, block = self.build_weight(lin, self.bits_for(name))
        print(f"  {name} [{oc}x{ic}] int{bits} block{block} {time.time()-t0:.1f}s", flush=True)
        pre_r, pre_c, conv, post_c = (self.tensor(graph, f"{name}/{s}") for s in
                                      ("pre_reshape", "pre_convert", "conv", "post_convert"))
        if bits == 16:
            qp = {"type": 3}
        else:
            bs = ic if block == 0 else block
            sym = self.sym and not self.last_prequant  # Q4_K weights carry a per-block min: asymmetric
            qp = {"quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0, "useInt32": False, "has_scaleInt": False,
                  "shapeInt32": int32, "type": 1, "aMaxOrBits": bits,
                  "aMin": 0 if sym else 1, "readType": 0 if sym else oc * (ic // bs), "weightSize": 0,
                  "scaleStorage": "FP16" if self.scale16 else "FP32"}
        return [
            {"name": f"{name}/pre_reshape", "type": "Reshape", "inputIndexes": op["inputIndexes"],
             "outputIndexes": pre_r, "main_type": "Reshape", "main": {"dims": [-1, ic, 1, 1], "dimType": "NCHW"},
             "defaultDimentionFormat": "NHWC"},
            {"name": f"{name}/pre_convert", "inputIndexes": pre_r, "outputIndexes": pre_c, "type": "ConvertTensor",
             "main_type": "TensorConvertInfo", "main": {"source": "NCHW", "dest": "NC4HW4"},
             "defaultDimentionFormat": "NHWC"},
            {"name": name, "inputIndexes": pre_c, "outputIndexes": conv, "type": "Convolution",
             "main_type": "Convolution2D",
             "main": {"common": {"dilateX": 1, "dilateY": 1, "strideX": 1, "strideY": 1, "kernelX": 1, "kernelY": 1,
                                 "padX": 0, "padY": 0, "group": 1, "outputCount": oc, "relu": False,
                                 "padMode": "CAFFE", "relu6": False, "inputCount": ic, "hasOutputShape": False},
                      "quanParameter": qp, "external": ext},
             "defaultDimentionFormat": "NHWC"},
            {"name": f"{name}/post_convert", "inputIndexes": conv, "outputIndexes": post_c, "type": "ConvertTensor",
             "main_type": "TensorConvertInfo", "main": {"source": "NC4HW4", "dest": "NCHW"},
             "defaultDimentionFormat": "NHWC"},
            {"name": f"{name}/post_reshape", "type": "Reshape", "inputIndexes": post_c,
             "outputIndexes": op["outputIndexes"], "main_type": "Reshape",
             "main": {"dims": [1, -1, oc], "dimType": "NCHW"}, "defaultDimentionFormat": "NHWC"},
        ]

    def rebuild_layernorm(self, op):
        m = op["main"]
        if "gamma" not in m:
            return [op]
        gl = self.write(m.pop("gamma"))
        bl = self.write(m.pop("beta"))
        m["external"] = [self.offset, gl, bl]
        self.offset += gl + bl
        return [op]

    def rebuild_const(self, op):
        m = op["main"]
        if "float32s" not in m or len(m["float32s"]) < 64:
            return [op]
        n = self.write(np.array(m.pop("float32s"), dtype=np.float32))
        m["external"] = [self.offset, n]
        self.offset += n
        return [op]

    def __call__(self, json_path, weight_path):
        g = json.load(open(json_path))
        # pull small external data (norm weights, constants) back into the graph before rewriting the file
        if os.path.exists(weight_path):
            with open(weight_path, "rb") as f:
                for op in g["oplists"]:
                    ext = op.get("main", {}).get("external") if isinstance(op.get("main"), dict) else None
                    if ext is None:
                        continue
                    if op["type"] == "LayerNorm":
                        f.seek(ext[0])
                        op["main"]["gamma"] = np.frombuffer(f.read(ext[1]), np.float32).tolist()
                        op["main"]["beta"] = np.frombuffer(f.read(ext[2]), np.float32).tolist()
                        del op["main"]["external"]
                    elif op["type"] == "Const":
                        f.seek(ext[0])
                        op["main"]["float32s"] = np.frombuffer(f.read(ext[1]), np.float32).tolist()
                        del op["main"]["external"]
        self.offset = 0
        new_ops = []
        with open(weight_path, "wb") as self.f:
            for op in g["oplists"]:
                t = op["type"]
                if t == "Extra":
                    attrs = {a["key"]: a for a in op["main"]["attr"]}
                    if op["main"].get("type") == "FakeLinear" or "in_features" in attrs:
                        new_ops += self.rebuild_linear(op, g)
                        continue
                    new_ops.append(op)
                elif t == "LayerNorm":
                    new_ops += self.rebuild_layernorm(op)
                elif t == "Const":
                    new_ops += self.rebuild_const(op)
                else:
                    new_ops.append(op)
        g["oplists"] = new_ops
        json.dump(g, open(json_path, "w"))


SCALE16 = True
FUSE = True


def to_mnn(onnx_path, mnn_path, weight_ops, bits_for, block, hqq=False, sym=False, fuse=None):
    fuse = FUSE if fuse is None else fuse
    weight_path = mnn_path + ".weight"
    json_path = mnn_path + ".json"
    args = [MNNCONVERT, "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", mnn_path, "--allowCustomOp",
            "--saveExternalData", "--bizCode", "qwen_image21"]
    if fuse:
        args.append("--transformerFuse")
    run(args)
    run([MNNCONVERT, "-f", "MNN", "--modelFile", mnn_path, "--JsonFile", json_path])
    Rebuilder(weight_ops, bits_for, block, sym=sym, hqq=hqq, scale16=SCALE16)(json_path, weight_path)
    run([MNNCONVERT, "-f", "JSON", "--modelFile", json_path, "--MNNModel", mnn_path])
    os.remove(json_path)
    print("wrote", mnn_path, os.path.getsize(weight_path) / 1e9, "GB weights", flush=True)


def export_onnx(model, inputs, path, input_names, output_names, dynamic_axes):
    Q.Linear.export_mode = True
    try:
        torch.onnx.export(model, inputs, path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17, dynamo=False, do_constant_folding=True)
    finally:
        Q.Linear.export_mode = False


def linears(module):
    return {m.name: m for m in module.modules() if isinstance(m, Q.Linear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf")
    ap.add_argument("--random_layers", type=int, default=0, help="test export with random weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--aux_bits", type=int, default=8, help="bits for embedders / modulation / proj_out")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--hqq", action="store_true")
    ap.add_argument("--only", default="txt_in,img_in,dit")
    ap.add_argument("--gate_first", type=int, default=1)
    ap.add_argument("--layers", type=int, default=Q.LAYERS, help="export only the first N blocks (testing)")
    ap.add_argument("--scale16", type=int, default=1, help="store quant scales as fp16")
    ap.add_argument("--fuse", type=int, default=1, help="MNNConvert --transformerFuse")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    onnx_dir = os.path.join(a.out, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    layers = Q.LAYERS
    if a.random_layers:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
        torch.manual_seed(0)
        layers = a.random_layers
        ref = QwenImage21Transformer2DModel(num_layers=layers)
        with torch.no_grad():
            for _, p in ref.named_parameters():
                p.copy_(torch.randn_like(p) * (0.02 if p.dim() > 1 else 0.1))
        sd = {k: v.detach() for k, v in ref.state_dict().items()}
        torch.save(sd, os.path.join(a.out, "random_sd.pt"))
        wp = lambda n: sd[n + ".weight"]
        params = lambda n: sd[n]
    else:
        W = GGUFWeights(a.gguf, gate_first=bool(a.gate_first))
        wp, params = W, W.param
        layers = a.layers

    def bits_for(name):
        return a.bits if name.startswith("transformer_blocks.") else a.aux_bits

    global SCALE16, FUSE
    SCALE16 = bool(a.scale16)
    FUSE = bool(a.fuse)
    only = a.only.split(",")
    if "txt_in" in only:
        m = Q.TxtIn(wp, params).eval()
        p = os.path.join(onnx_dir, "txt_in.onnx")
        export_onnx(m, (torch.randn(1, 8, Q.DIM),), p, ["txt"], ["txt_h"], {"txt": {1: "L"}, "txt_h": {1: "L"}})
        to_mnn(p, os.path.join(a.out, "txt_in.mnn"), linears(m), lambda n: a.aux_bits, a.block, a.hqq)
    if "img_in" in only:
        m = Q.ImgIn(wp).eval()
        p = os.path.join(onnx_dir, "img_in.onnx")
        export_onnx(m, (torch.randn(1, 16, Q.IN_CH),), p, ["lat"], ["img_h"], {"lat": {1: "N"}, "img_h": {1: "N"}})
        # img_in is tiny (64x4096) and int8 produced NaN rows on MNN CPU for VAE-encoded latents: keep fp16
        to_mnn(p, os.path.join(a.out, "img_in.mnn"), linears(m), lambda n: 16, a.block, a.hqq)
    if "dit" in only:
        m = Q.DiT(wp, params, layers=layers).eval()
        n, pl = 16, 8
        inputs = (torch.randn(1, n, Q.DIM), torch.tensor([0.5]), torch.randn(n, 64), torch.randn(n, 64),
                  torch.zeros(1, 1, n, pl + n),
                  *[torch.randn(2, pl, Q.HEADS, Q.HEAD_DIM) for _ in range(layers)])
        p = os.path.join(onnx_dir, "dit.onnx")
        past_names = [f"past_kv_{i}" for i in range(layers)]
        present_names = [f"present_kv_{i}" for i in range(layers)]
        dynamic = {"hidden": {1: "N"}, "rope_cos": {0: "N"}, "rope_sin": {0: "N"},
                   "attn_mask": {2: "N", 3: "T"}, "out": {1: "N"}}
        for nm in past_names:
            dynamic[nm] = {1: "P"}
        for nm in present_names:
            dynamic[nm] = {1: "N"}
        export_onnx(m, inputs, p, ["hidden", "timestep", "rope_cos", "rope_sin", "attn_mask"] + past_names,
                    ["out"] + present_names, dynamic)
        to_mnn(p, os.path.join(a.out, "dit.mnn"), linears(m), bits_for, a.block, a.hqq)


if __name__ == "__main__":
    main()
