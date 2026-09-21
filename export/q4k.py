"""Lossless GGUF Q4_K -> MNN asymmetric int4 (block 32).

Q4_K super-block (256 weights, 144 bytes): d fp16, dmin fp16, 12 bytes of packed 6-bit (scale, min) for 8 sub-blocks,
128 bytes of 4-bit q. Sub-block j: w = d*sc_j*q - dmin*m_j with q in [0, 15].
MNN asymmetric int4: stored u in [0, 15], alpha = (zero, scale), w = zero + scale*(u - 8).
So u = q, scale = d*sc_j, zero = 8*d*sc_j - dmin*m_j.
"""
import numpy as np

QK_K = 256
BLOCK_BYTES = 144


def q4k_decode(raw, oc, ic):
    """raw: uint8 [oc, ic/256*144] -> q uint8 [oc, ic], scale/zero float32 [oc, ic/32]"""
    nb = ic // QK_K
    b = np.ascontiguousarray(raw).reshape(oc, nb, BLOCK_BYTES)
    d = b[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float32)        # [oc, nb]
    dmin = b[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    s = b[:, :, 4:16].astype(np.uint16)                                       # [oc, nb, 12]
    qs = b[:, :, 16:144]                                                      # [oc, nb, 128]
    sc = np.empty((oc, nb, 8), np.float32)
    mn = np.empty((oc, nb, 8), np.float32)
    for j in range(4):
        sc[..., j] = s[..., j] & 63
        mn[..., j] = s[..., j + 4] & 63
    for j in range(4, 8):
        sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
        mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    q = np.empty((oc, nb, 8, 32), np.uint8)
    for c in range(4):
        ql = qs[:, :, 32 * c:32 * c + 32]
        q[:, :, 2 * c] = ql & 0xF
        q[:, :, 2 * c + 1] = ql >> 4
    scale = d[..., None] * sc                                                 # [oc, nb, 8]
    minv = dmin[..., None] * mn
    zero = 8.0 * scale - minv
    return q.reshape(oc, ic), scale.reshape(oc, ic // 32), zero.reshape(oc, ic // 32)


def dequant(q, scale, zero):
    oc, ic = q.shape
    return (zero[..., None] + scale[..., None] * (q.reshape(oc, ic // 32, 32).astype(np.float32) - 8)).reshape(oc, ic)


def mnn_pack(q, scale, zero):
    """-> packed uint8 weights (high nibble first) and alpha [(zero, scale) per block] as MNN expects."""
    flat = q.reshape(-1, 2)
    packed = (flat[:, 0] << 4 | flat[:, 1]).astype(np.uint8)
    alpha = np.stack([zero.reshape(-1), scale.reshape(-1)], axis=-1).reshape(-1).astype(np.float32)
    return packed, alpha
