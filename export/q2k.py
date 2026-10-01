"""Lossless GGUF Q2_K -> MNN asymmetric int2 (block 16).

Q2_K super-block (256 weights, 84 bytes): 16 bytes of packed 4-bit (scale, min) for 16 sub-blocks of 16 weights,
64 bytes of 2-bit q, d fp16, dmin fp16. Sub-block j: w = d*sc_j*q - dmin*m_j with q in [0, 3].
MNN asymmetric int2: stored u in [0, 3], alpha = (zero, scale), w = zero + scale*(u - 2).
So u = q, scale = d*sc_j, zero = 2*d*sc_j - dmin*m_j.
"""
import numpy as np

QK_K = 256
BITS = 2
BLOCK = 16
BLOCK_BYTES = 84


def q2k_decode(raw, oc, ic):
    """raw: uint8 [oc, ic/256*84] -> q uint8 [oc, ic], scale/zero float32 [oc, ic/16]"""
    nb = ic // QK_K
    b = np.ascontiguousarray(raw).reshape(oc, nb, BLOCK_BYTES)
    scales = b[:, :, 0:16]                                                    # [oc, nb, 16]
    qs = b[:, :, 16:80]                                                       # [oc, nb, 64]
    d = b[:, :, 80:82].copy().view(np.float16)[..., 0].astype(np.float32)     # [oc, nb]
    dmin = b[:, :, 82:84].copy().view(np.float16)[..., 0].astype(np.float32)

    sc = (scales & 0xF).astype(np.float32)                                    # [oc, nb, 16]
    mn = (scales >> 4).astype(np.float32)

    q = np.empty((oc, nb, 16, 16), np.uint8)
    for c in range(2):
        chunk = qs[:, :, 32 * c:32 * c + 32]                                  # [oc, nb, 32]
        for s in range(4):
            shift = 2 * s
            q[:, :, c * 8 + 2 * s] = (chunk[:, :, 0:16] >> shift) & 3
            q[:, :, c * 8 + 2 * s + 1] = (chunk[:, :, 16:32] >> shift) & 3

    scale = d[..., None] * sc                                                 # [oc, nb, 16]
    minv = dmin[..., None] * mn
    zero = 2.0 * scale - minv
    return q.reshape(oc, ic), scale.reshape(oc, ic // 16), zero.reshape(oc, ic // 16)


def dequant(q, scale, zero):
    oc, ic = q.shape
    return (zero[..., None] + scale[..., None] * (q.reshape(oc, ic // 16, 16).astype(np.float32) - 2)).reshape(oc, ic)


def mnn_pack(q, scale, zero):
    """-> packed uint8 weights (MSB first, 4 per byte) and alpha [(zero, scale) per block] as MNN expects."""
    flat = q.reshape(-1, 4)
    packed = (flat[:, 0] << 6 | flat[:, 1] << 4 | flat[:, 2] << 2 | flat[:, 3]).astype(np.uint8)
    alpha = np.stack([zero.reshape(-1), scale.reshape(-1)], axis=-1).reshape(-1).astype(np.float32)
    return packed, alpha
