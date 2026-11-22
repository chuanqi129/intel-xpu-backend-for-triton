# Descriptor loads whose width is not a whole number of 4-byte granules (#8814).
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = pytest.mark.skipif(not is_xpu(), reason="2D block IO is specific to the XPU backend")


@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, lda: tl.constexpr, ldb: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TRANS_B: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, K), (lda, 1), (BM, BK))
    b_desc = tl.make_tensor_descriptor(b_ptr, (N, K) if TRANS_B else (K, N), (ldb, 1), (BN, BK) if TRANS_B else
                                       (BK, BN))
    acc = tl.zeros((BM, BN), tl.int32 if a_ptr.dtype.element_ty == tl.int8 else tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        b = b_desc.load([0, k * BK]).T if TRANS_B else b_desc.load([k * BK, 0])
        acc = tl.dot(a_desc.load([0, k * BK]), b, acc)
    tl.store(c_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], acc)


@triton.jit
def _tile(a_ptr, c_ptr, M: tl.constexpr, W: tl.constexpr, lda: tl.constexpr, y, xq, G: tl.constexpr, BM: tl.constexpr,
          BK: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, W), (lda, 1), (BM, BK))
    a = a_desc.load([y, xq * G])
    k = tl.arange(0, BK)
    c = tl.dot(a, (k[:, None] == k[None, :]).to(a.dtype), out_dtype=tl.int32 if a.dtype == tl.int8 else tl.float32)
    tl.store(c_ptr + tl.arange(0, BM)[:, None] * BK + k[None, :], c)


def _view(rows, cols, dtype, device):
    # Padding that must never be read: NaN, or an int8 value that skews sums.
    pitch = (cols + 63) // 64 * 64 + 64
    buf = torch.full((rows, pitch), 3 if dtype == torch.int8 else float("nan"), dtype=dtype, device=device)
    buf[:, :cols] = torch.randint(-8, 8, (rows, cols), device=device).to(dtype)
    return buf[:, :cols]


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dtype, K", [(torch.float16, 7347), (torch.float16, 33), (torch.float16, 31),
                                      (torch.bfloat16, 255), (torch.int8, 131), (torch.int8, 132)])
def test_gemm(dtype, K, trans_b, device):
    if dtype == torch.int8 and trans_b:
        pytest.skip("int8 B is not loaded transposed")
    M, N, BK = 64, 32, 64 if dtype == torch.int8 else 32
    a, b = _view(M, K, dtype, device), _view(N, K, dtype, device) if trans_b else _view(K, N, dtype, device)
    c = torch.empty((M, N), dtype=torch.int32 if dtype == torch.int8 else torch.float32, device=device)
    k = _gemm[(1, )](a, b, c, M, N, K, a.stride(0), b.stride(0), BM=M, BN=N, BK=BK, TRANS_B=trans_b)
    ref = a.double() @ (b.T if trans_b else b).double()
    torch.testing.assert_close(c.double(), ref, rtol=0, atol=0)
    granule = 4 // dtype.itemsize
    marked = K % granule != 0 and (K - K % granule) * dtype.itemsize >= 64
    assert ("ttig.block_io_unaligned_width" in k.asm["ttgir"]) == marked


@pytest.mark.parametrize("dtype, W, x",
                         [(torch.float16, 67, x) for x in (-32, 0, 36, 66, 68)] + [(torch.int8, 66, x)
                                                                                   for x in (-64, 0, 4, 64)])
@pytest.mark.parametrize("y", [-3, 40])
def test_tile_offsets(dtype, W, x, y, device):
    M, BM, BK, G = 48, 64, 64 if dtype == torch.int8 else 32, 4 // dtype.itemsize
    a = _view(M, W, dtype, device)
    c = torch.empty((BM, BK), dtype=torch.int32 if dtype == torch.int8 else torch.float32, device=device)
    k = _tile[(1, )](a, c, M, W, a.stride(0), y, x // G, G=G, BM=BM, BK=BK)
    assert "ttig.block_io_unaligned_width" in k.asm["ttgir"]
    ref = torch.zeros((BM, BK), dtype=torch.float64)
    r0, r1, c0, c1 = max(y, 0), min(y + BM, M), max(x, 0), min(x + BK, W)
    if r0 < r1 and c0 < c1:
        ref[r0 - y:r1 - y, c0 - x:c1 - x] = a[r0:r1, c0:c1].cpu().double()
    torch.testing.assert_close(c.cpu().double(), ref, rtol=0, atol=0)
