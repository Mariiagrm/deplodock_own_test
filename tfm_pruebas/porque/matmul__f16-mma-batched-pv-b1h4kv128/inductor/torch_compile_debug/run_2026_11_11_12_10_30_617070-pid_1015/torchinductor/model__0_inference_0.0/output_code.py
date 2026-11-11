# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/deplodock/tfm_pruebas/porque/matmul__f16-mma-batched-pv-b1h4kv128/inductor/cache/ww/cwwlndreqntq2z5olhasv32qx6d3w7l7t5gpm22yjnpj7syc2wci.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "f16[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   return %expand
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 655360}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/deplodock/tfm_pruebas/porque/matmul__f16-mma-batched-pv-b1h4kv128/inductor/cache/2i/c2ixsvrhytuqqci3jb64nxagqgi46vw3vubnimf5jqr6oghuicgt.py
# Topologically Sorted Source Nodes: [to, out, to_1], Original ATen: [aten._to_copy, aten.view, aten.bmm]
# Source node to ATen node mapping:
#   out => bmm, view, view_1
#   to => convert_element_type
#   to_1 => convert_element_type_1
# Graph fragment:
#   %expand : Tensor "f32[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0" = PlaceHolder[target=expand]
#   %expand_1 : Tensor "f32[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0" = PlaceHolder[target=expand_1]
#   %convert_element_type : Tensor "f32[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [4, 128, 128]), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %view_1 : Tensor "f32[4, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [4, 128, 128]), kwargs = {})
#   %bmm : Tensor "f32[4, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   return %bmm
triton_tem_fused__to_copy_bmm_view_1 = async_compile.triton('triton_tem_fused__to_copy_bmm_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_bmm_view_1', 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_bmm_view_1(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 128
    N = 128
    K = 128
    BATCH = 4

    stride_aq = 16384
    stride_am = 128
    stride_ak = 1

    stride_bq = 16384
    stride_bk = 128
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)

    # Reconstruct batch index from grid_y/grid_z split (handles batch > 65535)
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    # Clamp to valid range for safe pointer arithmetic; out-of-bounds CTAs are
    # masked off at the store below.
    idx_q_clamped = tl.minimum(idx_q, BATCH - 1)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q_clamped*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q_clamped*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_remaining in range(K, 0, -BLOCK_K):

        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_q = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N) & (idx_q < BATCH)

    # inductor generates a suffix
    xindex = idx_n + 128*idx_m + 16384*idx_q
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 4, 128, 128), (65536, 16384, 128, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((1, 4, 128, 128), (65536, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg0_1, buf0, 65536, stream=raw_stream0)
            del arg0_1
            assert_size_stride(arg1_1, (1, 4, 128, 128), (65536, 16384, 128, 1), 'input')
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((1, 4, 128, 128), (65536, 16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [to_1], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_0:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg1_1, buf1, 65536, stream=raw_stream0)
            del arg1_1
            buf2 = empty_strided_cuda((4, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [to, out, to_1], Original ATen: [aten._to_copy, aten.view, aten.bmm]
            # [Provenance debug handles] triton_tem_fused__to_copy_bmm_view_1:3
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_bmm_view_1.run(buf0, buf1, buf2, 8, 4, 1, stream=raw_stream0)
            del buf0
            del buf1
        return (reinterpret_tensor(buf2, (1, 4, 128, 128), (65536, 16384, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 4, 128, 128), (65536, 16384, 128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, 4, 128, 128), (65536, 16384, 128, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
