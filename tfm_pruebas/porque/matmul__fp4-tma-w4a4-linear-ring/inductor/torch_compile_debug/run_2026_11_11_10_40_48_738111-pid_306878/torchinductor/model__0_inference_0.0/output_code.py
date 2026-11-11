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


# kernel path: /workspace/deplodock/tfm_pruebas/porque/matmul__fp4-tma-w4a4-linear-ring/inductor/cache/hp/chpeoycqnhk4doxr4ot7cdxxmcgcrdx2e5dyidwxvygk3quqv5gd.py
# Topologically Sorted Source Nodes: [to_13, long_1, lt_1, add_1, where_1, reshape_6, index_select_1, v_11, v_12, view_4, v_14, getitem_4, view_3, getitem_5, v_13, v_15, tensor_1, getitem_6, view_5, getitem_7, v_17, v_18], Original ATen: [aten._to_copy, aten.lt, aten.add, aten.where, aten.view, aten.index_select, aten.select, aten.permute, aten.unsqueeze, aten.expand, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_1
#   getitem_4 => select_3, select_4, select_5
#   getitem_5 => unsqueeze_3, unsqueeze_4, unsqueeze_5
#   getitem_6 => select_6
#   getitem_7 => unsqueeze_6
#   index_select_1 => index_1
#   long_1 => convert_element_type_5
#   lt_1 => lt_1
#   reshape_6 => view_7
#   tensor_1 => convert_element_type_7
#   to_13 => convert_element_type_4
#   v_11 => view_8
#   v_12 => view_9
#   v_13 => expand_2
#   v_14 => convert_element_type_6
#   v_15 => mul_2
#   v_17 => expand_3
#   v_18 => mul_3
#   view_3 => permute_2
#   view_4 => view_10
#   view_5 => permute_3
#   where_1 => where_1
# Graph fragment:
#   %arg4_1 : Tensor "u8[24, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f16[256, 2][2, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg7_1 : Tensor "u8[24, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg6_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_4 : Tensor "i32[24, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.int32), kwargs = {})
#   %convert_element_type_5 : Tensor "i64[24, 1024][1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_4, torch.int64), kwargs = {})
#   %lt_1 : Tensor "b8[24, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%convert_element_type_5, 0), kwargs = {})
#   %add_1 : Tensor "i64[24, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_5, 256), kwargs = {})
#   %where_1 : Tensor "i64[24, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_1, %add_1, %convert_element_type_5), kwargs = {})
#   %view_7 : Tensor "i64[24576][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where_1, [-1]), kwargs = {})
#   %index_1 : Tensor "f16[24576, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg5_1, [%view_7]), kwargs = {})
#   %view_8 : Tensor "f16[24, 1024, 2][2048, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [24, 1024, 2]), kwargs = {})
#   %view_9 : Tensor "f16[24, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [24, 128, 16]), kwargs = {})
#   %view_10 : Tensor "f8e4m3fn[24, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%arg7_1, torch.float8_e4m3fn), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[24, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_10, torch.float32), kwargs = {})
#   %select_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg6_1, 0, 0), kwargs = {})
#   %select_4 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_3, 0, 0), kwargs = {})
#   %select_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_4, 0, 0), kwargs = {})
#   %permute_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_5, []), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_2, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_3, 1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 2), kwargs = {})
#   %expand_2 : Tensor "f32[24, 128, 1][0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_5, [24, 128, 1]), kwargs = {})
#   %mul_2 : Tensor "f32[24, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %expand_2), kwargs = {})
#   %convert_element_type_7 : Tensor "f16[24, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float16), kwargs = {})
#   %select_6 : Tensor "f16[24, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%convert_element_type_7, 2, 0), kwargs = {})
#   %permute_3 : Tensor "f16[24, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_6, [0, 1]), kwargs = {})
#   %unsqueeze_6 : Tensor "f16[24, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_3, 2), kwargs = {})
#   %expand_3 : Tensor "f16[24, 128, 16][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_6, [24, 128, 16]), kwargs = {})
#   %mul_3 : Tensor "f16[24, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %expand_3), kwargs = {})
#   return %mul_3
triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0 = async_compile.triton('triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u8', 'in_ptr1': '*fp16', 'in_ptr2': '*u8', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 224256}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 // 2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp2 + tmp5
    tmp7 = tl.where(tmp4, tmp6, tmp2)
    tmp8 = (tl.full([XBLOCK], 256, tl.int32)).to(tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 256), "index out of bounds: 0 <= tmp11 < 256")
    tmp13 = tl.load(in_ptr1 + (2*tmp11 + ((x0 % 2))), None).to(tl.float32)
    tmp15 = tmp14.to(tl.float8e4nv, bitcast=True)
    tmp16 = tmp15.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /workspace/deplodock/tfm_pruebas/porque/matmul__fp4-tma-w4a4-linear-ring/inductor/cache/ep/cepdafan7p76bwixmmvxik2wxfphixgiqeap4pq3qqsbg6o6llai.py
# Topologically Sorted Source Nodes: [to, long, lt, add, where, reshape, index_select, v, v_7, v_8, view_1, v_2, getitem, view, getitem_1, v_1, v_3, to_6, v_5, getitem_2, view_2, getitem_3, v_6, v_9], Original ATen: [aten._to_copy, aten.lt, aten.add, aten.where, aten.view, aten.index_select, aten.select, aten.permute, aten.unsqueeze, aten.expand, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   getitem => select, select_1
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_2 => select_2
#   getitem_3 => unsqueeze_2
#   index_select => index
#   long => convert_element_type_1
#   lt => lt
#   reshape => view
#   to => convert_element_type
#   to_6 => convert_element_type_3
#   v => view_1
#   v_1 => expand
#   v_2 => convert_element_type_2
#   v_3 => mul
#   v_5 => view_3
#   v_6 => expand_1
#   v_7 => view_4
#   v_8 => view_5
#   v_9 => mul_1
#   view => permute
#   view_1 => view_2
#   view_2 => permute_1
#   where => where
# Graph fragment:
#   %arg0_1 : Tensor "u8[256, 1024][1024, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[256, 2][2, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "u8[256, 128][128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg2_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "i32[256, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int32), kwargs = {})
#   %convert_element_type_1 : Tensor "i64[256, 1024][1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int64), kwargs = {})
#   %lt : Tensor "b8[256, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%convert_element_type_1, 0), kwargs = {})
#   %add : Tensor "i64[256, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 256), kwargs = {})
#   %where : Tensor "i64[256, 1024][1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %add, %convert_element_type_1), kwargs = {})
#   %view : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%where, [-1]), kwargs = {})
#   %index : Tensor "f16[262144, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view]), kwargs = {})
#   %view_1 : Tensor "f16[256, 1024, 2][2048, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [256, 1024, 2]), kwargs = {})
#   %view_4 : Tensor "f16[256, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [256, 2048]), kwargs = {})
#   %view_5 : Tensor "f16[256, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [256, 128, 16]), kwargs = {})
#   %view_2 : Tensor "f8e4m3fn[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%arg3_1, torch.float8_e4m3fn), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %select : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg2_1, 0, 0), kwargs = {})
#   %select_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select, 0, 0), kwargs = {})
#   %permute : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, []), kwargs = {})
#   %unsqueeze : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %expand : Tensor "f32[256, 128][0, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [256, 128]), kwargs = {})
#   %mul : Tensor "f32[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %expand), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.float16), kwargs = {})
#   %view_3 : Tensor "f16[256, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [256, 128, 1]), kwargs = {})
#   %select_2 : Tensor "f16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_3, 2, 0), kwargs = {})
#   %permute_1 : Tensor "f16[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_2, [0, 1]), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[256, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_1, 2), kwargs = {})
#   %expand_1 : Tensor "f16[256, 128, 16][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_2, [256, 128, 16]), kwargs = {})
#   %mul_1 : Tensor "f16[256, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %expand_1), kwargs = {})
#   return %mul_1
triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1 = async_compile.triton('triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u8', 'in_ptr1': '*fp16', 'in_ptr2': '*u8', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 2392064}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 // 2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp2 + tmp5
    tmp7 = tl.where(tmp4, tmp6, tmp2)
    tmp8 = (tl.full([XBLOCK], 256, tl.int32)).to(tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 256), "index out of bounds: 0 <= tmp11 < 256")
    tmp13 = tl.load(in_ptr1 + (2*tmp11 + ((x0 % 2))), None).to(tl.float32)
    tmp15 = tmp14.to(tl.float8e4nv, bitcast=True)
    tmp16 = tmp15.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        assert_size_stride_grouped((arg4_1, arg5_1, arg7_1, arg6_1), ((24, 1024), (256, 2), (24, 128, 1), (1, 1, 1)), ((1024, 1), (2, 1), (128, 1, 1), (1, 1, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg4_1 = copy_if_misaligned(arg4_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            arg7_1 = copy_if_misaligned(arg7_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            buf0 = empty_strided_cuda((24, 128, 16), (2048, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [to_13, long_1, lt_1, add_1, where_1, reshape_6, index_select_1, v_11, v_12, view_4, v_14, getitem_4, view_3, getitem_5, v_13, v_15, tensor_1, getitem_6, view_5, getitem_7, v_17, v_18], Original ATen: [aten._to_copy, aten.lt, aten.add, aten.where, aten.view, aten.index_select, aten.select, aten.permute, aten.unsqueeze, aten.expand, aten.mul]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_0.run(arg4_1, arg5_1, arg7_1, arg6_1, buf0, 49152, stream=raw_stream0)
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            assert_size_stride_grouped((arg0_1, arg1_1, arg3_1, arg2_1), ((256, 1024), (256, 2), (256, 128), (1, 1)), ((1024, 1), (2, 1), (128, 1), (1, 1)), 'input')
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_cuda((256, 128, 16), (2048, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [to, long, lt, add, where, reshape, index_select, v, v_7, v_8, view_1, v_2, getitem, view, getitem_1, v_1, v_3, to_6, v_5, getitem_2, view_2, getitem_3, v_6, v_9], Original ATen: [aten._to_copy, aten.lt, aten.add, aten.where, aten.view, aten.index_select, aten.select, aten.permute, aten.unsqueeze, aten.expand, aten.mul]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_expand_index_select_lt_mul_permute_select_unsqueeze_view_where_1.run(arg0_1, arg1_1, arg3_1, arg2_1, buf1, 524288, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            buf2 = empty_strided_cuda((24, 256), (256, 1), torch.float16)
            # Topologically Sorted Source Nodes: [to_13, long_1, lt_1, add_1, where_1, reshape_6, index_select_1, v_11, v_12, view_4, v_14, getitem_4, view_3, getitem_5, v_13, v_15, tensor_1, getitem_6, view_5, getitem_7, v_17, v_18, v_19, to, long, lt, add, where, reshape, index_select, v, v_7, v_8, view_1, v_2, getitem, view, getitem_1, v_1, v_3, to_6, v_5, getitem_2, view_2, getitem_3, v_6, v_9, v_10, v_20], Original ATen: [aten._to_copy, aten.lt, aten.add, aten.where, aten.view, aten.index_select, aten.select, aten.permute, aten.unsqueeze, aten.expand, aten.mul, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(reinterpret_tensor(buf0, (24, 2048), (2048, 1), 0), reinterpret_tensor(buf1, (2048, 256), (1, 2048), 0), out=buf2)
            del buf0
            del buf1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.uint8)
    arg1_1 = rand_strided((256, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.uint8)
    arg4_1 = rand_strided((24, 1024), (1024, 1), device='cuda:0', dtype=torch.uint8)
    arg5_1 = rand_strided((256, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((24, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.uint8)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
