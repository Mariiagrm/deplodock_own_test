
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/workspace/deplodock/tfm_pruebas/porque/matmul__fp4-tma-w4a4-linear-ring/inductor/cache'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHAO_BUILD_VERSION'] = '+git1272f3cf'
os.environ['TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION'] = '0'
os.environ['TRITON_CUPTI_LIB_PATH'] = '/usr/local/cuda/lib64'
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.0;12.0'
os.environ['TORCH_NCCL_USE_COMM_NONBLOCKING'] = '0'
os.environ['TORCHTITAN_BUILD_VERSION'] = '0.2.0+gite98ae995'
os.environ['PYTORCH_VERSION'] = '2.10.0a0+a36e1d3'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TRITON_CUDACRT_PATH'] = '/usr/local/cuda/include'
os.environ['TRITON_CUOBJDUMP_PATH'] = '/usr/local/cuda/bin/cuobjdump'
os.environ['TRITON_NVDISASM_PATH'] = '/usr/local/cuda/bin/nvdisasm'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.10.0a0+a36e1d3'
os.environ['TRITON_CUPTI_INCLUDE_PATH'] = '/usr/local/cuda/include'
os.environ['NVIDIA_PYTORCH_VERSION'] = '26.01'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TRITON_CUDART_PATH'] = '/usr/local/cuda/include'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.max_autotune = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.15.0+cu132
# torch cuda version: 13.2
# torch git version: d0f2af8b66b20da4baa50644c0cd3f887a2eae2e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Tue_Dec_16_07:23:41_PM_PST_2025 
# Cuda compilation tools, release 13.1, V13.1.115 
# Build cuda_13.1.r13.1/compiler.37061995_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 5090 : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.int32);  arg0_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.int64);  convert_element_type = None
        lt = torch.ops.aten.lt.Scalar(convert_element_type_1, 0)
        add = torch.ops.aten.add.Tensor(convert_element_type_1, 256)
        where = torch.ops.aten.where.self(lt, add, convert_element_type_1);  lt = add = convert_element_type_1 = None
        view = torch.ops.aten.view.default(where, [-1]);  where = None
        index = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1 = torch.ops.aten.view.default(index, [256, 1024, 2]);  index = None
        select = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        select_1 = torch.ops.aten.select.int(select, 0, 0);  select = None
        permute = torch.ops.aten.permute.default(select_1, []);  select_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        expand = torch.ops.aten.expand.default(unsqueeze_1, [256, 128]);  unsqueeze_1 = None
        view_2 = torch.ops.aten.view.dtype(arg3_1, torch.float8_e4m3fn);  arg3_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, expand);  convert_element_type_2 = expand = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None
        view_3 = torch.ops.aten.view.default(convert_element_type_3, [256, 128, 1]);  convert_element_type_3 = None
        select_2 = torch.ops.aten.select.int(view_3, 2, 0);  view_3 = None
        permute_1 = torch.ops.aten.permute.default(select_2, [0, 1]);  select_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_2, [256, 128, 16]);  unsqueeze_2 = None
        view_4 = torch.ops.aten.view.default(view_1, [256, 2048]);  view_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [256, 128, 16]);  view_4 = None
        mul_1 = torch.ops.aten.mul.Tensor(view_5, expand_1);  view_5 = expand_1 = None
        view_6 = torch.ops.aten.view.default(mul_1, [256, 2048]);  mul_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(arg4_1, torch.int32);  arg4_1 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.int64);  convert_element_type_4 = None
        lt_1 = torch.ops.aten.lt.Scalar(convert_element_type_5, 0)
        add_1 = torch.ops.aten.add.Tensor(convert_element_type_5, 256)
        where_1 = torch.ops.aten.where.self(lt_1, add_1, convert_element_type_5);  lt_1 = add_1 = convert_element_type_5 = None
        view_7 = torch.ops.aten.view.default(where_1, [-1]);  where_1 = None
        index_1 = torch.ops.aten.index.Tensor(arg5_1, [view_7]);  arg5_1 = view_7 = None
        view_8 = torch.ops.aten.view.default(index_1, [24, 1024, 2]);  index_1 = None
        view_9 = torch.ops.aten.view.default(view_8, [24, 128, 16]);  view_8 = None
        select_3 = torch.ops.aten.select.int(arg6_1, 0, 0);  arg6_1 = None
        select_4 = torch.ops.aten.select.int(select_3, 0, 0);  select_3 = None
        select_5 = torch.ops.aten.select.int(select_4, 0, 0);  select_4 = None
        permute_2 = torch.ops.aten.permute.default(select_5, []);  select_5 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(permute_2, 0);  permute_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_5, [24, 128, 1]);  unsqueeze_5 = None
        view_10 = torch.ops.aten.view.dtype(arg7_1, torch.float8_e4m3fn);  arg7_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_6, expand_2);  convert_element_type_6 = expand_2 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None
        select_6 = torch.ops.aten.select.int(convert_element_type_7, 2, 0);  convert_element_type_7 = None
        permute_3 = torch.ops.aten.permute.default(select_6, [0, 1]);  select_6 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(permute_3, 2);  permute_3 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_6, [24, 128, 16]);  unsqueeze_6 = None
        mul_3 = torch.ops.aten.mul.Tensor(view_9, expand_3);  view_9 = expand_3 = None
        view_11 = torch.ops.aten.view.default(mul_3, [24, 2048]);  mul_3 = None
        permute_4 = torch.ops.aten.permute.default(view_6, [1, 0]);  view_6 = None
        mm = torch.ops.aten.mm.default(view_11, permute_4);  view_11 = permute_4 = None
        return (mm,)

def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    reader.tensor(buf0, (256, 1024), dtype=torch.uint8, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (256, 2), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 1), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    reader.tensor(buf3, (256, 128), dtype=torch.uint8, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 24576, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    reader.tensor(buf4, (24, 1024), dtype=torch.uint8, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (256, 2), dtype=torch.float16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1, 1, 1), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    reader.tensor(buf7, (24, 128, 1), dtype=torch.uint8, is_leaf=True)  # arg7_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # mod(*args)