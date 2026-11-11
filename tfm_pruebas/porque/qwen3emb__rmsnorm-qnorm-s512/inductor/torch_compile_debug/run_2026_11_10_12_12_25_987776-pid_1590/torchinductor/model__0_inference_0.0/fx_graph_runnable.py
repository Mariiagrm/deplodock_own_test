
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/workspace/deplodock/tfm_pruebas/porque/qwen3emb__rmsnorm-qnorm-s512/inductor/cache'
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



    def forward(self, arg0_1, arg1_1):
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [3], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(arg0_1, rsqrt);  arg0_1 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        return (mul_1,)

def load_args(reader):
    buf0 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 16, 512, 128), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None, is_inference=True)
        # mod(*args)