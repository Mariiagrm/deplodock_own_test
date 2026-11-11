class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 16, 512, 128]", arg1_1: "f32[128]"):
        # File: /workspace/deplodock/venv/lib/python3.12/site-packages/torch/nn/functional.py:3012 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        pow_1: "f32[1, 16, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
        mean: "f32[1, 16, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [3], True);  pow_1 = None
        add: "f32[1, 16, 512, 1]" = torch.ops.aten.add.Scalar(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 16, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 16, 512, 128]" = torch.ops.aten.mul.Tensor(arg0_1, rsqrt);  arg0_1 = rsqrt = None
        mul_1: "f32[1, 16, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        return (mul_1,)
