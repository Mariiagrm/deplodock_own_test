class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[128, 132]", arg1_1: "f16[132, 128]"):
        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:291 in _eval, code: out = ins[0].to(dtype) @ ins[1].to(dtype)
        convert_element_type: "f32[128, 132]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[132, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mm: "f32[128, 128]" = torch.ops.aten.mm.default(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        return (mm,)
