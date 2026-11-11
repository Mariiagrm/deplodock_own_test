class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 4, 128, 128]", arg1_1: "f16[1, 4, 128, 128]"):
        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:291 in _eval, code: out = ins[0].to(dtype) @ ins[1].to(dtype)
        convert_element_type: "f32[1, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        expand: "f32[1, 4, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, [1, 4, 128, 128]);  convert_element_type = None
        view: "f32[4, 128, 128]" = torch.ops.aten.reshape.default(expand, [4, 128, 128]);  expand = None
        convert_element_type_1: "f32[1, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        expand_1: "f32[1, 4, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_1, [1, 4, 128, 128]);  convert_element_type_1 = None
        view_1: "f32[4, 128, 128]" = torch.ops.aten.reshape.default(expand_1, [4, 128, 128]);  expand_1 = None
        bmm: "f32[4, 128, 128]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[1, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 4, 128, 128]);  bmm = None
        return (view_2,)
