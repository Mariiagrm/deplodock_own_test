class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "u8[256, 1024]", arg1_1: "f16[256, 2]", arg2_1: "f32[1, 1]", arg3_1: "u8[256, 128]", arg4_1: "u8[24, 1024]", arg5_1: "f16[256, 2]", arg6_1: "f32[1, 1, 1]", arg7_1: "u8[24, 128, 1]"):
        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:578 in fn, code: env[nid] = v if out_dtype is None else v.to(out_dtype)
        convert_element_type: "i32[256, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int32);  arg0_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:231 in _eval, code: data, indices = ins[0], ins[1].long()
        convert_element_type_1: "i64[256, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.int64);  convert_element_type = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:233 in _eval, code: indices = torch.where(indices < 0, indices + data.shape[axis], indices)
        lt: "b8[256, 1024]" = torch.ops.aten.lt.Scalar(convert_element_type_1, 0)
        add: "i64[256, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, 256)
        where: "i64[256, 1024]" = torch.ops.aten.where.self(lt, add, convert_element_type_1);  lt = add = convert_element_type_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:237 in _eval, code: return torch.index_select(data, axis, indices.reshape(-1)).reshape(shape)
        view: "i64[262144]" = torch.ops.aten.view.default(where, [-1]);  where = None
        index: "f16[262144, 2]" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "f16[256, 1024, 2]" = torch.ops.aten.view.default(index, [256, 1024, 2]);  index = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:442 in _index_map, code: view = tensor[tuple(index)].permute(tuple(sorted(range(len(mapped)), key=mapped.__getitem__)))
        select: "f32[1]" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        select_1: "f32[]" = torch.ops.aten.select.int(select, 0, 0);  select = None
        permute: "f32[]" = torch.ops.aten.permute.default(select_1, []);  select_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:443 in _index_map, code: return view[tuple(slice(None) if axis in mapped else None for axis in range(len(out_shape)))].expand(out_shape)
        unsqueeze: "f32[1]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        unsqueeze_1: "f32[1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        expand: "f32[256, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [256, 128]);  unsqueeze_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:117 in <lambda>, code: return lambda a: a[0].contiguous().view(bits_dtype).to(torch.float32)
        view_2: "f8e4m3fn[256, 128]" = torch.ops.aten.view.dtype(arg3_1, torch.float8_e4m3fn);  arg3_1 = None
        convert_element_type_2: "f32[256, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:143 in <lambda>, code: "multiply": lambda a: a[0] * a[1],
        mul: "f32[256, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, expand);  convert_element_type_2 = expand = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:578 in fn, code: env[nid] = v if out_dtype is None else v.to(out_dtype)
        convert_element_type_3: "f16[256, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:337 in _eval, code: return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
        view_3: "f16[256, 128, 1]" = torch.ops.aten.view.default(convert_element_type_3, [256, 128, 1]);  convert_element_type_3 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:442 in _index_map, code: view = tensor[tuple(index)].permute(tuple(sorted(range(len(mapped)), key=mapped.__getitem__)))
        select_2: "f16[256, 128]" = torch.ops.aten.select.int(view_3, 2, 0);  view_3 = None
        permute_1: "f16[256, 128]" = torch.ops.aten.permute.default(select_2, [0, 1]);  select_2 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:443 in _index_map, code: return view[tuple(slice(None) if axis in mapped else None for axis in range(len(out_shape)))].expand(out_shape)
        unsqueeze_2: "f16[256, 128, 1]" = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_1: "f16[256, 128, 16]" = torch.ops.aten.expand.default(unsqueeze_2, [256, 128, 16]);  unsqueeze_2 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:337 in _eval, code: return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
        view_4: "f16[256, 2048]" = torch.ops.aten.view.default(view_1, [256, 2048]);  view_1 = None
        view_5: "f16[256, 128, 16]" = torch.ops.aten.view.default(view_4, [256, 128, 16]);  view_4 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:143 in <lambda>, code: "multiply": lambda a: a[0] * a[1],
        mul_1: "f16[256, 128, 16]" = torch.ops.aten.mul.Tensor(view_5, expand_1);  view_5 = expand_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:337 in _eval, code: return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
        view_6: "f16[256, 2048]" = torch.ops.aten.view.default(mul_1, [256, 2048]);  mul_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:578 in fn, code: env[nid] = v if out_dtype is None else v.to(out_dtype)
        convert_element_type_4: "i32[24, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.int32);  arg4_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:231 in _eval, code: data, indices = ins[0], ins[1].long()
        convert_element_type_5: "i64[24, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.int64);  convert_element_type_4 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:233 in _eval, code: indices = torch.where(indices < 0, indices + data.shape[axis], indices)
        lt_1: "b8[24, 1024]" = torch.ops.aten.lt.Scalar(convert_element_type_5, 0)
        add_1: "i64[24, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_5, 256)
        where_1: "i64[24, 1024]" = torch.ops.aten.where.self(lt_1, add_1, convert_element_type_5);  lt_1 = add_1 = convert_element_type_5 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:237 in _eval, code: return torch.index_select(data, axis, indices.reshape(-1)).reshape(shape)
        view_7: "i64[24576]" = torch.ops.aten.view.default(where_1, [-1]);  where_1 = None
        index_1: "f16[24576, 2]" = torch.ops.aten.index.Tensor(arg5_1, [view_7]);  arg5_1 = view_7 = None
        view_8: "f16[24, 1024, 2]" = torch.ops.aten.view.default(index_1, [24, 1024, 2]);  index_1 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:337 in _eval, code: return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
        view_9: "f16[24, 128, 16]" = torch.ops.aten.view.default(view_8, [24, 128, 16]);  view_8 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:442 in _index_map, code: view = tensor[tuple(index)].permute(tuple(sorted(range(len(mapped)), key=mapped.__getitem__)))
        select_3: "f32[1, 1]" = torch.ops.aten.select.int(arg6_1, 0, 0);  arg6_1 = None
        select_4: "f32[1]" = torch.ops.aten.select.int(select_3, 0, 0);  select_3 = None
        select_5: "f32[]" = torch.ops.aten.select.int(select_4, 0, 0);  select_4 = None
        permute_2: "f32[]" = torch.ops.aten.permute.default(select_5, []);  select_5 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:443 in _index_map, code: return view[tuple(slice(None) if axis in mapped else None for axis in range(len(out_shape)))].expand(out_shape)
        unsqueeze_3: "f32[1]" = torch.ops.aten.unsqueeze.default(permute_2, 0);  permute_2 = None
        unsqueeze_4: "f32[1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        expand_2: "f32[24, 128, 1]" = torch.ops.aten.expand.default(unsqueeze_5, [24, 128, 1]);  unsqueeze_5 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:117 in <lambda>, code: return lambda a: a[0].contiguous().view(bits_dtype).to(torch.float32)
        view_10: "f8e4m3fn[24, 128, 1]" = torch.ops.aten.view.dtype(arg7_1, torch.float8_e4m3fn);  arg7_1 = None
        convert_element_type_6: "f32[24, 128, 1]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:143 in <lambda>, code: "multiply": lambda a: a[0] * a[1],
        mul_2: "f32[24, 128, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_6, expand_2);  convert_element_type_6 = expand_2 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:578 in fn, code: env[nid] = v if out_dtype is None else v.to(out_dtype)
        convert_element_type_7: "f16[24, 128, 1]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:442 in _index_map, code: view = tensor[tuple(index)].permute(tuple(sorted(range(len(mapped)), key=mapped.__getitem__)))
        select_6: "f16[24, 128]" = torch.ops.aten.select.int(convert_element_type_7, 2, 0);  convert_element_type_7 = None
        permute_3: "f16[24, 128]" = torch.ops.aten.permute.default(select_6, [0, 1]);  select_6 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:443 in _index_map, code: return view[tuple(slice(None) if axis in mapped else None for axis in range(len(out_shape)))].expand(out_shape)
        unsqueeze_6: "f16[24, 128, 1]" = torch.ops.aten.unsqueeze.default(permute_3, 2);  permute_3 = None
        expand_3: "f16[24, 128, 16]" = torch.ops.aten.expand.default(unsqueeze_6, [24, 128, 16]);  unsqueeze_6 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:143 in <lambda>, code: "multiply": lambda a: a[0] * a[1],
        mul_3: "f16[24, 128, 16]" = torch.ops.aten.mul.Tensor(view_9, expand_3);  view_9 = expand_3 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:337 in _eval, code: return ins[0].reshape(_shape_ints(node.output.shape, sym_env))
        view_11: "f16[24, 2048]" = torch.ops.aten.view.default(mul_3, [24, 2048]);  mul_3 = None

        # File: /workspace/deplodock/emmy/compiler/backend/torch_ref.py:272 in _eval, code: return F.linear(ins[0], ins[1], ins[2] if op.has_bias else None)
        permute_4: "f16[2048, 256]" = torch.ops.aten.permute.default(view_6, [1, 0]);  view_6 = None
        mm: "f16[24, 256]" = torch.ops.aten.mm.default(view_11, permute_4);  view_11 = permute_4 = None
        return (mm,)
