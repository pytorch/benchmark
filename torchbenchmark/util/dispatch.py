import functools
import json
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, Generator, Iterable, Tuple

import torch
import torchvision
import torchvision.extension
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

aten = torch.ops.aten
torchvision = torch.ops.torchvision  # noqa: F811
c10d = torch.ops.c10d


dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}

tensor_type = torch._C.TensorType.get()


def truncate_input(arg):
    if arg in dtype_abbrs:
        return dtype_abbrs[arg]
    elif isinstance(arg, torch.device):
        return arg.type
    else:
        return arg


def serialize_sparse_tensor(e):
    if isinstance(e, torch._subclasses.FakeTensor):
        return FuncCallWrapper("ST", list(e.shape), e.dtype, e.layout, e.is_coalesced())
    else:
        return FuncCallWrapper(
            "ST", list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()
        )


def contains_tensor(elems):
    for elem in pytree.tree_leaves(elems):
        if isinstance(elem, torch.Tensor):
            return True
    return False


def serialize_tensor(e):
    if not e.is_contiguous():
        return FuncCallWrapper("T", list(e.shape), e.dtype, stride=e.stride())
    else:
        return FuncCallWrapper("T", list(e.shape), e.dtype)


def serialize_torch_args(e):
    if isinstance(e, torch.Tensor):
        if e.is_sparse:
            return serialize_sparse_tensor(e)
        return serialize_tensor(e)
    else:
        return truncate_input(e)


def skip_args(elems):
    for i in pytree.tree_leaves(elems):
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False


def contains_tensor_types(type):
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


# Serialize Function Call
class FuncCallWrapper:
    def __init__(self, call, *args, **kwargs):
        self.call = call
        self.args = tree_map(truncate_input, args)
        self.kwargs = tree_map(truncate_input, kwargs) if kwargs is not None else {}

    def __repr__(self):
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = "".join(
            [f", {str(key)}={value}" for key, value in self.kwargs.items()]
        )
        out = f"{self.call}({args}{kwargs})".strip('"')
        # f strings introduce quotations we dont want
        for key in dtype_abbrs_parsing:
            out = out.replace(f"'{key}'", key)
        return out


@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema

    # skip constructors
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True
    if "_like" in op.name():
        return True

    # allow in place writes
    if schema.is_mutable:
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]

    # skip aliasing unless there are multiple outputs
    if len(tensor_outputs) != 1:
        return False

    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(
                tensor_outputs[0].alias_info.after_set
            ):
                return True

    return False


class OperatorInputsMode(TorchDispatchMode):
    def __init__(self, output_filename, func_db=None):
        super().__init__()
        self.func_db = defaultdict(Counter) if func_db is None else func_db
        self.output_filename = output_filename

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        arg_meta, kwarg_meta = tree_map(serialize_torch_args, (args, kwargs))

        out = func_overload(*args, **kwargs)

        inputs = (args, kwargs)
        if contains_tensor(inputs) and not skip_args(inputs) and contains_tensor(out):
            serialized_str = repr((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][serialized_str] += 1

        return out

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.log_to_file(self.output_filename)

    def log_to_file(self, output_filename, skip_non_compute_operators=True):
        sorted_operators = sorted(self.func_db.keys())
        json_obj = {}
        for operator in sorted_operators:
            if skip_non_compute_operators and non_compute_operator(eval(operator)):
                continue
            json_obj[operator] = []
            operator_inputs = self.func_db[operator]
            for inputs, count in operator_inputs.items():
                op_hit = {}
                op_hit["count"] = count
                # repr will add quotation marks around the dtype strings
                for dtype_abbr in dtype_abbrs.values():
                    inputs = inputs.replace("'" + dtype_abbr + "'", dtype_abbr)
                op_hit["inputs"] = inputs
                json_obj[operator].append(op_hit)
        with open(output_filename, "w") as f:
            json.dump(json_obj, f, indent=4)
