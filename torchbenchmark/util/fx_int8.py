import re

import torch
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qconfig_propagation_list,
    QuantWrapper,
)
from torch.ao.quantization.quantize_fx import _fuse_fx, convert_fx, prepare_fx
from torchbenchmark.util.env_check import is_hf_model


def _append_attr(fx_module, module, fx_white_list=[]):
    fx_attr = dir(fx_module)
    org_attr = dir(module)
    ignore_match_patterns = [
        r"_",
        r"quant",
        r"dequant",
        r"weight",
        r"bias",
        r"activation_post_process",
    ]
    ignore_search_patterns = [r"_scale_", r"_zero_point_", r"_activation_post_process_"]
    add_special_patterns = [
        r"_forward_hooks",
        r"_forward_pre_hooks",
        r"_backward_hooks",
    ]
    attr_names = []
    for i in org_attr:
        if (
            type(module) in fx_white_list
            and type(module) != torch.nn.Sequential
            and any([re.search(p, i) for p in add_special_patterns])
        ):
            continue
        if any([re.search(p, i) for p in add_special_patterns]) or (
            i not in fx_attr
            and not any([re.match(p, i) for p in ignore_match_patterns])
            and not any([re.search(p, i) for p in ignore_search_patterns])
        ):
            attr_names.append(i)
    for name in attr_names:
        attr = getattr(module, name, None)
        if isinstance(attr, torch.nn.Module) or isinstance(
            attr, torch.quantization.qconfig.QConfig
        ):
            continue
        setattr(fx_module, name, attr)
    return fx_module


def get_sub_module(model, module_dict, prefix):
    fx_white_list = get_default_qconfig_propagation_list()
    ignore_list = []
    if is_hf_model:
        import transformers

        ignore_list.extend(
            [
                transformers.models.gpt2.modeling_gpt2.GPT2Attention,
                transformers.models.t5.modeling_t5.T5DenseActDense,
            ]
        )

    def _get_sub_module(model, module_dict, prefix, sub_module_list):
        for name, module in model.named_children():
            quant_wrap_flag = False
            if type(module) in ignore_list:
                continue
            op_name = prefix + "." + name if prefix != "" else name
            if op_name not in module_dict:
                continue
            if type(module) in fx_white_list and type(module) != torch.nn.Sequential:
                module = QuantWrapper(module)
                quant_wrap_flag = True
            try:
                graph_module = torch.fx.symbolic_trace(module)
                if not quant_wrap_flag and str(module.get_submodule).count("\n") != str(
                    graph_module.get_submodule
                ).count("\n"):
                    continue
                _fuse_fx(graph_module, False)
                setattr(model, name, module)
                sub_module_list.append(op_name)
            except:
                module = _get_sub_module(module, module_dict, op_name, sub_module_list)
                setattr(model, name, module)
        return model

    sub_module_list = []
    model = _get_sub_module(model, module_dict, prefix, sub_module_list)
    return model, sub_module_list


def prepare_sub_module(sub_module_list, model, prefix, quant_engine: str = "x86"):
    qconfig_mapping = get_default_qconfig_mapping(quant_engine)
    for name, module in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        if op_name in sub_module_list:
            prepared_module = prepare_fx(module, qconfig_mapping, None)
            _append_attr(prepared_module, module)
            setattr(model, name, prepared_module)
        else:
            prepared_module = prepare_sub_module(
                sub_module_list, module, op_name, quant_engine
            )
            _append_attr(prepared_module, module)
            setattr(model, name, prepared_module)
    return model


def convert_sub_module(sub_module_list, model, prefix):
    for name, module in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        if op_name in sub_module_list:
            convert_module = convert_fx(module)
            setattr(model, name, convert_module)
        else:
            convert_module = convert_sub_module(sub_module_list, module, op_name)
            setattr(model, name, convert_module)
    return model
