from typing import Any, Callable

import torch


def setup_baseline():
    from torchao.quantization.utils import recommended_inductor_config_setter

    recommended_inductor_config_setter()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 10000


def torchao_optimize_ctx(quantization: str):
    from torchao.quantization.quant_api import (
        autoquant,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass
    import torchao

    def inner(model_iter_fn: Callable):
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            if getattr(module, "_quantized", None) is None:
                if quantization == "noquant":
                    if isinstance(example_inputs, dict):
                        args = ()
                        kwargs = example_inputs
                    else:
                        args = example_inputs
                        kwargs = {}

                    print("noquant run")
                    from torchao.utils import benchmark_model, profiler_runner
                    model = torch.compile(module, mode="max-autotune")
                    benchmark_model(model, 20, args, kwargs)
                    print("elapsed_time: ", benchmark_model(model, 100, args, kwargs), " milliseconds")
                #     profiler_runner("noquant.json.gz", benchmark_model, model, 5, inputs)

                if quantization == "int8dynamic":
                    quantize_(
                        module,
                        int8_dynamic_activation_int8_weight(),
                        set_inductor_config=False,
                    )

                    print("int8dynamic run")
                    from torchao.utils import benchmark_model, profiler_runner
                    torchao.quantization.utils.recommended_inductor_config_setter()
                    model = torch.compile(module, mode="max-autotune")
                    if isinstance(example_inputs, dict):
                        args = ()
                        kwargs = example_inputs
                    else:
                        args = example_inputs
                        kwargs = {}
                    benchmark_model(model, 20, args, kwargs)
                    print("elapsed_time: ", benchmark_model(model, 100, args, kwargs), " milliseconds")

                elif quantization == "int8weightonly":
                    quantize_(module, int8_weight_only(), set_inductor_config=False)

                    print("int8weightonly run")
                    from torchao.utils import benchmark_model, profiler_runner
                    torchao.quantization.utils.recommended_inductor_config_setter()
                    model = torch.compile(module, mode="max-autotune")
                    if isinstance(example_inputs, dict):
                        args = ()
                        kwargs = example_inputs
                    else:
                        args = example_inputs
                        kwargs = {}
                    benchmark_model(model, 20, args, kwargs)
                    print("elapsed_time: ", benchmark_model(model, 100, args, kwargs), " milliseconds")

                elif quantization == "int4weightonly":
                    quantize_(module, int4_weight_only(), set_inductor_config=False)
                if quantization == "autoquant":
                    from torchao.quantization import autoquant_v2

                    # autoquant(module, example_input=example_inputs, manual=True, error_on_unseen=False, set_inductor_config=True)
                    print("calling autoquant v2")
                    autoquant_v2(module, example_input=example_inputs, manual=True, error_on_unseen=False, set_inductor_config=True)
                    if isinstance(example_inputs, dict):
                        module(**example_inputs)
                    else:
                        module(*example_inputs)
                    module.finalize_autoquant()

                    # from torchao.quantization.autoquant import AUTOQUANT_CACHE
                    from torchao.quantization.autoquant_v2 import AUTOQUANT_CACHE

                    if len(AUTOQUANT_CACHE) == 0:
                        raise Exception(  # noqa: TRY002`
                            "NotAutoquantizable"
                            f"Found no autoquantizable layers in model {type(module)}, stopping autoquantized run"
                        )

                    print("autoquant run")
                    from torchao.utils import benchmark_model, profiler_runner
                    torchao.quantization.utils.recommended_inductor_config_setter()
                    model = torch.compile(module, mode="max-autotune")
                    if isinstance(example_inputs, dict):
                        args = ()
                        kwargs = example_inputs
                    else:
                        args = example_inputs
                        kwargs = {}
                    benchmark_model(model, 20, args, kwargs)
                    print("elapsed_time: ", benchmark_model(model, 100, args, kwargs), " milliseconds")
                    # profiler_runner("quant.json.gz", benchmark_model, model, 5, inputs)
                else:
                    unwrap_tensor_subclass(module)
                setattr(module, "_quantized", True)  # noqa: B010


            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner
