
# By default, FlopCountAnalysis count one fused-mult-add (FMA) as one flop.
# However, in our context, we count 1 FMA as 2 flops instead of 1.
# https://github.com/facebookresearch/fvcore/blob/7a0ef0c0839fa0f5e24d2ef7f5d48712f36e7cd7/fvcore/nn/flop_count.py
def enable_fvcore_flops(model: 'torchbenchmark.util.model.BenchmarkModel', flops_fma=2.0):
    assert hasattr(model, 'TORCHVISION_MODEL') and model.TORCHVISION_MODEL, "fvcore flops is only available on torchvision models!"
    assert model.test == "eval", "fvcore flops is only available on inference tests, as it doesn't measure backward pass."
    from fvcore.nn import FlopCountAnalysis
    model.flops = FlopCountAnalysis(model.model, tuple(model.example_inputs)).total()
    model.flops = model.flops / model.batch_size * flops_fma
