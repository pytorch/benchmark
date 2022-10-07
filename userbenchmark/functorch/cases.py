from .util import BenchmarkCase
from torchbenchmark.models.lennard_jones import Model as LJModel
from torchbenchmark.models.functorch_maml_omniglot import Model as FTMamlOmniglot
from torchbenchmark.models.functorch_dp_cifar10 import Model as FTDPCifar10
from .vmap_hessian_fc import VmapHessianFC
from .simple_models import (
    SimpleCNN,
    SimpleMLP,
    VmapWrapper,
    EnsembleMultiWrapper,
    EnsembleSingleWrapper,
    PerSampleGradWrapper,
)


class TorchBenchModelWrapper(BenchmarkCase):
    def __init__(self, name, model, device):
        self.model = model('train', device)
        self.name_ = f'{name}_{device}'

    def name(self):
        return self.name_

    def run(self):
        return self.model.train()


# functorch user benchmark
# ------------------------
# This userbenchmark is used for regression testing of:
# - microbenchmarks,
# - low-quality models that shouldn't go into torchbenchmark
# - pieces of models where we do not have access to the full model.
# - models in torchbenchmark that have not yet made it to a release branch
# (and therefore are not being tracked for regressions).
#
# When adding a functorch-related benchmark, please prefer finding a high-quality
# model that uses the benchmark and adding it to the torchbenchmark suite.
# There is better infra support there and other folks use those models
# for cross-cutting tests.
benchmark_cases = [
    # [models from torchbench that haven't made it to stable yet]
    lambda: TorchBenchModelWrapper('lennard_jones', LJModel, 'cpu'),
    lambda: TorchBenchModelWrapper('lennard_jones', LJModel, 'cuda'),
    lambda: TorchBenchModelWrapper('functorch_maml_omniglot', FTMamlOmniglot, 'cpu'),
    lambda: TorchBenchModelWrapper('functorch_maml_omniglot', FTMamlOmniglot, 'cuda'),
    lambda: TorchBenchModelWrapper('functorch_dp_cifar10', FTDPCifar10, 'cuda'),
    # end [models from torchbench that haven't made it to stable yet]
    VmapHessianFC,
    # [combinations from functorch tutorials]
    lambda: VmapWrapper(SimpleMLP, 'cpu'),
    lambda: VmapWrapper(SimpleMLP, 'cuda'),
    lambda: EnsembleMultiWrapper(SimpleMLP, 'cpu'),
    lambda: EnsembleMultiWrapper(SimpleMLP, 'cuda'),
    lambda: EnsembleMultiWrapper(SimpleCNN, 'cuda'),
    lambda: EnsembleSingleWrapper(SimpleMLP, 'cpu'),
    lambda: EnsembleSingleWrapper(SimpleMLP, 'cuda'),
    lambda: EnsembleSingleWrapper(SimpleCNN, 'cuda'),
    lambda: PerSampleGradWrapper(SimpleMLP, 'cpu'),
    lambda: PerSampleGradWrapper(SimpleMLP, 'cuda'),
    lambda: PerSampleGradWrapper(SimpleCNN, 'cuda'),
    # end [combinations from functorch tutorials]
]
