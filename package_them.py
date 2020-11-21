from torchbenchmark import load_model, model_names
from contextlib import redirect_stderr, redirect_stdout
import torch
from torch.package import PackageExporter, PackageImporter
import tempfile
from pathlib import Path

no_cpu_impl = [
    'Background_Matting',
    'maskrcnn_benchmark',
    'moco',
    'pytorch_CycleGAN_and_pix2pix',
    'tacotron2',
]

def tpe(x):
    if isinstance(x, list) or isinstance(x, tuple):
       return f"({','.join(str(tpe(e)) for e in x)})"
    else:
        return type(x)

def check_close(a, b):
    if isinstance(a, (list, tuple)):
        for ae, be in zip(a, b):
            check_close(ae, be)
    else:
        print(torch.max(torch.abs(a - b)))
        assert torch.allclose(a, b)


class Exporter(PackageExporter):
    def require_module(self, m, dependencies=True):
        if 'numpy' in m or 'scipy' in m:
            self.mock_module(m)
        else:
            super().require_module(m, dependencies)

t2tsource = """
import torch
# annotating with jit interface messes up packaging
class TensorToTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
"""
class BERT_pytorch(Exporter):
    def require_module(self, m, dependencies=True):
        if m == 'torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.tensor2tensor':
            self.save_source_string('torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.tensor2tensor', t2tsource)
        else:
            super().require_module(m, dependencies)

class demucs(Exporter):
    def require_module(self, m, dependencies=True):
        if 'tqdm' in m:
            self.mock_module(m)
        else:
            super().require_module(m, dependencies)

class dlrm(Exporter):
    def require_module(self, m, dependencies=True):
        patterns = ['dlrm.data_utils', 'dlrm.data_loader_pytorch', 'dlrm_data_pytorch', 'onnx', 'sklearn']
        if any(x in m for x in patterns):
            self.mock_module(m)
        else:
            super().require_module(m, dependencies)

class fastNLP(Exporter):
    def require_module(self, m, dependencies=True):
        patterns = ['_logger', 'vocabulary', 'fastNLP.core', 'fastNLP.io.file_utils']
        if any(x in m for x in patterns):
            self.mock_module(m)
        elif m == 'regex':
            self.save_source_string('regex', """
# result is unused but the pattern is compiled when the file is imported
def compile(*args, **kwargs):
    return None
            """)
        else:
            super().require_module(m, dependencies)

class yolov3(Exporter):
    def require_module(self, m, dependencies=True):
        patterns = ['yolo_utils.utils']
        if any(x in m for x in patterns):
            self.mock_module(m)
        else:
            super().require_module(m, dependencies)

with open('model_logs.txt', 'w') as model_logs:

    for model_name in model_names():
        result_file = f'results/{model_name}'
        if model_name in no_cpu_impl or Path(result_file).exists():
            continue
        print(f'packaging {model_name}')
        with redirect_stdout(model_logs), redirect_stderr(model_logs):
            Model = load_model(model_name)
            m = Model(jit=False, device='cpu')
            module, eg = m.get_module()

        if model_name == 'yolov3':
            # clean up some numpy objects in the model
            module.module_defs = None
            module.version = None
            module.seen = None

        with tempfile.TemporaryDirectory(dir='.') as tempdirname:
            model_path = Path(tempdirname) / model_name
            exporter_class = globals().get(model_name, Exporter)
            with exporter_class(str(Path(tempdirname) / model_name)) as exporter:
                exporter.save_pickle('model', 'model.pkl', module)
                exporter.save_pickle('model', 'eg.pkl', eg)

            importer = PackageImporter(str(model_path))
            module2 = importer.load_pickle('model', 'model.pkl')
            eg2 = importer.load_pickle('model', 'eg.pkl')

            with torch.no_grad():
                r = module(*eg)
                r2 = module2(*eg2)
            check_close(r, r2)
            model_path.replace(result_file)

