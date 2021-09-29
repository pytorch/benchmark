"""
A Benchmark Summary Metadata generator that can create or update model system-level configurations.
"""
import argparse
from distutils.util import strtobool
import importlib
import os
import pathlib
import yaml

from torchbenchmark import list_models, _list_model_paths, ModelTask, ModelDetails

TIMEOUT = 300  # seconds
torchbench_dir = 'torchbenchmark'
model_dir = 'models'

_DEFAULT_METADATA_ = {
    'train_benchmark': True,
    'train_deterministic': False,
    'eval_benchmark': True,
    'eval_deterministic': False,
    'eval_nograd': True,
    'optimized_for_inference': False,
    # 'origin': None,
    # 'train_dtype': 'float32',
    # 'eval_dtype': 'float32',
}


def _parser_helper(input):
    return None if input is None else bool(strtobool(str(input)))


def _process_model_details_to_metadata(model_detail: ModelDetails):
    metadata = {}
    for k, _ in _DEFAULT_METADATA_.items():
        v = getattr(model_detail, k, None)
        if v is not None:
            metadata[k] = v
    return metadata


def _extract_detail(path):
    name = os.path.basename(path)
    task = ModelTask(path, timeout=TIMEOUT)
    try:
        task.make_model_instance(device="cuda", jit=False)
        task.set_train()
        task.train()
        task.extract_details_train()
        assert (
            not task.model_details.optimized_for_inference or
            task.worker.load_stmt("hasattr(model, 'eval_model')"))
        task.set_eval()
        task.eval()
        task.extract_details_eval()
    except NotImplementedError:
        print(f'Model {name} is not fully implemented. skipping...')
    return _process_model_details_to_metadata(task._details)


def _extract_all_details(model_names):
    details = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        if model_name not in model_names:
            continue
        details.append(_extract_detail(model_path))
    return details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--debug", default=False, action="store_true", help="Show debugging output.")
    parser.add_argument("--model", default=None,
                        help="Full or partial name of a model to update. If partial, picks the first match.  \
                              If absent, applies to all models.")
    parser.add_argument("--train-benchmark", default=_DEFAULT_METADATA_['train_benchmark'], type=_parser_helper,
                        help="Whether to enable PyTorch benchmark mode during train.")
    parser.add_argument("--train-deterministic", default=_DEFAULT_METADATA_['train_deterministic'], type=_parser_helper,
                        help="Whether to enable deterministic during train.")
    parser.add_argument("--eval-benchmark", default=_DEFAULT_METADATA_['eval_benchmark'], type=_parser_helper,
                        help="Whether to enable PyTorch benchmark mode during eval.")
    parser.add_argument("--eval-deterministic", default=_DEFAULT_METADATA_['eval_deterministic'], type=_parser_helper,
                        help="Whether to enable deterministic during eval.")
    parser.add_argument("--eval-nograd", default=_DEFAULT_METADATA_['eval_nograd'], type=_parser_helper,
                        help="Whether to enable no_grad during eval.")
    parser.add_argument("--optimized-for-inference", default=_DEFAULT_METADATA_['optimized_for_inference'],
                        type=_parser_helper,
                        help="Whether to enable optimized_for_inference.")
    # parser.add_argument("--origin", default=_DEFAULT_METADATA_['origin'],
    #                     help="Location of benchmark's origin. Such as torchtext or torchvision.")
    # parser.add_argument("--train-dtype", default=_DEFAULT_METADATA_['train_dtype'],
    #                     choices=['float32', 'float16', 'bfloat16', 'amp'], help="Which fp type to perform training.")
    # parser.add_argument("--eval-dtype", default=_DEFAULT_METADATA_['eval_dtype'],
    #                     choices=['float32', 'float16', 'bfloat16', 'amp'], help="Which fp type to perform eval.")
    args = parser.parse_args()

    # Find the list of matching models.
    models = list_models(model_match=args.model)
    model_names = [m.name for m in models]

    if args.model is not None:
        if not models:
            print(f"Unable to find model matching: {args.model}.")
            exit(-1)
        print(f"Applying config to select models: {model_names}.")
    else:
        print("Applying config to all models.")

    # Extract all model details from models.
    extracted_details = _extract_all_details(model_names)
    if args.debug:
        print("Debug mode enabled - printing all extracted metadata.")
        for ed in extracted_details:
            attrs = vars(ed)
            print(f'model: {ed.name} , details: {attrs}')
