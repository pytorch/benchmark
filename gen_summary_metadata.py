"""
A Benchmark Summary Metadata tool to extract and generate metadata from models at runtime.
"""
import argparse
from distutils.util import strtobool
import os
import yaml
from typing import Any, Dict, List, Tuple

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


def _process_model_details_to_metadata(model_detail: ModelDetails) -> Dict[str, Any]:
    metadata = {}
    metadata['optimized_for_inference'] = model_detail.optimized_for_inference
    for k, _ in _DEFAULT_METADATA_.items():
        v = model_detail.metadata[k] if k in model_detail.metadata else None
        if v is not None:
            metadata[k] = v
    return metadata


def _extract_detail(path: str) -> Dict[str, Any]:
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


def _extract_all_details(model_names: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    details = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        if model_name not in model_names:
            continue
        ed = _extract_detail(model_path)
        details.append((model_path, ed))
    return details


def _print_extracted_details(extracted_details: List[Tuple[str, Dict[str, Any]]]):
    for path, ex_detail in extracted_details:
        name = os.path.basename(path)
        print(f'Model: {name} , Details: {ex_detail}')


def _maybe_override_extracted_details(args, extracted_details: List[Tuple[str, Dict[str, Any]]]):
    for _path, ex_detail in extracted_details:
        if args.train_benchmark is not None:
            ex_detail['train_benchmark'] = args.train_benchmark
        elif args.train_deterministic is not None:
            ex_detail['train_deterministic'] = args.train_deterministic
        elif args.eval_benchmark is not None:
            ex_detail['eval_benchmark'] = args.eval_benchmark
        elif args.eval_deterministic is not None:
            ex_detail['eval_deterministic'] = args.eval_deterministic
        elif args.eval_nograd is not None:
            ex_detail['eval_nograd'] = args.eval_nograd
        elif args.optimized_for_inference is not None:
            ex_detail['optimized_for_inference'] = args.optimized_for_inference


def _write_metadata_yaml_files(extracted_details: List[Tuple[str, Dict[str, Any]]]):
    for path, ex_detail in extracted_details:
        metadata_path = path + "/metadata.yaml"
        with open(metadata_path, 'w') as file:
            yaml.dump(ex_detail, file)
            print(f"Processed file: {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", default=None,
                        help="Full or partial name of a model to update. If partial, picks the first match.  \
                              If absent, applies to all models.")
    parser.add_argument("--extract-only", default=False, type=_parser_helper,
                        help="Only extract model details.")
    parser.add_argument("--train-benchmark", default=None, type=_parser_helper,
                        help="Whether to enable PyTorch benchmark mode during train.")
    parser.add_argument("--train-deterministic", default=None, type=_parser_helper,
                        help="Whether to enable deterministic during train.")
    parser.add_argument("--eval-benchmark", default=None, type=_parser_helper,
                        help="Whether to enable PyTorch benchmark mode during eval.")
    parser.add_argument("--eval-deterministic", default=None, type=_parser_helper,
                        help="Whether to enable deterministic during eval.")
    parser.add_argument("--eval-nograd", default=None, type=_parser_helper,
                        help="Whether to enable no_grad during eval.")
    parser.add_argument("--optimized-for-inference", default=None, type=_parser_helper,
                        help="Whether to enable optimized_for_inference.")
    # parser.add_argument("--origin", default=None,
    #                     help="Location of benchmark's origin. Such as torchtext or torchvision.")
    # parser.add_argument("--train-dtype", default=None,
    #                     choices=['float32', 'float16', 'bfloat16', 'amp'], help="Which fp type to perform training.")
    # parser.add_argument("--eval-dtype", default=None,
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
    print("Printing extracted metadata.")
    _print_extracted_details(extracted_details)

    # Stop here for extract-only.
    if args.extract_only:
        print("Extract-only is set. Stop here.")
        exit(0)

    # Apply details passed in by flags.
    _maybe_override_extracted_details(args, extracted_details)
    print("Printing metadata after applying any modifications.")
    _print_extracted_details(extracted_details)

    # TODO: Modify and update the model to apply metadata changes by the user.

    # Generate a metadata files for each matching model.
    _write_metadata_yaml_files(extracted_details)
