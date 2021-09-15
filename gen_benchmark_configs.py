"""
A benchmark configuration generator that can create or update model system-level configurations.
"""
import argparse
from distutils.util import strtobool
import importlib
import os
from pathlib import Path
import sys
import yaml

import torch

torchbench_dir = 'torchbenchmark'
model_dir = 'models'

_DEFAULT_CONFIG_ = {
  'benchmark':               True,
  'origin':                  None,
  'eval_deterministic':      True,
  'eval_nograd':             True,
  'optimized_for_inference': False,
  'train_dtype':             'float32',
  'eval_dtype':              'float32',
}

def _parser_helper(input):
  if x is None:
    return None
  return bool(strtobool(str(x)))

def _list_module_paths():
  p = Path(__file__).parent.joinpath(torchbench_dir).joinpath(model_dir)
  return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())

def _list_modules():
  modules = []
  for module_path in _list_module_paths():
    print(f'Imported module_path: {module_path}')
    model_name = os.path.basename(module_path)
    try:
      module = importlib.import_module(f'torchbenchmark.models.{model_name}', package=__name__)
    except ModuleNotFoundError as e:
      print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
      continue
    model = getattr(module, 'Model', None)
    if model is None:
      continue
    if hasattr(model, 'name'):
      module.name = model.name
    else:
      module.name = model_name
      model.name = model_name
    modules.append(module)
  return modules

class ModelConfig():
  def __init__(self, name):
    self.name = name
    for k, v in _DEFAULT_CONFIG_.items():
      setattr(self, k, v)

def _extract_config(module):
  model = getattr(module, 'Model', None)
  mc = ModelConfig(module.name)
  if model:
    for k,_ in _DEFAULT_CONFIG_.items():
      v = getattr(model, k, None)
      if v is not None:
        setattr(mc, k, v)
  return mc

def _extract_all_configs(modules):
  configs = []
  for module in modules:
    configs.append(_extract_config(module))
  return configs

if __name__ == "__main__":
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("--model", default=None, help="Full or partial name of a model to update. If partial, picks the first match. If absent, applies to all models.")
  parser.add_argument("--benchmark", default=_DEFAULT_CONFIG_['benchmark'], type=_parser_helper, help="Whether to enable PyTorch benchmark mode.")
  parser.add_argument("--origin", default=_DEFAULT_CONFIG_['origin'], help="Location of benchmark's origin. Such as torchtext or torchvision.")
  parser.add_argument("--eval-deterministic", default=_DEFAULT_CONFIG_['eval_deterministic'], type=_parser_helper, help="Whether to enable deterministic during eval.")
  parser.add_argument("--eval-nograd", default=_DEFAULT_CONFIG_['eval_nograd'], type=_parser_helper, help="Whether to enable no_grad during eval.")
  parser.add_argument("--optimized-for-inference", default=_DEFAULT_CONFIG_['optimized_for_inference'], type=_parser_helper, help="Whether to enable optimized_for_inference.")
  parser.add_argument("--train-dtype", default=_DEFAULT_CONFIG_['train_dtype'], choices=['float32', 'float16', 'bfloat16', 'amp'], help="Which fp type to perform training.")
  parser.add_argument("--eval-dtype", default=_DEFAULT_CONFIG_['eval_dtype'], choices=['float32', 'float16', 'bfloat16', 'amp'], help="Which fp type to perform eval.")
  args = parser.parse_args()

  # Import all modules
  modules = _list_modules()

  # Find the given module, if given.
  chosen_module = None
  if args.model is not None:
    for module in modules:
      if args.model.lower() in module.name.lower():
        chosen_module = module
        break
    if chosen_module is not None:
      print(f"Applying config to {module.name}.")
      modules.clear()
      modules.append(chosen_module)
    else:
      print(f"Unable to find model matching {args.model}.")
      exit(-1)
  else:
    print("Applying config to all models.")

  # Extract all attributes possible from Models
  extracted_configs = _extract_all_configs(modules)

  for ec in extracted_configs:
    attrs = vars(ec)
    print(f'model {ec.name} , ofi: {attrs}')

