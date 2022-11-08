# The Guidance to Sync DCGM APIs with Upstream

## Folder Structure
There are two folders in `components/model_analyzer`:
- `dcgm`: the DCGM APIs that are used by `TorchBenchAnalyzer`
- `tb_dcgm_types`: the data structure for each metrics, a configuration file, logger, exception, and aggragator.

## How to sync with upstream
Because of the different folder structure, we need to sync the [upstream DCGM APIs](https://github.com/NVIDIA/DCGM/tree/master/testing/python3) with the following steps.
1. Use vscode or other editor compare the following files' difference. 
    `dcgm_agent.py, dcgm_field_helpers.py, dcgm_field_internal.py, dcgm_fields.py, dcgm_structs.py, dcgm_value.py`
2. If there are no changes for function names or function parameters, we can directly copy the upstream file to the corresponding file in `components/model_analyzer/dcgm`.
3. Update imports in those files. Please follow the previous code in TorchBenchAnalyzer.
4. Copy some code segements to make it work with torchbench. [DcgmFieldGroup](https://github.com/pytorch/benchmark/blob/main/components/model_analyzer/dcgm/dcgm_field_helpers.py#L23-L29)
5. If there are API changes, we need to update the corresponding code in `dcgm_monitor.py`, `TorchBenchAnalyzer.py`, and any other files that use them.