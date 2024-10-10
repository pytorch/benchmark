import importlib
import os
import pathlib
from typing import List

from torchbenchmark import dir_contains_file

OPBENCH_DIR = "operators"
INTERNAL_OPBENCH_DIR = "fb"

from enum import Enum


class OP_COLLECTION(Enum):
    """
    Collection of operators to benchmark.
    For DEFAULT collection, we don't need to add ops in COLLECTION_MAP. All ops not in
    other specific collections will be included in DEFAULT collection.
    """

    LIGER = "liger"
    DEFAULT = "default"
    ALL = "all"


COLLECTION_MAP = {
    OP_COLLECTION.LIGER: ["FusedLinearCrossEntropy"],
}


def _is_internal_operator(op_name: str) -> bool:
    p = (
        pathlib.Path(__file__)
        .parent.parent.joinpath(OPBENCH_DIR)
        .joinpath(INTERNAL_OPBENCH_DIR)
        .joinpath(op_name)
    )
    if p.exists() and p.joinpath("__init__.py").exists():
        return True
    return False


def _list_opbench_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.parent.joinpath(OPBENCH_DIR)
    # Only load the model directories that contain a "__init.py__" file
    opbench = sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir() and dir_contains_file(child, "__init__.py")
    )
    p = p.joinpath(INTERNAL_OPBENCH_DIR)
    if p.exists():
        o = sorted(
            str(child.absolute())
            for child in p.iterdir()
            if child.is_dir() and dir_contains_file(child, "__init__.py")
        )
        opbench.extend(o)
    return opbench


def list_operators(op_collection: OP_COLLECTION = OP_COLLECTION.DEFAULT) -> List[str]:
    operators = list(map(lambda y: os.path.basename(y), _list_opbench_paths()))
    if op_collection != OP_COLLECTION.ALL:
        if op_collection == OP_COLLECTION.DEFAULT:
            all_other_operators = [
                op for sublist in COLLECTION_MAP.values() for op in sublist
            ]
            operators = [op for op in operators if op not in all_other_operators]
        else:
            operators = COLLECTION_MAP[op_collection]
    if INTERNAL_OPBENCH_DIR in operators:
        operators.remove(INTERNAL_OPBENCH_DIR)
    return operators


def load_opbench_by_name(op_name: str):
    opbench_list = filter(
        lambda x: op_name.lower() == x.lower(),
        map(lambda y: os.path.basename(y), _list_opbench_paths()),
    )
    opbench_list = list(opbench_list)
    if not opbench_list:
        raise RuntimeError(f"{op_name} is not found in the Tritonbench operator list.")
    assert (
        len(opbench_list) == 1
    ), f"Found more than one operators {opbench_list} matching the required name: {op_name}"
    op_name = opbench_list[0]
    op_pkg = (
        op_name
        if not _is_internal_operator(op_name)
        else f"{INTERNAL_OPBENCH_DIR}.{op_name}"
    )
    module = importlib.import_module(f".{op_pkg}", package=__name__)

    Operator = getattr(module, "Operator", None)
    if Operator is None:
        print(f"Warning: {module} does not define attribute Operator, skip it")
        return None
    if not hasattr(Operator, "name"):
        Operator.name = op_name
    return Operator
