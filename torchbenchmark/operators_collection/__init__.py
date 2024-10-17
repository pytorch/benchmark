import importlib
import pathlib
from typing import List

OP_COLLECTION_PATH = "operators_collection"


def list_operator_collections() -> List[str]:
    """
    List the available operator collections.

    This function retrieves the list of available operator collections by scanning the directories
    in the current path that contain an "__init__.py" file.

    Returns:
        List[str]: A list of names of the available operator collections.
    """
    p = pathlib.Path(__file__).parent
    # only load the directories that contain a "__init__.py" file
    collection_paths = sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir() and child.joinpath("__init__.py").exists()
    )
    filtered_collections = [pathlib.Path(path).name for path in collection_paths]
    return filtered_collections


def list_operators_by_collection(op_collection: str = "default") -> List[str]:
    """
    List the operators from the specified operator collections.

    This function retrieves the list of operators from the specified operator collections.
    If the collection name is "all", it retrieves operators from all available collections.
    If the collection name is not specified, it defaults to the "default" collection.

    Args:
        op_collection (str): Names of the operator collections to list operators from.
        It can be a single collection name or a comma-separated list of names.
        Special value "all" retrieves operators from all collections.

    Returns:
        List[str]: A list of operator names from the specified collection(s).

    Raises:
        ModuleNotFoundError: If the specified collection module is not found.
        AttributeError: If the specified collection module does not have a 'get_operators' function.
    """

    def _list_all_operators(collection_name: str):
        try:
            module_name = f".{collection_name}"
            module = importlib.import_module(module_name, package=__name__)
            if hasattr(module, "get_operators"):
                return module.get_operators()
            else:
                raise AttributeError(
                    f"Module '{module_name}' does not have a 'get_operators' function"
                )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Module '{module_name}' not found")

    if op_collection == "all":
        collection_names = list_operator_collections()
    else:
        collection_names = op_collection.split(",")

    all_operators = []
    for collection_name in collection_names:
        all_operators.extend(_list_all_operators(collection_name))
    return all_operators
