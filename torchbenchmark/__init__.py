import io
import os
from pathlib import Path
import subprocess
import sys
from urllib import request
import importlib
import tempfile
from typing import Any, List, Tuple

proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

this_dir = Path(__file__).parent.absolute()
model_dir = 'models'
install_file = 'install.py'


def _test_https(test_url: str = 'https://github.com', timeout: float = 0.5) -> bool:
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True


def _install_deps(model_path: str, verbose: bool = True) -> Tuple[bool, Any]:
    run_args = [
        [sys.executable, install_file],
    ]
    run_kwargs = {
        'cwd': model_path,
        'check': True,
    }

    output_buffer = None
    _, stdout_fpath = tempfile.mkstemp()
    try:
        output_buffer = io.FileIO(stdout_fpath, mode="w")
        if os.path.exists(os.path.join(model_path, install_file)):
            if not verbose:
                run_kwargs['stderr'] = subprocess.STDOUT
                run_kwargs['stdout'] = output_buffer
            subprocess.run(*run_args, **run_kwargs)  # type: ignore
        else:
            return (False, f"No install.py is found in {model_path}.", None)
    except subprocess.CalledProcessError as e:
        return (False, e.output, io.FileIO(stdout_fpath, mode="r").read().decode())
    except Exception as e:
        return (False, e, io.FileIO(stdout_fpath, mode="r").read().decode())
    finally:
        del output_buffer
        os.remove(stdout_fpath)

    return (True, None, None)


def _list_model_paths() -> List[str]:
    p = Path(__file__).parent.joinpath(model_dir)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())


def setup(verbose: bool = True, continue_on_fail: bool = False) -> bool:
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    failures = {}
    for model_path in _list_model_paths():
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg, stdout_stderr = _install_deps(model_path, verbose=verbose)
        if success:
            print("OK")
        else:
            print("FAIL")
            try:
                errmsg = errmsg.decode()
            except Exception:
                pass

            # If the install was very chatty, we don't want to overwhelm.
            # This will not affect verbose mode, which does not catch stdout
            # and stderr.
            log_lines = (stdout_stderr or "").splitlines(keepends=False)
            if len(log_lines) > 40:
                log_lines = log_lines[:20] + ["..."] + log_lines[-20:]
                stdout_stderr = "\n".join(log_lines)

            if stdout_stderr:
                errmsg = f"{stdout_stderr}\n\n{errmsg or ''}"

            failures[model_path] = errmsg
            if not continue_on_fail:
                break
    for model_path in failures:
        print(f"Error for {model_path}:")
        print("---------------------------------------------------------------------------")
        print(failures[model_path])
        print("---------------------------------------------------------------------------")
        print()

    return len(failures) == 0


def list_models():
    models = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        try:
            module = importlib.import_module(f'.models.{model_name}', package=__name__)
        except ModuleNotFoundError as e:
            print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
            continue
        Model = getattr(module, 'Model', None)
        if Model is None:
            print(f"Warning: {module} does not define attribute Model, skip it")
            continue
        if not hasattr(Model, 'name'):
            Model.name = model_name
        models.append(Model)
    return models
