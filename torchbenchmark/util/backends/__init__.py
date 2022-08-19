"""
Utils for managing backends
"""
import functools

BACKENDS = dict()

def create_backend(fn):
    @functools.wraps(fn)
    def inner(model: 'torchbenchmark.util.model.BenchmarkModel', **kwargs):
        if model is None:
            return None

        try:
            fn(model, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"{fn.__name__} error: {e}")
            raise

    BACKENDS[fn.__name__] = inner
    return inner

def list_backends():
    """
    Return valid strings that can be passed to:
        @torchdynamo.optimize(<backend>)
        def foo(...):
           ....
    """
    return sorted(BACKENDS.keys())

# register the backends
from .jit import torchscript
from .blade import blade

__all__ = [list_backends, create_backend ]