from functools import reduce
import operator

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def mm_flops(x, y):
    m = x[0]
    k = x[1]
    assert k == y[0]
    n = y[1]
    return m * n * (k + k - 1)

def sigmoid_flops(x):
    # This is totally wrong.  Just using the relu number
    # For LSTM this will be OK because matrix multiply swamps
    return 2 * prod(x)

def tanh_flops(x):
    # https://stackoverflow.com/questions/41251698/how-many-flops-does-tanh-need
    return 8 * prod(x)
