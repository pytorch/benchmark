from typing import Union
from numbers import Number

def format_size(nbytes: int, precision=2) -> str:
    units = ['B', 'K', 'M', 'G']
    units_ix = 0
    while nbytes/1024 >= 1 and units_ix < len(units)-1:
        nbytes /= 1024
        units_ix += 1
    
    nbytes = round(nbytes, precision)
    return f"{nbytes:g}{units[units_ix]}"
    
def parse_size(nbytes: Union[str, int]) -> int:
    """Convert formatted string with unit to bytes"""
    if isinstance(nbytes, Number):
        return int(nbytes)
    elif isinstance(nbytes, str):
        if not nbytes:
            raise ValueError(f"Invalid size: '{nbytes}'")
        options = {
            'g': 1024*1024*1024,
            'm': 1024*1024,
            'k': 1024,
        }
        unit = 1
        key = nbytes[-1].lower()
        if key in options:
            unit = options[key]
            value = int(nbytes[:-1])
        else:
            value = int(nbytes)
        count = unit * value 
        return count
    else:
        raise TypeError(f"Invalid size type: {type(nbytes)}, expected str or int")