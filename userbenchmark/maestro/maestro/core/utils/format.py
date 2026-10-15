from typing import Union
from numbers import Number
from decimal import Decimal
import re

def format_size(nbytes: int, precision=2) -> str:
    units = ['B', 'K', 'M', 'G']
    units_ix = 0
    while nbytes/1024 >= 1 and units_ix < len(units)-1:
        nbytes /= 1024
        units_ix += 1

    nbytes = round(nbytes, precision)
    return f"{nbytes:g}{units[units_ix]}"

def parse_size(nbytes: Union[str, int]) -> int:
    """Parse nonnegative integral bytes, with binary K/M/G or KiB/MiB/GiB.

    Fractional magnitudes are accepted when they resolve to whole bytes.
    """
    if isinstance(nbytes, Number):
        try:
            count = int(nbytes)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ValueError(f"Size must resolve to nonnegative integral bytes, got {nbytes!r}") from exc
        if isinstance(nbytes, bool) or count != nbytes or count < 0:
            raise ValueError(f"Size must resolve to nonnegative integral bytes, got {nbytes!r}")
        return count
    if not isinstance(nbytes, str):
        raise TypeError(f"Invalid size type: {type(nbytes)}, expected str or number")

    match = re.fullmatch(r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*(b|k|m|g|kib|mib|gib)?", nbytes.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid size: {nbytes!r}")
    units = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3,
             "kib": 1024, "mib": 1024**2, "gib": 1024**3}
    numerator, denominator = Decimal(match.group(1)).as_integer_ratio()
    numerator *= units[(match.group(2) or "").lower()]
    if numerator < 0 or numerator % denominator:
        raise ValueError(f"Size must resolve to nonnegative integral bytes, got {nbytes!r}")
    return numerator // denominator
