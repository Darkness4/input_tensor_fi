from dataclasses import dataclass
from typing import Type

from inputtensorfi.manipulation.bit.action import _BitAction


@dataclass
class BitFault:
    """A rgb positioned pixel."""

    x: int
    y: int
    rgb: int
    bit: int
    bit_action: Type[_BitAction]
