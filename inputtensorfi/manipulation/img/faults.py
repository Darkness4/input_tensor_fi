from dataclasses import asdict, astuple, dataclass, fields
from typing import Type

from inputtensorfi.manipulation.bit.action import _BitAction


@dataclass
class PixelFault:
    """A pîxel-fault description."""

    x: int
    y: int
    r: int
    g: int
    b: int

    def __post_init__(self):
        self.x = int(self.x)
        self.y = int(self.y)
        self.r = int(self.r)
        self.g = int(self.g)
        self.b = int(self.b)

    @property
    def coordinates(self):
        return (self.x, self.y)

    @property
    def rgb(self):
        return (self.r, self.g, self.b)

    def to_dict(self):
        return asdict(self)

    def to_tuple(self):
        return astuple(self)


@dataclass
class BitFault:
    """A bit-fault description."""

    x: int
    y: int
    rgb: int
    bit: int
    bit_action: Type[_BitAction]

    def to_dict(self):
        return asdict(self)

    def to_tuple(self):
        return astuple(self)