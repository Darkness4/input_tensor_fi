"""A pixel."""

from dataclasses import dataclass


@dataclass
class PixelFault:
    """A rgb positioned pixel."""

    x: int
    y: int
    r: int
    g: int
    b: int

    @property
    def coordinates(self):
        return (self.x, self.y)

    @property
    def rgb(self):
        return (self.r, self.g, self.b)
