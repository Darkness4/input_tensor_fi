"""
Bit manipulation are stored here.

Inherit [_BitAction] to create a new bit manipulation.
"""
from abc import ABCMeta, abstractmethod


class _BitAction(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def call(number: int, bit: int) -> int:
        """Do something with the [bit]th of the [number]."""
        raise NotImplementedError("do is not implemented.")


class BitSet(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], set the [bit] to 1."""
        mask = 1 << bit
        return number | mask


class BitReset(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], set the [bit] to 0."""
        mask = 1 << bit
        mask = 0xFFFFFFFF ^ mask
        return number & mask


class BitFlip(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], flip the [bit]."""
        mask = 1 << bit
        return number ^ mask
