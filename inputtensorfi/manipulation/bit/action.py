"""
Bit manipulation are stored here.

Inherit [_BitAction] to create a new bit manipulation.
"""
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class _BitAction(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def call(number: int, bit: int) -> int:
        """Do something with the [bit]th of the [number]."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def as_tensor(bit: int) -> tf.Tensor:
        """Convert in a tensor."""
        raise NotImplementedError()


class BitSet(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], set the [bit] to 1."""
        mask = 1 << bit
        return number | mask

    @staticmethod
    def as_tensor(bit: int) -> tf.Tensor:
        mask = tf.bitwise.left_shift(tf.constant(1), tf.constant(bit))

        def tensor(input: tf.Tensor) -> tf.Tensor:
            return tf.bitwise.bitwise_or(tf.cast(input, mask.dtype), mask)

        return tensor


class BitReset(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], set the [bit] to 0."""
        mask = 1 << bit
        mask = 0xFFFFFFFF ^ mask
        return number & mask

    @staticmethod
    def as_tensor(bit: int) -> tf.Tensor:
        mask = tf.bitwise.left_shift(
            tf.constant(1, dtype=tf.int64), tf.constant(bit, dtype=tf.int64)
        )
        mask = tf.bitwise.bitwise_xor(tf.constant(0xFFFFFFFF), mask)

        def tensor(input: tf.Tensor) -> tf.Tensor:
            return tf.bitwise.bitwise_and(tf.cast(input, mask.dtype), mask)

        return tensor


class BitFlip(_BitAction):
    @staticmethod
    def call(number: int, bit: int) -> int:
        """For a [number], flip the [bit]."""
        mask = 1 << bit
        return number ^ mask

    @staticmethod
    def as_tensor(bit: int) -> tf.Tensor:
        mask = tf.bitwise.left_shift(tf.constant(1), tf.constant(bit))

        def tensor(input: tf.Tensor) -> tf.Tensor:
            return tf.bitwise.bitwise_xor(tf.cast(input, mask.dtype), mask)

        return tensor
