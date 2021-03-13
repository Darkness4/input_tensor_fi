from inputtensorfi.manipulation.bit import action


def test_bit_set():
    number = 0b10001101

    result = action.BitSet.call(number, 2)
    assert result == 0b10001101

    result = action.BitSet.call(number, 1)
    assert result == 0b10001111


def test_bit_set_tensor():
    number = 0b10001101

    tensor = action.BitSet.as_tensor(2)
    result = tensor(number).numpy()
    assert result == 0b10001101

    tensor = action.BitSet.as_tensor(1)
    result = tensor(number).numpy()
    assert result == 0b10001111


def test_bit_reset():
    number = 0b10001101
    result = action.BitReset.call(number, 2)
    assert result == 0b10001001

    result = action.BitReset.call(number, 1)
    assert result == 0b10001101


def test_bit_reset_tensor():
    number = 0b10001101

    tensor = action.BitReset.as_tensor(2)
    result = tensor(number).numpy()
    assert result == 0b10001001

    tensor = action.BitReset.as_tensor(1)
    result = tensor(number).numpy()
    assert result == 0b10001101


def test_bit_flip():
    number = 0b10001101
    result = action.BitFlip.call(number, 2)
    assert result == 0b10001001

    result = action.BitFlip.call(number, 1)
    assert result == 0b10001111


def test_bit_flip_tensor():
    number = 0b10001101

    tensor = action.BitFlip.as_tensor(2)
    result = tensor(number).numpy()
    assert result == 0b10001001

    tensor = action.BitFlip.as_tensor(1)
    result = tensor(number).numpy()
    assert result == 0b10001111
