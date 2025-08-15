from numa.poly import degrees


def test_degrees_1():
    assert degrees(1) == [(1, 0), (0, 1)]


def test_degrees_2():
    assert degrees(2) == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]


def test_degrees_3():
    assert degrees(3) == [
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
    ]
