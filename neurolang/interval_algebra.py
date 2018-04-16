def before(i1, i2) -> bool:
    return i1[0] < i1[1] < i2[0] < i2[1]


def overlaps(i1, i2) -> bool:
    return i1[0] < i2[0] < i1[1] < i2[1]


def during(i1, i2) -> bool:
    return i2[0] < i1[0] < i1[1] < i2[1]


def meets(i1, i2) -> bool:
    return i1[0] < i1[1] == i2[0] < i2[1]


def starts(i1, i2) -> bool:
    return i1[0] == i2[0] < i1[1] < i2[1]


def finishes(i1, i2) -> bool:
    return i2[0] < i1[0] < i1[1] == i2[1]


def equals(i1, i2) -> bool:
    return i1[0] == i2[0] < i1[1] == i2[1]


def converse(operation):
    return lambda x, y: operation(y, x)


def negate(operation):
    return lambda x, y : not operation(x, y)