import operator
from abc import ABC, abstractmethod


class Semiring(ABC):
    """Abstract base for (T, ⊕, ⊗, 0, 1) semirings used in provenance
    computation over probabilistic relational algebra.
    """

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        return self._name

    @abstractmethod
    def add(self, a, b):
        """Semiring addition (⊕): sum or max over values."""

    @abstractmethod
    def mul(self, a, b):
        """Semiring multiplication (⊗): product of values."""

    @abstractmethod
    def zero(self):
        """Additive identity element."""

    @abstractmethod
    def one(self):
        """Multiplicative identity element."""

    @abstractmethod
    def agg_add(self):
        """Return the aggregation callable (e.g. ``sum`` or ``max``)."""

    def __repr__(self):
        return f"<Semiring: {self._name}>"


class ProbabilitySemiring(Semiring):
    """Probability semiring: (+, ×) over [0, 1]."""

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def zero(self):
        return 0.0

    def one(self):
        return 1.0

    def agg_add(self):
        return sum


class MaxProductSemiring(Semiring):
    """Max-product semiring: (max, ×) over [0, 1]."""

    def add(self, a, b):
        return max(a, b)

    def mul(self, a, b):
        return a * b

    def zero(self):
        return 0.0

    def one(self):
        return 1.0

    def agg_add(self):
        return max


__all__ = [
    "Semiring",
    "ProbabilitySemiring",
    "MaxProductSemiring",
]
