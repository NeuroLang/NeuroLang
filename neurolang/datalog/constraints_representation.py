"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets and has support for constraints.
"""

from .basic_representation import DatalogProgram


class DatalogConstraintsProgramMixin:

    protected_keywords = set({"__constraints__"})

    def load_constraints(self, union_of_constraints):
        self.symbol_table["__constraints__"] = union_of_constraints

    def get_constraints(self):
        return self.symbol_table["__constraints__"]


class DatalogConstraintsProgram(
    DatalogConstraintsProgramMixin, DatalogProgram
):
    pass
