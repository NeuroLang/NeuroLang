from .expression_processing import (
    CausalInterventionIdentification,
    CausalInterventionRewriter,
)
from ...logic import Union


class CausalProgram:
    def rewrite_program(self, program):
        cii = CausalInterventionIdentification()
        cii.walk(program)
        cir = CausalInterventionRewriter(cii.intervention)

        new_program = cir.walk(program)

        program_rewritten = set()
        for f1 in new_program.formulas:
            if isinstance(f1, Union):
                for f2 in f1.formulas:
                    program_rewritten.add(f2)
            else:
                program_rewritten.add(f1)

        return program_rewritten.union(cir.new_facts), cir.new_query

