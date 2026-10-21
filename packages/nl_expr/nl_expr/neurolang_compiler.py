class NeuroLangIntermediateRepresentationCompiler(object):
    def __init__(self, solver, symbols=None):
        self.solver = solver
        if symbols is not None:
            for k, v in symbols.items():
                solver.symbol_table[k] = v

    def compile(self, intermediate_representation):
        return self.solver.walk(intermediate_representation)

    def push_scope(self):
        self.solver.symbol_table = self.solver.symbol_table.create_scope()

    def pop_scope(self):
        self.solver.symbol_table = self.solver.symbol_table.enclosing_scope

    @property
    def symbol_table(self):
        return self.solver.symbol_table
