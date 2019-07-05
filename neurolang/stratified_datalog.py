class StratifiedDatalog():

    _precedence_graph = {}

    def solve(self, datalog_program):

        for v in datalog_program.intensional_database().values():
            for rule in v.expressions:
                self._add_node(rule)


    def _add_node(self, implication):

        for symbol_ant in implication.antecedent:
            for symbol_con in implication.consequent:
                self._precedence_graph[symbol_ant.name] = symbol_con

        print(self._precedence_graph)
        romper
