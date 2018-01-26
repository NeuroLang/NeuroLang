from .neurolang import SetBasedSolver, Symbol, is_subtype
from pprint import pprint
from io import StringIO


class Sulcus(object):
    def __init__(self, anterior, posterior, inferior, superior):
        self.anterior = anterior
        self.posterior = posterior
        self.inferior = inferior
        self.superior = superior

    def __getitem__(self, k):
        if k in ('anterior', 'posterior', 'inferior', 'superior'):
            return getattr(self, k)

    def intersection(self, sulcus):
        new_anterior = self.anterior.intersection(sulcus.anterior)
        new_posterior = self.posterior.intersection(sulcus.posterior)
        new_inferior = self.inferior.intersection(sulcus.inferior)
        new_superior = self.superior.intersection(sulcus.superior)

        return Sulcus(new_anterior, new_posterior, new_inferior, new_superior)

    def union(self, sulcus):
        new_anterior = self.anterior.union(sulcus.anterior)
        new_posterior = self.posterior.union(sulcus.posterior)
        new_inferior = self.inferior.union(sulcus.inferior)
        new_superior = self.superior.union(sulcus.superior)

        return Sulcus(new_anterior, new_posterior, new_inferior, new_superior)

    def __repr__(self):
        strio = StringIO()
        for a in ('anterior', 'posterior', 'inferior', 'superior'):
            strio.write(a + ": ")
            pprint(getattr(self, a), stream=strio)
        return(strio.getvalue())


class SulcusSolver(SetBasedSolver):
    type = Sulcus
    type_name = "sulcus"
    plural_type_name = "sulci"

    def predicate(self, ast):
        if isinstance(ast['argument'], Symbol):
            argument = ast['argument']
        elif isinstance(ast['argument'], str):
            argument = self.symbol_table[ast['argument']]
        else:
            raise

        if ast['identifier'] in (
            "anterior_to", "posterior_to", "superior_to", "inferior_to"
        ):
            predicate = ast['identifier'][:-3]
            if not is_subtype(argument.type, self.type):
                raise ValueError()
            return argument.value[predicate]
        if ast['identifier'] == 'with_limb':
            if not is_subtype(argument.type, self.type):
                raise
            return argument.value
        else:
            return super().predicate(self, ast)
