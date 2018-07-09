from .CD_relations import cardinal_relation
from .regions import Region
from .solver import DatalogSolver


class RegionSolver(DatalogSolver):
    type = Region
    type_name = 'Region'

    def __new__(cls, *args, **kwargs):
        cardinal_operations = {
            'inferior_of': 'I', 'superior_of': 'S',
            'posterior_of': 'P', 'anterior_of': 'A',
            'left_of': 'L', 'right_of': 'R',
            'overlapping': 'O'
        }

        def build_function(relation):
            def f(self, x: Region, y: Region)->bool:
                return bool(cardinal_relation(
                    x, y, relation,
                    refine_overlapping=False,
                    stop_at=None
                ))
            return f

        for key, value in cardinal_operations.items():
            setattr(cls, f'predicate_{key}', build_function(value))

        return DatalogSolver.__new__(cls)
