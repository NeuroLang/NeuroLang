import numpy as np
from nilearn import datasets

import nibabel as nib

from ._backend import BACKENDS, use_backend


class TimeDatalogJoin:

    """
    Time a four-way join solved through the ``NeurolangDL`` frontend.

    Only solving the program is timed; loading the relations and parsing
    the program happen in ``setup``.
    """

    params = [[1000, 3000], BACKENDS]
    param_names = ['n', 'backend']
    timeout = 5 * 60

    def setup(self, n, backend):
        use_backend(backend)
        from neurolang.frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(i, i % 50) for i in range(n)], name='a')
        nl.add_tuple_set([(i % 50, i % 30) for i in range(n)], name='b')
        nl.add_tuple_set([(i % 30, i) for i in range(n)], name='c')
        nl.add_tuple_set([(i,) for i in range(0, n, 10)], name='d')
        nl.execute_datalog_program(
            'q(x, w) :- a(x, y), b(y, z), c(z, w), d(x).'
        )
        self.nl = nl

    def time_four_way_join(self, n, backend):
        """A four-way join of relations with ``n`` tuples."""
        len(self.nl.solve_all()['q'])


class TimeTransitiveClosure:

    """
    Time a recursive reachability query through ``NeurolangDL``.

    The graph is small: the polars backend re-evaluates its lazy query
    plans at every chase iteration, which makes recursion slow on it.
    """

    params = [[20, 40], BACKENDS]
    param_names = ['nodes', 'backend']
    timeout = 5 * 60

    def setup(self, m, backend):
        use_backend(backend)
        from neurolang.frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set(
            [(i, i + 1) for i in range(m)] +
            [(i, (i * 7) % m) for i in range(0, m, 3)],
            name='edge'
        )
        nl.execute_datalog_program(
            'reach(x, y) :- edge(x, y).\n'
            'reach(x, y) :- reach(x, z), edge(z, y).'
        )
        self.nl = nl

    def time_transitive_closure(self, m, backend):
        """Reachability over a chain of ``m`` nodes with shortcuts."""
        len(self.nl.solve_all()['reach'])


class TimeProbabilisticQuery:

    """Time a MARG query solved through the ``NeurolangPDL`` frontend."""

    params = [[200, 400], BACKENDS]
    param_names = ['n', 'backend']
    timeout = 5 * 60

    def setup(self, n, backend):
        use_backend(backend)
        from neurolang.frontend import NeurolangPDL

        nl = NeurolangPDL()
        nl.add_probabilistic_facts_from_tuples(
            [(0.5, i, i % 20) for i in range(n)], name='p'
        )
        nl.add_tuple_set([(i % 20, i % 7) for i in range(n)], name='r')
        nl.add_uniform_probabilistic_choice_over_set(
            [(i,) for i in range(0, n, 5)], name='s'
        )
        self.nl = nl

    def time_marg_query(self, n, backend):
        """``P(z(x, w))`` from probabilistic facts and a choice."""
        with self.nl.scope:
            len(self.nl.execute_datalog_program(
                'z(x, w, PROB) :- p(x, y), r(y, w), s(x).\n'
                'ans(x, w, q) :- z(x, w, q).'
            ))


class TimeRegionComparisons:
    params = [
        [
            'anterior_of', 'posterior_of', 'overlapping',
            'superior_of', 'inferior_of', 'left_of', 'right_of'
        ],
        BACKENDS,
    ]

    param_names = ['direction', 'backend']

    timeout = 10 * 60

    def setup(self, direction, backend):
        # Skipped (asv skips a benchmark whose setup raises
        # NotImplementedError): it downloads the Destrieux atlas and takes
        # minutes. To run it, call use_backend(backend) and self.setup_().
        raise NotImplementedError("Skip benchmark")

    def setup_(self):
        from neurolang import frontend as fe

        nl = fe.NeurolangDL()

        destrieux_atlas = datasets.fetch_atlas_destrieux_2009()
        yeo_atlas = datasets.fetch_atlas_yeo_2011()

        img = nib.load(destrieux_atlas['maps'])
        aff = img.affine
        data = img.get_fdata().astype(int)
        rset = []
        for label, name in destrieux_atlas['labels']:
            if label == 0:
                continue
            voxels = np.transpose((data == label).nonzero())
            if len(voxels) == 0:
                continue
            rset.append((
                name,
                fe.ExplicitVBR(
                    voxels, aff,
                    image_dim=img.shape, prebuild_tree=True
                )
            ))
        nl.add_tuple_set(rset, name='destrieux')

        img = nib.load(yeo_atlas['thick_17'])
        aff = img.affine
        data = img.get_data().squeeze()
        rset = []
        for label in range(1, 18):
            name = str(label)
            if label == 0:
                continue
            voxels = np.transpose((data == label).nonzero())
            if len(voxels) == 0:
                continue
            rset.append((
                name,
                fe.ExplicitVBR(
                    voxels, aff,
                    image_dim=data.shape,
                    prebuild_tree=True
                )
            ))
        nl.add_tuple_set(rset, name='yeo')
        self.nl = nl

    def time_spatial_relation(self, direction, backend):
        with self.nl.scope as e:
            self.nl.query(
                (e.d,),
                e.destrieux(e.d, e.dr) & e.yeo('1', e.dy) &
                e[direction](e.dr, e.dy)
            )
