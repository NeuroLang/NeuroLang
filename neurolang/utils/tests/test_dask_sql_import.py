import typing
from neurolang.type_system import Unknown
from neurolang.utils import NamedRelationalAlgebraFrozenSet


def test_proper_dask_sql_import():
    """
    Some documentation on this test:
    The dask_sql library uses jpype to start a JVM and access Java objects
    from python. This JVM needs to be started at import time, otherwise
    it can cause major side-effects which are hard to track. See
    https://github.com/jpype-project/jpype/issues/933 for reference.

    So we need to make sure that dask_sql is loaded very early, even if
    we're not using the dask backend (as demonstrated by this test). So
    there is an `import dask_sql` statement at the top of
    `neurolang.utils.relational_algebra_set.__init__.py`. Removing this
    statement will cause this test to fail, as dask_sql will then not be
    imported at the top of this test file (due to
    `NamedRelationalAlgebraFrozenSet` import) but in the middle of the test.

    As to the exact reason why this test fails when dask_sql is late imported
    I have no idea.
    """
    a = typing.Tuple[Unknown]
    b = typing.Tuple[Unknown]

    assert a is b

    import dask_sql

    c = typing.Tuple[Unknown]

    assert c is a
