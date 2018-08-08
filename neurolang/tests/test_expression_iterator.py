from .. import expression_walker
from .. import expressions


def test_leaves():
    for expression in (
            expressions.Constant(1),
            expressions.Symbol('a')
    ):
        walk = list(expression_walker.expression_iterator(expression))
        assert walk == [(None, expression)]

    for expression in (
            expressions.Constant(1),
            expressions.Symbol('a')
    ):
        for dfs in (True, False):
            walk = list(expression_walker.expression_iterator(
                expression, include_level=True, dfs=dfs
            ))
            assert walk == [(None, expression, 0)]


def test_tuple():
    expression = expressions.Constant((expressions.Symbol('b'),  expressions.Symbol('c')))

    for dfs in (True, False):
        walk = list(expression_walker.expression_iterator(expression))
        res = [
            (None, expression, 0),
            (None, expression.value[0], 1), (None, expression.value[1], 1)
        ]
        assert walk == [r[:2] for r in res]

        walk = list(expression_walker.expression_iterator(
            expression, include_level=True, dfs=dfs
        ))
        assert walk == res


def test_function_application():
    expression = expressions.FunctionApplication(
            expressions.Symbol('a'),
            (expressions.Symbol('b'),  expressions.Symbol('c'))
    )

    for dfs in (True, False):
        walk = list(expression_walker.expression_iterator(expression))
        res = [
            (None, expression, 0),
            ('functor', expression.functor, 1), ('args', expression.args, 1),
            (None, expression.args[0], 2), (None, expression.args[1], 2)
        ]
        assert walk == [r[:2] for r in res]

        walk = list(expression_walker.expression_iterator(
            expression, include_level=True, dfs=dfs
        ))
        assert walk == res


def test_function_application_nested_dfs():
    expression = expressions.FunctionApplication(
            expressions.Symbol('a'),
            (
                expressions.FunctionApplication(
                    expressions.Symbol('c'), tuple()
                ),
                expressions.Symbol('b')
            )
    )

    walk = list(expression_walker.expression_iterator(expression))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1), ('args', expression.args, 1),
        (None, expression.args[0], 2),
        ('functor', expression.args[0].functor, 3),
        ('args', expression.args[0].args, 3),
        (None, expression.args[1], 2),
    ]

    assert walk == [r[:2] for r in res]
    walk = list(expression_walker.expression_iterator(
        expression, include_level=True
    ))
    assert walk == res


def test_function_application_nested_bfs():
    expression = expressions.FunctionApplication(
            expressions.Symbol('a'),
            (
                expressions.FunctionApplication(
                    expressions.Symbol('c'), tuple()
                ),
                expressions.Symbol('b')
            )
    )

    walk = list(expression_walker.expression_iterator(expression, dfs=False))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1), ('args', expression.args, 1),
        (None, expression.args[0], 2), (None, expression.args[1], 2),
        ('functor', expression.args[0].functor, 3),
        ('args', expression.args[0].args, 3),
    ]

    assert walk == [r[:2] for r in res]

    walk = list(expression_walker.expression_iterator(
        expression, dfs=False, include_level=True
    ))
    assert walk == res
