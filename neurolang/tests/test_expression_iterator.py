from .. import expression_walker
from .. import expressions


def test_leaves():
    for expression in (
            expressions.Constant(1),
            expressions.Symbol('a')
    ):
        walk = list(expression_walker.expression_dfs_iterator(expression))
        assert walk == [(None, expression)]

    for expression in (
            expressions.Constant(1),
            expressions.Symbol('a')
    ):
        walk = list(expression_walker.expression_dfs_iterator(
            expression, include_level=True
        ))
        assert walk == [(None, expression, 0)]


def test_function_application():
    expression = expressions.FunctionApplication(
            expressions.Symbol('a'),
            (expressions.Symbol('b'),  expressions.Symbol('c'))
    )

    walk = list(expression_walker.expression_dfs_iterator(expression))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1), ('args', expression.args, 1),
        (None, expression.args[0], 2), (None, expression.args[1], 2)
    ]
    assert walk == [r[:2] for r in res]

    walk = list(expression_walker.expression_dfs_iterator(
        expression, include_level=True
    ))
    assert walk == res


def test_function_application_nested():
    expression = expressions.FunctionApplication(
            expressions.Symbol('a'),
            (
                expressions.Symbol('b'),
                expressions.FunctionApplication(
                    expressions.Symbol('c'), tuple()
                )
            )
    )

    walk = list(expression_walker.expression_dfs_iterator(expression))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1), ('args', expression.args, 1),
        (None, expression.args[0], 2), (None, expression.args[1], 2),
        ('functor', expression.args[1].functor, 3),
        ('args', expression.args[1].args, 3),
    ]

    assert walk == [r[:2] for r in res]
    walk = list(expression_walker.expression_dfs_iterator(
        expression, include_level=True
    ))
    assert walk == res
