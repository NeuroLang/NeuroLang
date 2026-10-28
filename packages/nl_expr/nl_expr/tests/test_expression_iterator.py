from .. import expression_walker
from .. import expressions

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def test_leaves():
    for expression in (C_(1), S_('a')):
        walk = list(expression_walker.expression_iterator(expression))
        assert walk == [(None, expression)]

    for expression in (C_(1), S_('a')):
        for dfs in (True, False):
            walk = list(
                expression_walker.expression_iterator(
                    expression, include_level=True, dfs=dfs
                )
            )
            assert walk == [(None, expression, 0)]


def test_tuple():
    expression = C_((S_('b'), S_('c')))

    for dfs in (True, False):
        walk = list(expression_walker.expression_iterator(expression))
        res = [
            (None, expression, 0),
            (None, expression.value[0], 1),
            (None, expression.value[1], 1)
        ]
        assert walk == [r[:2] for r in res]

        walk = list(
            expression_walker.expression_iterator(
                expression, include_level=True, dfs=dfs
            )
        )
        assert walk == res


def test_function_application():
    expression = F_(S_('a'), (S_('b'), S_('c')))

    for dfs in (True, False):
        walk = list(expression_walker.expression_iterator(expression))
        res = [
            (None, expression, 0),
            ('functor', expression.functor, 1),
            ('args', expression.args, 1),
            (None, expression.args[0], 2),
            (None, expression.args[1], 2)
        ]
        assert walk == [r[:2] for r in res]

        walk = list(
            expression_walker.expression_iterator(
                expression, include_level=True, dfs=dfs
            )
        )
        assert walk == res


def test_function_application_nested_dfs():
    expression = F_(S_('a'), (F_(S_('c'), tuple()), S_('b')))

    walk = list(expression_walker.expression_iterator(expression))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1),
        ('args', expression.args, 1),
        (None, expression.args[0], 2),
        ('functor', expression.args[0].functor, 3),
        ('args', expression.args[0].args, 3),
        (None, expression.args[1], 2),
    ]

    assert walk == [r[:2] for r in res]
    walk = list(
        expression_walker.expression_iterator(expression, include_level=True)
    )
    assert walk == res


def test_function_application_nested_bfs():
    expression = F_(S_('a'), (F_(S_('c'), tuple()), S_('b')))

    walk = list(expression_walker.expression_iterator(expression, dfs=False))
    res = [
        (None, expression, 0),
        ('functor', expression.functor, 1),
        ('args', expression.args, 1),
        (None, expression.args[0], 2),
        (None, expression.args[1], 2),
        ('functor', expression.args[0].functor, 3),
        ('args', expression.args[0].args, 3),
    ]

    assert walk == [r[:2] for r in res]

    walk = list(
        expression_walker.expression_iterator(
            expression, dfs=False, include_level=True
        )
    )
    assert walk == res
