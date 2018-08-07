from .. import neurolang as nl
import operator


def associative_command():
    element1 = nl.Symbol('element1')
    element2 = nl.Symbol('element2')
    element3 = nl.Symbol('element3')
    op = nl.Symbol('op')

    a = nl.FunctionApplication(
            op,
            (
                nl.FunctionApplication(op, (element1, element2)),
                element3
            )
        )

    b = nl.FunctionApplication(
            op,
            (
                element1,
                nl.FunctionApplication(op, (element2, element3))
            )
        )
    return a, b


def commutative_command():
    element1 = nl.Symbol('element1')
    element2 = nl.Symbol('element2')
    op = nl.Symbol('op')
    a = nl.FunctionApplication(op, (element1, element2))
    b = nl.FunctionApplication(op, (element2, element1))
    return a, b


def identity_command():
    op = nl.Symbol('op')
    element1 = nl.Symbol('element1')
    null = nl.Symbol('null')
    a = element1
    b = nl.FunctionApplication(op, (element1, null))
    return a, b


def inverse_command():
    op = nl.Symbol('op')
    element1 = nl.Symbol('element1')
    null = nl.Symbol('null')

    a = null
    b = nl.FunctionApplication(op, (element1, element1))

    return a, b


def left_distributy_command():
    op_dot = nl.Symbol('op_dot')
    op_cross = nl.Symbol('op_cross')
    element1 = nl.Symbol('element1')
    element2 = nl.Symbol('element2')
    element3 = nl.Symbol('element3')
    a = nl.FunctionApplication(
            op_dot,
            (
                element1,
                nl.FunctionApplication(
                    op_cross,
                    (element2, element3)
                )
            )
        )
    b = nl.FunctionApplication(
            op_cross,
            (
                nl.FunctionApplication(
                    op_dot,
                    (element1, element2)
                ),
                nl.FunctionApplication(
                    op_dot,
                    (element1, element3)
                )
            )
        )
    return a, b


def right_distributy_command():
    op_dot = nl.Symbol('op_dot')
    op_cross = nl.Symbol('op_cross')
    element1 = nl.Symbol('element1')
    element2 = nl.Symbol('element2')
    element3 = nl.Symbol('element3')
    a = nl.FunctionApplication(
            op_dot,
            (
                nl.FunctionApplication(
                    op_cross,
                    (element2, element3)
                ),
                element1
            )
        )
    b = nl.FunctionApplication(
            op_cross,
            (
                nl.FunctionApplication(
                    op_dot,
                    (element2, element1)
                ),
                nl.FunctionApplication(
                    op_dot,
                    (element3, element1)
                )
            )
        )
    return a, b


def check_command(command, nli):
    a, b = command()
    a_res = nli.compile(a)
    b_res = nli.compile(b)
    assert a_res == b_res


def check_is_abelian_group(op, inv_op, nli, null=None):
    nli.push_scope()
    if null is not None:
        nli.symbol_table['null'] = nl.Constant(null)
    check_is_monoid(operation=op, nli=nli, null=null)
    nli.symbol_table['op'] = nl.Constant(op)
    check_command(commutative_command, nli)
    nli.symbol_table['op'] = nl.Constant(inv_op)
    check_command(inverse_command, nli)

    nli.pop_scope()


def check_is_monoid(operation, nli, null=None):
    nli.push_scope()

    if null is not None:
        nli.symbol_table['null'] = nl.Constant(null)
    nli.symbol_table['op'] = nl.Constant(operation)
    check_command(associative_command, nli)
    check_command(identity_command, nli)

    nli.pop_scope()


def check_cross_op_is_distributive_with_respect_to_dot_op(cross, dot, nli):
    nli.push_scope()
    nli.symbol_table['op_dot'] = nl.Constant(dot)
    nli.symbol_table['op_cross'] = nl.Constant(cross)

    check_command(left_distributy_command, nli)
    check_command(right_distributy_command, nli)
    nli.pop_scope()


def check_algebraic_structure_is_a_ring(
    nli,
    op_add=operator.add, op_inv_add=operator.sub,
    op_mul=operator.mul, op_inv_mul=operator.truediv
):
    check_is_abelian_group(op=op_add, inv_op=op_inv_add, nli=nli, null=0)
    check_is_monoid(operation=op_mul, nli=nli, null=1)
    check_cross_op_is_distributive_with_respect_to_dot_op(cross=op_add,
                                                          dot=op_mul,
                                                          nli=nli)


def test_algebraic_structure_of_naturals():
    elements = (1, 2, 3)
    null_element = 0
    symbols = {
        nl.Symbol[int]('element{}'.format(i + 1)): nl.Constant[int](e)
        for i, e in enumerate(elements)
    }
    symbols[nl.Symbol[int]('null')] = nl.Constant[int](null_element)

    class TheSolver(
        nl.NumericOperationsSolver[int],
        nl.GenericSolver
    ):
        pass

    nli = nl.NeuroLangIntermediateRepresentationCompiler(
        solver=TheSolver(),
        symbols=symbols
    )
    check_algebraic_structure_is_a_ring(
        op_add=operator.add, op_inv_add=operator.sub,
        op_mul=operator.mul, op_inv_mul=operator.truediv,
        nli=nli
    )
