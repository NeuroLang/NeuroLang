import operator

from .. import expressions
from ..neurolang_compiler import NeuroLangIntermediateRepresentationCompiler
from ..solver import GenericSolver, NumericOperationsSolver

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def associative_command():
    element1 = S_('element1')
    element2 = S_('element2')
    element3 = S_('element3')
    op = S_('op')

    a = F_(op, (F_(op, (element1, element2)), element3))

    b = F_(op, (element1, F_(op, (element2, element3))))
    return a, b


def commutative_command():
    element1 = S_('element1')
    element2 = S_('element2')
    op = S_('op')
    a = F_(op, (element1, element2))
    b = F_(op, (element2, element1))
    return a, b


def identity_command():
    op = S_('op')
    element1 = S_('element1')
    null = S_('null')
    a = element1
    b = F_(op, (element1, null))
    return a, b


def inverse_command():
    op = S_('op')
    element1 = S_('element1')
    null = S_('null')

    a = null
    b = F_(op, (element1, element1))

    return a, b


def left_distributy_command():
    op_dot = S_('op_dot')
    op_cross = S_('op_cross')
    element1 = S_('element1')
    element2 = S_('element2')
    element3 = S_('element3')
    a = F_(op_dot, (element1, F_(op_cross, (element2, element3))))
    b = F_(
        op_cross,
        (F_(op_dot, (element1, element2)), F_(op_dot, (element1, element3)))
    )
    return a, b


def right_distributy_command():
    op_dot = S_('op_dot')
    op_cross = S_('op_cross')
    element1 = S_('element1')
    element2 = S_('element2')
    element3 = S_('element3')
    a = F_(op_dot, (F_(op_cross, (element2, element3)), element1))
    b = F_(
        op_cross,
        (F_(op_dot, (element2, element1)), F_(op_dot, (element3, element1)))
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
        nli.symbol_table['null'] = C_(null)
    check_is_monoid(operation=op, nli=nli, null=null)
    nli.symbol_table['op'] = C_(op)
    check_command(commutative_command, nli)
    nli.symbol_table['op'] = C_(inv_op)
    check_command(inverse_command, nli)

    nli.pop_scope()


def check_is_monoid(operation, nli, null=None):
    nli.push_scope()

    if null is not None:
        nli.symbol_table['null'] = C_(null)
    nli.symbol_table['op'] = C_(operation)
    check_command(associative_command, nli)
    check_command(identity_command, nli)

    nli.pop_scope()


def check_cross_op_is_distributive_with_respect_to_dot_op(cross, dot, nli):
    nli.push_scope()
    nli.symbol_table['op_dot'] = C_(dot)
    nli.symbol_table['op_cross'] = C_(cross)

    check_command(left_distributy_command, nli)
    check_command(right_distributy_command, nli)
    nli.pop_scope()


def check_algebraic_structure_is_a_ring(
    nli,
    op_add=operator.add,
    op_inv_add=operator.sub,
    op_mul=operator.mul,
    op_inv_mul=operator.truediv
):
    check_is_abelian_group(op=op_add, inv_op=op_inv_add, nli=nli, null=0)
    check_is_monoid(operation=op_mul, nli=nli, null=1)
    check_cross_op_is_distributive_with_respect_to_dot_op(
        cross=op_add, dot=op_mul, nli=nli
    )


def test_algebraic_structure_of_naturals():
    elements = (1, 2, 3)
    null_element = 0
    symbols = {
        S_[int]('element{}'.format(i + 1)): C_[int](e)
        for i, e in enumerate(elements)
    }
    symbols[S_[int]('null')] = C_[int](null_element)

    class TheSolver(NumericOperationsSolver[int], GenericSolver):
        pass

    nli = NeuroLangIntermediateRepresentationCompiler(
        solver=TheSolver(), symbols=symbols
    )
    check_algebraic_structure_is_a_ring(
        op_add=operator.add,
        op_inv_add=operator.sub,
        op_mul=operator.mul,
        op_inv_mul=operator.truediv,
        nli=nli
    )
