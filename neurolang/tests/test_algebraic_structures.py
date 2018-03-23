from .. import neurolang as nl

associative_command = '''
    a = (element1 {op} element2) {op} element3
    b = element1 {op} (element2 {op} element3)
'''

commutative_command = '''
    a = element1 {op} element2
    b = element2 {op} element1
'''

identity_command = '''
    a = element1
    b = element1 {op} null
'''

inverse_command = '''
    a = null
    b = element1 {op} element1
'''

left_distributy_command = '''
    a = element1 {op_dot} (element2 {op_cross} element3)
    b = (element1 {op_dot} element2) {op_cross} (element1 {op_dot} element3)
'''

right_distributy_command = '''
    a = (element2 {op_cross} element3) {op_dot} element1
    b = (element2 {op_dot} element1) {op_cross} (element3 {op_dot} element1)
'''


def check_command(command, nli):
    nli.compile(nl.parser(command))
    assert nli.symbol_table['a'].value == nli.symbol_table['b'].value


def check_is_abelian_group(op, inv_op, nli, null=None):
    if null is not None:
        old_null = nli.symbol_table['null'].value
        nli.symbol_table['null'].value = null

    check_is_monoid(operation=op, nli=nli)
    check_command(commutative_command.format(op=op), nli)
    check_command(inverse_command.format(op=inv_op), nli)

    # Restore nli
    if null is not None:
        nli.symbol_table['null'].value = old_null


def check_is_monoid(operation, nli, null=None):
    if null is not None:
        old_null = nli.symbol_table['null'].value
        nli.symbol_table['null'].value = null

    check_command(associative_command.format(op=operation), nli)
    check_command(identity_command.format(op=operation), nli)

    # Restore nli
    if null is not None:
        nli.symbol_table['null'].value = old_null


def check_cross_op_is_distributive_with_respect_to_dot_op(cross, dot, nli):
    command = left_distributy_command.format(op_cross=cross, op_dot=dot)
    check_command(command, nli)

    command = right_distributy_command.format(op_cross=cross, op_dot=dot)
    check_command(command, nli)


def check_algebraic_structure_is_a_ring(nli, op_add='+', op_inv_add='-',
                                        op_mul='*', op_inv_mul='/'):
    check_is_abelian_group(op=op_add, inv_op=op_inv_add, nli=nli, null=0)
    check_is_monoid(operation=op_mul, nli=nli, null=1)
    check_cross_op_is_distributive_with_respect_to_dot_op(cross=op_add,
                                                          dot=op_mul,
                                                          nli=nli)


def test_algebraic_structure_of_naturals():
    elements = (1, 2, 3)
    null_element = 0
    symbols = {
        'element{}'.format(i + 1): nl.Constant[int](e)
        for i, e in enumerate(elements)
    }
    symbols['null'] = nl.Constant[int](null_element)
    nli = nl.NeuroLangIntermediateRepresentationCompiler(
        symbols=symbols, types=[(int, 'dummy')]
    )

    check_algebraic_structure_is_a_ring(op_add='+', op_inv_add='-',
                                        op_mul='*', op_inv_mul='/',
                                        nli=nli)
