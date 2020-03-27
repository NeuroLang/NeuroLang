from ...expressions import ExpressionBlock, Symbol, Constant
from ..expression_processing import concatenate_to_expression_block

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


def test_concatenate_to_expression_block():
    block1 = ExpressionBlock((P(x),))
    block2 = ExpressionBlock((Q(z), Q(y), Q(x)))
    block3 = ExpressionBlock((P(z), P(y)))
    assert P(y) in concatenate_to_expression_block(block1, [P(y)]).expressions
    for expression in block2.expressions:
        new_block = concatenate_to_expression_block(block1, block2)
        assert expression in new_block.expressions
        new_block = concatenate_to_expression_block(block1, block2)
        assert expression in new_block.expressions
    for expression in (
        block1.expressions + block2.expressions + block3.expressions
    ):
        new_block = concatenate_to_expression_block(
            concatenate_to_expression_block(block1, block2), block3
        )
        assert expression in new_block.expressions
