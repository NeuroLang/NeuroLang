from .. import ast


def test_ASTNode():
    ast_name = 'node'
    ast_children = {'c1': 1, 'c2': 2}
    a = ast.ASTNode(ast_name, ast_children)

    assert a.name == ast_name
    assert ast_children == a
    assert str(a) == ast_name + ' ' + str(ast_children)


def test_ASTWalker():
    ast_child12 = ast.ASTNode('child12', dict())
    ast_child1 = ast.ASTNode('child1', dict(child12=ast_child12))
    ast_root = ast.ASTNode('root', dict(child1=ast_child1))

    walker = ast.ASTWalker()
    assert ast_root == walker.evaluate(ast_root)

    class ASTWalkerTest(ast.ASTWalker):
        def root(self, ast):
            return ast['child1']

    walker = ASTWalkerTest()
    assert ast_child1 == walker.evaluate(ast_root)
    assert [ast_child1] == walker.evaluate([ast_root])
    assert 1 == walker.evaluate(1)
