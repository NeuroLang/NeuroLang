import lark

from .ast import ASTNode



class LarkASTConverter(object):

    @staticmethod
    def _default(ast):
        if isinstance(ast, lark.tree.Tree):
            return ASTNode(
                ast['parseinfo'].rule,
                ast
            )
        else:
            return ast
