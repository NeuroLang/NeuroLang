import tatsu

from .ast import ASTNode


class TatsuASTConverter(object):
    @staticmethod
    def _default(ast):
        if isinstance(ast, tatsu.ast.AST):
            return ASTNode(
                ast['parseinfo'].rule,
                ast
            )
        else:
            return ast
