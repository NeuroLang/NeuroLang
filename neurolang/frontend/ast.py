import logging


class ASTNode(dict):
    def __init__(self, name, children):
        self.name = name
        self.update(children)

    def __str__(self):
        return self.name + ' ' + str(dict(self))

    def __repr__(self):
        return self.name + ' ' + dict(self).__repr__()


class ASTWalker(object):
    def evaluate(self, ast):
        if isinstance(ast, ASTNode):
            logging.debug("evaluating %s" % ast.name)
            arguments = {
                k: self.evaluate(v)
                for k, v in ast.items()
            }
            new_node = ASTNode(ast.name, arguments)
            if hasattr(self, new_node.name):
                logging.debug('\tdeferring to class method %s' % ast.name)
                return getattr(self, new_node.name)(new_node)
            else:
                logging.debug('\tdeferring to _default method %s' % ast.name)
                return self._default(new_node)
        elif isinstance(ast, list):
            logging.debug("\tevaluating a list of nodes")
            return [self.evaluate(a) for a in ast]
        else:
            logging.debug("\tpassing through")
            return ast

    @staticmethod
    def _default(ast):
        return ast
