r"""
Defines Abstract Syntax Tree base classes
for text-parsed language code
=========================================

1- defines an AST node
2- defines an AST walker, able to go through
an AST to convert it to some other representation
"""
import logging
from typing import Any, List, Union

from .. import expressions as ir


class ASTNode(dict):
    """
    Class representing a Node in an
    Abstract Syntax Tree
    """

    def __init__(self, name, children):
        self.name = name
        self.update(children)

    def __str__(self):
        return self.name + " " + str(dict(self))

    def __repr__(self):
        return self.name + " " + dict(self).__repr__()


class ASTWalker(object):
    """
    Base class for Abstract Syntax Tree walkers.
    Walke through an AST translating nodes to
    the corresponding Neurolang intermediate representation
    """

    def evaluate(
        self, ast: Union[ASTNode, List[ASTNode], Any]
    ) -> Union[ir.Expression, List[ir.Expression], Any]:
        """
        Converts input to intermediate representation:
        - if input is an ASTNode, calls -if it exists- the class method
        corresponding to the node type (methods are declared
        in child class)
        - if input is a list of ASTNode, recursively calls itself
        on every node
        - else, simply passes ast through

        Parameters
        ----------
        ast : Union[ASTNode, List[ASTNode], Any]
            input to convert to intermediate representation

        Returns
        -------
        Union[ir.Expression, List[ir.Expression], Any]
            intermediate representation of input
        """
        if isinstance(ast, ASTNode):
            logging.debug("evaluating %s" % ast.name)
            arguments = {k: self.evaluate(v) for k, v in ast.items()}
            new_node = ASTNode(ast.name, arguments)
            if hasattr(self, new_node.name):
                logging.debug("\tdeferring to class method %s" % ast.name)
                return getattr(self, new_node.name)(new_node)
            else:
                logging.debug("\tdeferring to _default method %s" % ast.name)
                return self._default(new_node)
        elif isinstance(ast, list):
            logging.debug("\tevaluating a list of nodes")
            return [self.evaluate(a) for a in ast]
        else:
            logging.debug("\tpassing through")
            return ast

    @staticmethod
    def _default(ast: ASTNode) -> ASTNode:
        """
        Identity evaluation for a node
        whose type isn't covered by the class methods.
        Passes node through

        Parameters
        ----------
        ast : ASTNode
            Node of unsupported type

        Returnss
        -------
        ASTNode
            Input Node passed through
        """
        return ast
