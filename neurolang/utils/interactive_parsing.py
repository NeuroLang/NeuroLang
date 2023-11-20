# from __future__ import annotations

from dataclasses import dataclass
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from lark.parsers.lalr_interactive_parser import InteractiveParser

from ..exceptions import UnexpectedTokenError

# Code based on https://github.com/MegaIng/lark-autocomplete/blob/master/lark_autocomplete.py

# List of the different categories to by displayed in the frontend.
CATEGORIES = [
    'Signs',
    'Numbers',
    'Text',
    'Operators',
    'Cmd_identifier',
    'Functions',
    'Identifier_regexp',
    'Reserved words',
    'Boleans',
    'Expression symbols',
    'Python string',
    'Strings'
]

# Dictionary for the correspondence between the NeuroLang grammar rules and the frontend categories.
TERMINALS_TO_CATEGORIES = {
    'AMPERSAND': CATEGORIES[0],
    'AND_SYMBOL': CATEGORIES[0],
    'ANS': CATEGORIES[7],
    'AT': CATEGORIES[0],
    'CMD_IDENTIFIER': CATEGORIES[11],
    'COLON': CATEGORIES[0],
    'COMMA': CATEGORIES[0],
    'COMPARISON_OPERATOR': CATEGORIES[3],
    'CONDITION_OP': CATEGORIES[3],
    'CONJUNCTION_SYMBOL': CATEGORIES[0],
    'DOT': CATEGORIES[9],
    'DOTS': CATEGORIES[0],
    'EQUAL': CATEGORIES[0],
    'EXISTS': CATEGORIES[7],
    'EXISTS_SYMBOL': CATEGORIES[0],
    'EXISTS_WORD': CATEGORIES[7],
    'FALSE': CATEGORIES[8],
    'FLOAT': CATEGORIES[1],
    'IDENTIFIER_REGEXP': CATEGORIES[11],
    'IMPLICATION': CATEGORIES[9],
    'INT': CATEGORIES[1],
    'LAMBDA': CATEGORIES[5],
    'LPAR': CATEGORIES[0],
    'MINUS': CATEGORIES[3],
    'NEG_UNICODE': CATEGORIES[3],
    'PLUS': CATEGORIES[3],
    'POW': CATEGORIES[3],
    'PROBA_OP': CATEGORIES[9],
    'PYTHON_STRING': CATEGORIES[11],
    'RIGHT_IMPLICATION': CATEGORIES[9],
    'RPAR': CATEGORIES[0],
    'SEMICOLON': CATEGORIES[0],
    'SLASH': CATEGORIES[3],
    'STAR': CATEGORIES[3],
    'STATEMENT_OP': CATEGORIES[9],
    'SUCH_THAT': CATEGORIES[7],
    'SUCH_THAT_WORD': CATEGORIES[7],
    'TEXT': CATEGORIES[11],
    'TILDE': CATEGORIES[3],
    'TRUE': CATEGORIES[8]
}


@dataclass
class CompleteResult:
    pos: int
    prefix: str
    token_options: set

    def to_dictionary(self):
        return {
            "pos": self.pos,
            "prefix": self.prefix,
            # dictionary of the accepted tokens returned by the parser
            "token_options": self.token_options
        }


class LarkCompleter:
    def __init__(self, lark: Lark, start: str = None):
        if not lark.options.parser == "lalr":
            raise TypeError(
                "The given Lark instance must be using the LALR(1) parser")
        self.parser = lark
        self.start = start

    def complete(self, text: str) -> CompleteResult:
        interactive = self.parser.parse_interactive(text, self.start)
        lex_stream = interactive.lexer_thread.lex(interactive.parser_state)
        while True:
            try:
                token = next(lex_stream)
            except StopIteration:
                break
            except UnexpectedCharacters as e:
                return self.compute_options_unexpected_char(interactive, text, e)
            except UnexpectedToken as e:
                raise UnexpectedTokenError(str(e), line=e.line - 1, column=e.column - 1) from e
            interactive.feed_token(token)
        return self.compute_options_no_error(interactive, text)

    def compute_options_unexpected_char(
            self,
            interactive: InteractiveParser, text: str,
            e: UnexpectedCharacters
    ) -> CompleteResult:
        prefix = text[e.pos_in_stream:]
        return CompleteResult(e.pos_in_stream, prefix, interactive.accepts())

    def compute_options_no_error(self, interactive: InteractiveParser, text: str) -> CompleteResult:

        # Initialise the final accepted tokens dictionary
        accepted_tokens = {}
        for category in CATEGORIES:
            accepted_tokens[category] = set()

        # Get the accepted tokens given by the parser
        parser_accepted_tokens = list(interactive.accepts())

        # Remove the end of line accepted token
        if '$END' in parser_accepted_tokens:
            parser_accepted_tokens.remove('$END')

        # Process accepted tokens
        for token in parser_accepted_tokens:

            # Get terminal information
            terminal = self.parser.get_terminal(token)
            t_name = str(terminal.name)
            t_pattern = str(terminal.pattern).replace("'", "")

            if (t_name == 'CMD_IDENTIFIER'):
                t_pattern = '<command identifier>'
                accepted_tokens['commands'] = set()
            elif (t_name == 'FLOAT'):
                t_pattern = '<float>'
            elif (t_name == 'IDENTIFIER_REGEXP'):
                t_pattern = '<identifier regular expression>'
                accepted_tokens['functions'] = set()
                accepted_tokens['base symbols'] = set()
                accepted_tokens['query symbols'] = set()
            elif (t_name == 'INT'):
                t_pattern = '<integer>'
            elif (t_name == 'PYTHON_STRING'):
                t_pattern = '<quoted string>'
            elif (t_name == 'TEXT'):
                t_pattern = '<text>'
            else:
                t_pattern = t_pattern.replace(
                    '\\\\+', '+').replace('\\\\-', '-')

            t_pattern = t_pattern.replace('\\\\b', '\\b').replace('\\\\', '')

            # Add the processed accepted token to the final dictionary
            if ('|' in t_pattern):
                t_pattern = t_pattern.replace(
                    '(?:', '').replace(')', '').split('|')
                for pattern in t_pattern:
                    accepted_tokens[TERMINALS_TO_CATEGORIES[t_name]].add(
                        pattern)
            else:
                accepted_tokens[TERMINALS_TO_CATEGORIES[t_name]].add(t_pattern)

        return CompleteResult(len(text), "", accepted_tokens)