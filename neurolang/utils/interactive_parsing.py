from __future__ import annotations

from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from lark import Lark, Transformer, UnexpectedCharacters
from lark.exceptions import UnexpectedToken
from lark.parsers.lalr_interactive_parser import InteractiveParser

from dataclasses import dataclass


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
    'Python string'
]


TERMINALS_TO_CATEGORIES = {
    'AMPERSAND' : CATEGORIES[0],
    'AND_SYMBOL' : CATEGORIES[0],
    'ANS' : CATEGORIES[7],
    'AT' : CATEGORIES[0],
    'CMD_IDENTIFIER' : CATEGORIES[4],
    'COLON' : CATEGORIES[0],
    'COMMA' : CATEGORIES[0],
    'COMPARISON_OPERATOR' : CATEGORIES[3],  # TO DO : test the type of the collection and
    'CONDITION_OP' : CATEGORIES[3],
    'CONJUNCTION_SYMBOL' : CATEGORIES[0],
    'DOT' : CATEGORIES[9],
    'DOTS' : CATEGORIES[0],
    'EQUAL' : CATEGORIES[0],
    'EXISTS' : CATEGORIES[7],
    'EXISTS_SYMBOL' : CATEGORIES[0],
    'EXISTS_WORD' : CATEGORIES[7],
    'FALSE' : CATEGORIES[8],
    'FLOAT' : CATEGORIES[1],
    'IDENTIFIER_REGEXP' : CATEGORIES[6],
    'IMPLICATION' : CATEGORIES[9],
    'INT' : CATEGORIES[1],
    'LAMBDA' : CATEGORIES[5],
    'LPAR' : CATEGORIES[0],
    'MINUS' : CATEGORIES[3],
    'NEG_UNICODE' : CATEGORIES[3],
    'PLUS': CATEGORIES[3],
    'POW' : CATEGORIES[3],
    'PROBA_OP' : CATEGORIES[9],
    'PYTHON_STRING' : CATEGORIES[10],
    'RIGHT_IMPLICATION' : CATEGORIES[9],
    'RPAR' : CATEGORIES[0],
    'SEMICOLON' : CATEGORIES[0],
    'SLASH' : CATEGORIES[3],
    'STAR' : CATEGORIES[3],
    'STATEMENT_OP' : CATEGORIES[9],
    'SUCH_THAT' : CATEGORIES[7],
    'SUCH_THAT_WORD' : CATEGORIES[7],
    'TEXT' : CATEGORIES[2],
    'TILDE' : CATEGORIES[3],
    'TRUE': CATEGORIES[8]
}


@dataclass
class CompleteResult:
    pos: int
    prefix: str
    token_options: set[str]

    def to_dictionary(self):
        return {"pos" : self.pos, "prefix" : self.prefix, "token_options" : self.token_options}


class LarkCompleter:
    def __init__(self, lark: Lark, start: str = None):
        if not lark.options.parser == "lalr":
            raise TypeError("The given Lark instance must be using the LALR(1) parser")
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
            interactive.feed_token(token)
        return self.compute_options_no_error(interactive, text)

    def compute_options_unexpected_char(self, interactive: InteractiveParser, text: str,
                                        e: UnexpectedCharacters) -> CompleteResult:
        prefix = text[e.pos_in_stream:]

        return CompleteResult(e.pos_in_stream, prefix, interactive.accepts())

    def compute_options_no_error(self, interactive: InteractiveParser, text: str) -> CompleteResult:
        print("")
        print("___in compute_options_no_error()___")
        # # print("accepts :", accepts)
        # # print("type(accepts) :", type(accepts))
        # accepts = {}
        # for f in interactive.accepts():
        #     term_def = next(t for t in self.parser.terminals if t.name == f)
        #     # print(term_def.name)
        #     # print(str(term_def.pattern).replace("'", ""))
        #     accepts[term_def.name] = str(term_def.pattern).replace("'", "")





        toks = {}
        for i in CATEGORIES:
            toks[i] = []


        accepts = []
        a = list(interactive.accepts())
        if '$END' in a:
            a.remove('$END')
        for e in a:
            # a.remove(e) #data.discard(value) if you don't care if the item exists.
            t = self.parser.get_terminal(e)
            n = str(t.name)
            # print("n :", n)
            pat = str(t.pattern).replace("'", "")
            # toks[TERMINALS_TO_CATEGORIES[n]].append(pat)
            # toks['numbers'].append('integer')

            accepts.append(n + ' : ' + pat)

            if (n == 'FLOAT'):
                pat = 'float'
            elif (n == 'INT'):
                pat = 'integer'
            elif (n == 'IDENTIFIER_REGEXP'):
                pat = pat.replace('\\\\+', '\\+').replace('\\\\-', '\\-').replace('\\\\/', '\\/').replace('\\\\.', '\\.')
            else:
                pat = pat.replace('\\\\+', '+').replace('\\\\-', '-')
            # if ((n != 'FLOAT') & (n != 'DOUBLE_QUOTE')):
            pat = pat.replace('\\\\b', '\\b').replace('\\\\', '')
            # print("pat :", pat)
            # print("TERMINALS_TO_CATEGORIES[n] :", TERMINALS_TO_CATEGORIES[n])
            if ('|' in pat):
                pat = pat.replace('(?:', '').replace(')', '').split('|')
                for i in pat:
                    toks[TERMINALS_TO_CATEGORIES[n]].append(i)
            else:
                toks[TERMINALS_TO_CATEGORIES[n]].append(pat)
            # print("toks[TERMINALS_TO_CATEGORIES[n]] :", toks[TERMINALS_TO_CATEGORIES[n]])
        print("")
        for i in toks:
            print(i, ":", str(toks[i]))

        # print(accepts)
        # print("type(accepts) :", type(accepts))
        print("")
        return CompleteResult(len(text), "", toks)
