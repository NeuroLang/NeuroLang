class ParseException(Exception):
    """
    A problem during the parse stage.
    """

    pass


class AmbiguousSentenceException(ParseException):
    """
    A problem where the sentence got multiple parse trees. The parsed
    trees are attached in the interpretations array.
    """

    def __init__(self, sentence, interpretations):
        self.sentence = sentence
        self.interpretations = interpretations
        super().__init__(
            f"The sentence '{sentence}' has multiple interpretations"
        )


class CouldNotParseException(ParseException):
    """
    A problem which makes impossible to continue parsing the sentence
    and return a result.
    """

    def __init__(self, sentence):
        self.sentence = sentence
        super().__init__(f"The sentence '{sentence}' is not valid")


class TokenizeException(ParseException):
    """
    A problem while converting the string into an array of tokens.
    """

    pass


class ParseDatalogPredicateException(ParseException):
    """
    A problem while parsing an embedded datalog fragment inside
    the controlled natural language.
    """

    pass


class TranslateToDatalogException(Exception):
    """
    A problem during the translation to datalog from DRS.
    """

    pass


class GrammarException(Exception):
    """
    A problem with the grammar definition expression.
    """

    pass
