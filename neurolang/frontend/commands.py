import pandas as pd
from pandas.errors import ParserError

from ..exceptions import InvalidCommandExpression, UnsupportedProgramError
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..expressions import Command, Constant, Symbol


class CommandsMixin(PatternWalker):
    @add_match(Command(Symbol("load_csv"), ..., ...))
    def load_csv_command(self, command):
        """
        Process the `.load_csv` command. The load_csv command fetches tabular
        data from a given url (using pandas' `read_csv` method) and loads it
        into a relation with the given Symbol name.

        Usage
        -----
        The `.load_csv` command requires two positional arguments:
            - symbol: str, the name of the symbol to load the data into
            - url: str, the url to fetch the data from
        Other keyword arguments are passed to pandas' `read_csv` method with
        limited support (only string values).

        `.load_csv(Study, "https://github.com/neuroquery/neuroquery_data/raw/
        master/data/data-neuroquery_version-1_metadata.tsv.gz", sep="\t")`
        will load NeuroQuery's metadata table into the Study relation

        Raises
        ------
        InvalidCommandExpression
            raised if command args are invalid or if an error occurs while
            loading the data
        """
        try:
            symbol, url = command.args
            if not isinstance(symbol, Symbol):
                symbol = Symbol(symbol)
            url = _unwrap_expr(url)
            kwargs = {
                _unwrap_expr(k): _unwrap_expr(v) for k, v in command.kwargs
            }
        except ValueError:
            raise InvalidCommandExpression(
                "Could not extract Symbol name and url from arguments."
            )
        try:
            data = pd.read_csv(url, **kwargs)
        except ParserError as e:
            raise InvalidCommandExpression(
                f"An error occured while parsing data from {url}: "
            ) from e
        data = data.rename(columns={n: i for i, n in enumerate(data.columns)})
        self.add_extensional_predicate_from_tuples(symbol, data)

    @add_match(Command)
    def unknown_command(self, command):
        raise UnsupportedProgramError(
            f"The command statement {command} is not supported."
        )


def _unwrap_expr(expr):
    if isinstance(expr, Constant):
        return expr.value
    return expr.name
