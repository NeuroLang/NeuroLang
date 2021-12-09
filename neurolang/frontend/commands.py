import pandas as pd
from pandas.errors import ParserError

from ..exceptions import InvalidCommandExpression, UnsupportedProgramError
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..expressions import Command, Symbol


class CommandsMixin(PatternWalker):
    @add_match(Command("load_csv", ..., ...))
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

        `.load_csv(Study, "https://github.com/neuroquery/neuroquery_data/raw/master/data/data-neuroquery_version-1_metadata.tsv.gz", sep="\t")`
        will load NeuroQuery's metadata table into the Study relation

        Probabilistic choices or facts can be loaded using the `probabilistic`
        parameter:

        `.load_csv(Study, "...", probabilistic="choice")`
        loads the data as a probabilistic choice relation

        `.load_csv(Study, "...", probabilistic="facts")`
        loads the data as a probabilistic fact relation

        Raises
        ------
        InvalidCommandExpression
            raised if command args are invalid or if an error occurs while loading the data
        """
        try:
            symbol, url = command.args
            if not isinstance(symbol, Symbol):
                symbol = Symbol(symbol)
        except ValueError:
            raise InvalidCommandExpression(
                "Could not extract Symbol name and url from arguments."
            )
        probabilistic = command.kwargs.pop("probabilistic", None)
        try:
            data = pd.read_csv(url, **command.kwargs)
        except ParserError as e:
            raise InvalidCommandExpression(
                f"An error occured while parsing data from {url}: "
            ) from e
        data = data.rename(columns={n: i for i, n in enumerate(data.columns)})

        if probabilistic is None:
            self.add_extensional_predicate_from_tuples(symbol, data)
        elif probabilistic.lower() == "choice":
            self.add_probabilistic_choice_from_tuples(symbol, data)
        else:
            self.add_probabilistic_facts_from_tuples(symbol, data.to_records(index=False).tolist())

    @add_match(Command)
    def unknown_command(self, command):
        raise UnsupportedProgramError(
            f"The command statement {command} is not supported."
        )
