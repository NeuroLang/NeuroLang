from concurrent.futures import Future
import json

import numpy as np
import pandas as pd
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from typing import Any, Dict, List, Tuple, Type, Union


class CustomQueryResultsEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, QueryResults):
            return obj.__dict__
        return super().default(obj)


class QueryResults:
    def __init__(
        self,
        uuid: str,
        future: Future,
        page: int = 0,
        limit: int = 50,
        symbol: str = None,
    ):
        self.page = page
        self.limit = limit
        # First, set the uuid and status of the future
        self.uuid = uuid
        self.cancelled = future.cancelled()
        self.running = future.running()
        self.done = future.done()

        # Then, check if it's done
        if future.done():
            error = future.exception()
            if error is not None:
                self.set_error_details(error)
            else:
                results = future.result()
                if results is not None:
                    self.set_results_details(results, symbol)

    def set_error_details(self, error):
        self.message = str(error)
        self.errorName = str(type(error))
        if error.__doc__ is not None:
            self.errorDoc = error.__doc__

    def set_results_details(self, results, symbol):
        self.results = {}
        if symbol is None:
            for key, ras in results.items():
                self.results[key] = self.get_result_item(ras)
        else:
            self.results[symbol] = self.get_result_item(results[symbol])

    def get_result_item(self, ras: NamedRelationalAlgebraFrozenSet) -> Dict:
        """
        Serialize a result symbol into a dict of result values and metadata.

        Parameters
        ----------
        ras : NamedRelationalAlgebraFrozenSet
            the symbol to parse

        Returns
        -------
        Dict
            the parsed result values
        """
        df = ras.as_pandas_dataframe()
        result = self.get_result_item_columns(ras, df)
        values = self.get_result_item_values(ras.row_type, df)
        result["values"] = values
        return result

    def get_result_item_columns(
        self, ras: NamedRelationalAlgebraFrozenSet, df
    ) -> Dict:
        result = {}
        result["row_type"] = str(ras.row_type)
        result["columns"] = ras.columns
        result["size"] = df.shape[0]
        return result

    def get_result_item_values(
        self, row_type: Union[Any, Type[Tuple], None], df: pd.DataFrame
    ) -> List[List]:
        """
        Return the rows of the dataframe corresponding to the requested slice
        (as specified by page and limit values) in a json compatible form.

        Parameters
        ----------
        row_type : Union[Any, Type[Tuple],  None]
            the row_type.
        df : pd.DataFrame
            the dataframe.

        Returns
        -------
        List[List]
            the values.
        """
        rows = df.iloc[self.page * self.limit : (self.page + 1) * self.limit]
        for col, col_type in zip(rows.columns, row_type.__args__):
            if col_type == ExplicitVBR or col_type == ExplicitVBROverlay:
                # TODO: handle regions
                rows.loc[:, col] = rows[col].apply(lambda x: "ExplicitVBR")
            elif rows[col].dtype == np.object_:
                rows.loc[:, col] = rows[col].astype(str)

        return rows.values.tolist()
