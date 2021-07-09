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
        self, uuid: str, future: Future, page: int = 0, limit: int = 50
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
                self.message = str(error)
                self.errorName = str(type(error))
                try:
                    self.errorLocation = {
                        "lineNumber": error.from_line + 1,
                        "columnNumber": error.from_col + 1,
                    }
                except AttributeError:  # pragma: no cover
                    pass
            else:
                results = future.result()
                if results is not None:
                    self.results = {}
                    for key, ras in results.items():
                        df = ras.as_pandas_dataframe()
                        result = self.get_result_item_columns(ras, df)
                        values = self.get_result_item_values(ras.row_type, df)
                        result["values"] = values
                        self.results[key] = result

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
                rows[col] = rows[col].apply(lambda x: "ExplicitVBR")
            elif rows[col].dtype == np.object_:
                rows[col] = rows[col].astype(str)

        return rows.values.tolist()