from concurrent.futures import Future
import json
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
from typing import Any, Dict


class CustomQueryResultsEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, QueryResults):
            return obj.__dict__
        return super().default(obj)


class QueryResults:
    def __init__(self, uuid: str, future: Future):
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
                        result = self.get_result_item_values(ras)
                        self.results[key] = result

    def get_result_item_values(
        self, ras: NamedRelationalAlgebraFrozenSet
    ) -> Dict:
        result = {}
        result["row_type"] = str(ras.row_type)
        result["columns"] = ras.columns
        df = ras.as_pandas_dataframe()
        result["size"] = df.shape[0]
        return result
