import base64
import json
from concurrent.futures import Future
from typing import Any, Dict, List, Tuple, Type, Union
from nibabel.nifti1 import Nifti1Image

import numpy as np
import pandas as pd
from neurolang.regions import ExplicitVBR, ExplicitVBROverlay
from neurolang.type_system import get_args
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)


def base64_encode_nifti(image):
    """Returns base64 encoded string of the specified image.

    Parameters
    ----------
    image : nibabel.Nifti2Image
        image to be encoded.

    Returns
    -------
    str
        base64 encoded string of the image.
    """
    encoded_image = base64.encodebytes(image.to_bytes())
    enc = encoded_image.decode("utf-8")
    return enc


def base64_encode_vbr(vbr: Union[ExplicitVBR, ExplicitVBROverlay]):
    """Returns base64 encoded string of the ExplicitVBR.

    Parameters
    ----------
    vrb : Union[ExplicitVBR, ExplicitVBROverlay]
        volumetric brain region to be encoded.

    Returns
    -------
    str
        base64 encoded string of the vbr.
    """
    image = vbr.spatial_image()
    nifti_image = Nifti1Image(
        np.asanyarray(image.dataobj, dtype=np.float32), affine=image.affine
    )
    return base64_encode_nifti(nifti_image)


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
        symbol: str = None,
        start: int = 0,
        length: int = 50,
        sort: int = -1,
        asc: bool = True,
    ):
        self.start = start
        self.length = length
        self.sort = sort
        self.asc = asc
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
        result["row_type"] = [str(t) for t in get_args(ras.row_type)]
        result["columns"] = list(ras.columns)
        result["size"] = df.shape[0]
        return result

    def get_result_item_values(
        self, row_type: Union[Any, Type[Tuple], None], df: pd.DataFrame
    ) -> List[List]:
        """
        Return the rows of the dataframe corresponding to the requested slice
        (as specified by start, length & sort values) in a json compatible form.

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
        if self.sort > -1:
            df = df.sort_values(by=[df.columns[self.sort]], ascending=self.asc)
        rows = df.iloc[self.start : self.start + self.length].copy()
        for col, col_type in zip(rows.columns, row_type.__args__):
            if col_type == ExplicitVBR or col_type == ExplicitVBROverlay:
                rows.loc[:, col] = rows[col].apply(base64_encode_vbr)
            elif rows[col].dtype == np.object_:
                rows.loc[:, col] = rows[col].astype(str)

        return rows.values.tolist()
