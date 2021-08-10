import base64
import hashlib
import json
from concurrent.futures import Future
from neurolang.frontend.query_resolution_expressions import Symbol
from typing import Any, Dict, List, Tuple, Type, Union
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from tatsu.exceptions import FailedParse

import numpy as np
import pandas as pd
import nibabel as nib
from neurolang.regions import EmptyRegion, ExplicitVBR, ExplicitVBROverlay
from neurolang.type_system import get_args
from neurolang.utils.relational_algebra_set import (
    RelationalAlgebraFrozenSet,
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


def base64_encode_spatial(image: SpatialImage):
    """Returns base64 encoded string of a spatial image

    Parameters
    ----------
    image : nibabel.spatialimages.SpatialImage
        spatial image to be encoded.

    Returns
    -------
    str
        base64 encoded string of the vbr.
    """
    nifti_image = Nifti1Image(
        np.asanyarray(image.dataobj, dtype=np.float32), affine=image.affine
    )
    return base64_encode_nifti(nifti_image)


def calculate_image_center(image: SpatialImage):
    """Calculates center coordinates for the specified image."""
    coords = np.transpose(image.get_fdata().nonzero()).mean(0).astype(int)
    coords = nib.affines.apply_affine(image.affine, coords)
    return [int(c) for c in coords]


def serializeVBR(vbr: Union[ExplicitVBR, ExplicitVBROverlay]):
    """
    Serialize a Volumetric Brain Region object.

    Parameters
    ----------
    vbr : Union[ExplicitVBR, ExplicitVBROverlay]
        the volumetric brain region to serialize

    Returns
    -------
    Dict
        a dict containing the base64 encoded image, as well as min and max
        values, and a hash of the image.
    """
    if isinstance(vbr, EmptyRegion):
        return "Empty Region"

    image = vbr.spatial_image()
    flattened = image.get_fdata().flatten()
    min = flattened[flattened != 0].min()
    max = flattened.max()
    hash = hashlib.sha224(image.dataobj.tobytes()).hexdigest()
    return {
        "min": min,
        "max": max,
        "image": base64_encode_spatial(image),
        "hash": hash,
        "center": calculate_image_center(image),
    }


class CustomQueryResultsEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, QueryResults):
            return obj.__dict__
        return super().default(obj)


class QueryResults:
    """
    A representation of query results. This class is returned by
    the tornado application as a JSON serialized string.

    It contains
        * metadata information about the query:
            - cancelled, running, done
        * metadata about the requested information:
            - the specific symbol to return, as well as
            - start, length, sort, asc (parameters for which rows to return)
        * if the query resulted in an error, the error details
        * if the query resulted in data, the details for each requested symbol:
            - its columns, size, & row_type
            - the rows corresponding to the start, length, sort & asc params
    """

    def __init__(
        self,
        uuid: str,
        future: Future,
        symbol: str = None,
        start: int = 0,
        length: int = 50,
        sort: int = -1,
        asc: bool = True,
        get_values: bool = False,
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
                    self.set_results_details(results, symbol, get_values)

    def set_error_details(self, error):
        self.errorName = str(type(error))
        if isinstance(error, FailedParse):
            self.message = "An error occured while parsing your query."
            self.errorDoc = str(error)
            try:
                line_info = error.tokenizer.line_info(error.pos)
            except AttributeError:
                # support tatsu 4.x
                line_info = error.buf.line_info(error.pos)
            self.line_info = {
                "line": line_info.line,
                "col": line_info.col,
                "text": error.message,
            }
        else:
            self.message = str(error)
            if error.__doc__ is not None:
                self.errorDoc = error.__doc__

    def set_results_details(self, results, symbol, get_values):
        self.results = {}
        if symbol is None:
            for key, ras in results.items():
                self.results[key] = self.get_result_item(ras, get_values)
        else:
            self.results[symbol] = self.get_result_item(results[symbol], True)

    def get_result_item(
        self,
        symbol: Union[RelationalAlgebraFrozenSet, Symbol],
        get_item_values: bool = False,
    ) -> Dict:
        """
        Serialize a symbol into a dict of result values and metadata.

        Parameters
        ----------
        symbol : Union[RelationalAlgebraFrozenSet, Symbol]
            the symbol to parse
        get_item_values : bool
            get the values for the item

        Returns
        -------
        Dict
            the parsed result values
        """
        if isinstance(symbol, RelationalAlgebraFrozenSet):
            df = symbol.as_pandas_dataframe()
            result = self.get_result_item_columns(symbol, df)
            if get_item_values:
                values = self.get_result_item_values(symbol.row_type, df)
                result["values"] = values
        elif isinstance(symbol, Symbol):
            result = self.get_function_metadata(symbol)
        return result

    def get_function_metadata(self, symbol: Symbol):
        result = {
            "type": str(symbol.type),
            "doc": symbol.value.__doc__,
            "function": True,
        }
        return result

    def get_result_item_columns(
        self, symbol: RelationalAlgebraFrozenSet, df
    ) -> Dict:
        result = {}
        result["row_type"] = [str(t) for t in get_args(symbol.row_type)]
        result["columns"] = [str(c) for c in symbol.columns]
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
                rows.loc[:, col] = rows[col].apply(serializeVBR)
            elif rows[col].dtype == np.object_:
                rows.loc[:, col] = rows[col].astype(str)

        return rows.values.tolist()
