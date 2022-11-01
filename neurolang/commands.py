from pathlib import Path
from typing import AbstractSet, Dict, Tuple

import numpy as np
import nibabel
import pandas as pd
from nibabel.dataobj_images import DataobjImage
from nilearn.datasets.utils import _fetch_files
from pandas.errors import ParserError

from .exceptions import InvalidCommandExpression, UnsupportedProgramError
from .expression_pattern_matching import add_match
from .expression_walker import PatternWalker
from .expressions import Command, Constant, Symbol
from .regions import EmptyRegion, ExplicitVBR


class CommandsMixin(PatternWalker):
    @add_match(Command(Symbol("load_csv"), ..., ...))
    def load_csv(self, command):
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

    @add_match(Command(Symbol("load_neuro_image"), ..., ...))
    def load_neuro_image(self, command):
        try:
            if len(command.args) == 3:
                symbol, atlas_url, base_url = command.args
                base_url = _unwrap_expr(base_url)
            elif len(command.args) == 2:
                symbol, atlas_url = command.args
                base_url = None
            if not isinstance(symbol, Symbol):
                symbol = Symbol(symbol)
            atlas_url = _unwrap_expr(atlas_url)
            kwargs = {
                _unwrap_expr(k): _unwrap_expr(v) for k, v in command.kwargs
            }
        except ValueError:
            raise InvalidCommandExpression(
                "Could not extract Symbol name and url from arguments."
            )

        atlas_labels, spatial_image = self._fetch_atlas_map_and_labels(
            atlas_url, None, base_url, kwargs
        )

        self._add_neuro_image(symbol, spatial_image)

    @add_match(Command(Symbol("load_atlas"), ..., ...))
    def load_atlas(self, command):
        """
        Process the `.load_atlas` command. The load_atlas command fetches an
        atlas image with its corresponding labels and loads it into the
        neurolang instance.

        Usage
        -----
        The `.load_csv` command requires three positional arguments:
            - symbol: str, the name of the symbol to load the data into
            - atlas_url: str, the url for the atlas file
            - labels_url: str, the url for the atlas labels
            - base_url: str, optinal. if given, atlas and labels url are
                        considered relative to this base_url
        Other keyword arguments are passed to pandas' `read_csv` method.

        `.load_atlas(destrieux, "destrieux2009_rois_lateralized.nii.gz",
        "destrieux2009_rois_labels_lateralized.csv",
        "https://www.nitrc.org/frs/download.php/11942/destrieux2009.tgz")`

        will load Destrieux's atlas into the destrieux relation.

        Raises
        ------
        InvalidCommandExpression
            raised if command args are invalid or if an error occurs while
            loading the data
        """
        try:
            if len(command.args) == 4:
                symbol, atlas_url, labels_url, base_url = command.args
                base_url = _unwrap_expr(base_url)
            else:
                symbol, atlas_url, labels_url = command.args
                base_url = None
            if not isinstance(symbol, Symbol):
                symbol = Symbol(symbol)
            atlas_url = _unwrap_expr(atlas_url)
            labels_url = _unwrap_expr(labels_url)
            kwargs = {
                _unwrap_expr(k): _unwrap_expr(v) for k, v in command.kwargs
            }
        except ValueError:
            raise InvalidCommandExpression(
                "Could not extract Symbol name and url from arguments."
            )

        atlas_labels, spatial_image = self._fetch_atlas_map_and_labels(
            atlas_url, labels_url, base_url, kwargs
        )

        self._add_atlas_set(symbol, atlas_labels, spatial_image)

    def _fetch_atlas_map_and_labels(
        self, atlas_url, labels_url, base_url, kwargs
    ):
        data_dir = Path.home() / "neurolang_data"
        atlas_labels = None
        if base_url is None:
            opts = {}
            p = Path(atlas_url)
            files = [
                (p.name, p.parent, opts),
            ]
            if atlas_url is not None:
                p = Path(labels_url)
                files.append((p.name, p.parent, opts),)
        else:
            opts = {"uncompress": True}
            files = [(atlas_url, base_url, opts)]
            if labels_url is not None:
                files.append((labels_url, base_url, opts))

        files_ = _fetch_files(data_dir, files)

        if labels_url is not None:
            atlas_labels = pd.read_csv(files_[1], **kwargs)
            atlas_labels = {
                label: name for label, name in atlas_labels.to_records(index=False)
            }
        spatial_image = nibabel.load(files_[0])
        return atlas_labels, spatial_image

    def _add_atlas_set(
        self,
        symbol: Symbol,
        atlas_labels: Dict[int, str],
        spatial_image: DataobjImage,
    ) -> None:
        atlas_set = set()
        for idx, label_name in atlas_labels.items():
            region = (ExplicitVBR.from_spatial_image_label(spatial_image, idx),)
            if isinstance(region, EmptyRegion):
                continue
            atlas_set.add((label_name, region))

        type_ = Tuple[str, ExplicitVBR]
        self.add_extensional_predicate_from_tuples(
            Symbol[AbstractSet[type_]](symbol.name), atlas_set, type_=type_
        )

    def _add_neuro_image(
        self,
        symbol: Symbol,
        spatial_image: DataobjImage,
    ) -> None:

        fdata = spatial_image.get_fdata()
        non_zero_coords = np.transpose(fdata.nonzero())

        coords_xyz = nibabel.affines.apply_affine(spatial_image.affine, non_zero_coords[:, :3])
        data = np.c_[coords_xyz, non_zero_coords[:, 3:], fdata[tuple(non_zero_coords.T)]]

        type_ = Tuple[(float,) * data.shape[1]]
        self.add_extensional_predicate_from_tuples(
            Symbol[AbstractSet[type_]](symbol.name), data, type_=type_
        )

    @add_match(Command)
    def _unknown_command(self, command):
        raise UnsupportedProgramError(
            f"The command statement {command} is not supported."
        )


def _unwrap_expr(expr):
    if isinstance(expr, Constant):
        return expr.value
    return expr.name
