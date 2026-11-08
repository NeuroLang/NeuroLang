import re
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import AbstractSet, Dict, Tuple

import nibabel
import pandas as pd
from nibabel.dataobj_images import DataobjImage
from pandas.errors import ParserError

from nl_expr.exceptions import (
    InvalidCommandExpression,
    NeuroLangException,
    ParserError as DatalogParserError,
    UnsupportedProgramError,
)
from nl_expr.expression_pattern_matching import add_match
from nl_expr.expression_walker import PatternWalker
from nl_expr.expressions import Command, Constant, Query, Symbol
from .regions import EmptyRegion, ExplicitVBR
from .utils.nilearn_compat import fetch_files as _fetch_files

# The modules imported in a scope (see CommandsMixin.import_module).
IMPORTED_MODULES = Symbol("__imported_modules__")

# A module of a Python package, ``package:path/module.dl``.
_PACKAGE_RESOURCE = re.compile(r"([A-Za-z_][\w.]+):(.+)")


class DatalogModuleError(InvalidCommandExpression):
    """A Datalog module could not be imported."""


class CommandsMixin(PatternWalker):
    protected_keywords = {IMPORTED_MODULES.name}

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
                f"An error occurred while parsing data from {url}: "
            ) from e
        data = data.rename(columns={n: i for i, n in enumerate(data.columns)})
        self.add_extensional_predicate_from_tuples(symbol, data)

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
            - base_url: str, optional. if given, atlas and labels url are
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
        if base_url is None:
            opts = {}
            p = Path(atlas_url)
            files = [
                (p.name, p.parent, opts),
            ]
            p = Path(labels_url)
            files.append((p.name, p.parent, opts),)
        else:
            opts = {"uncompress": True}
            files = [(atlas_url, base_url, opts), (labels_url, base_url, opts)]

        files_ = _fetch_files(data_dir, files)
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

    @add_match(Command(Symbol("import"), ..., ...))
    def import_module(self, command):
        """
        Process the `.import` command, which runs the statements of a
        Datalog module, a file, as if they were written in its place:
        its facts, rules and commands, ``.import`` included.

        Usage
        -----
        `.import("bundles/uf.dl")` imports the file relative to the
        directory of the importing module, or of the program run from a
        file (see :meth:`importing_from`), or else to the working
        directory. `.import("package:bundles/uf.dl")` imports a module
        shipped in the Python package ``package``.

        A module is imported once: importing it again, in the scope it was
        imported in, does nothing. A module has no query.

        Raises
        ------
        DatalogModuleError
            raised if the module is not found, does not parse, has a
            query, is in an import cycle, or its statements fail.
        """
        if len(command.args) != 1 or command.kwargs:
            raise InvalidCommandExpression(
                'The .import command takes the path of a module, as in '
                '.import("module.dl").'
            )
        path = self._module_path(_unwrap_expr(command.args[0]))
        if path in self._importing:
            cycle = " -> ".join(
                str(p) for p in self._importing[self._importing.index(path):]
            )
            raise DatalogModuleError(f"Import cycle: {cycle} -> {path}")
        if str(path) in self.imported_modules():
            return
        program = _parse_module(path)
        before = dict(self.symbol_table._symbols)
        with self.importing_from(path):
            try:
                self.walk(program)
            except DatalogModuleError:
                raise
            except NeuroLangException as e:
                raise DatalogModuleError(
                    f"In the Datalog module {path}: {e}"
                ) from e
        imported = self.imported_modules()
        imported[str(path)] = tuple(
            symbol for symbol, value in self.symbol_table._symbols.items()
            if before.get(symbol) is not value and symbol != IMPORTED_MODULES
        )
        self.symbol_table[IMPORTED_MODULES] = Constant[Tuple](
            tuple(imported.items()), auto_infer_type=False, verify_type=False
        )

    _importing = ()

    @contextmanager
    def importing_from(self, path):
        """
        Run the block as an import of the module at *path*: the modules
        it imports are relative to its directory.
        """
        importing = self._importing
        self._importing = importing + (Path(path).resolve(),)
        try:
            yield
        finally:
            self._importing = importing

    def imported_modules(self):
        """
        Return the paths of the modules imported in the scope, mapped to
        the symbols each defined.
        """
        imported = self.symbol_table.get(IMPORTED_MODULES)
        return dict(imported.value) if imported is not None else {}

    def _module_path(self, name):
        """Return the path of the module *name* (see :meth:`import_module`)."""
        match = _PACKAGE_RESOURCE.fullmatch(name)
        if match:
            package, name = match.groups()
            try:
                path = Path(str(resources.files(package).joinpath(name)))
            except ModuleNotFoundError as e:
                raise DatalogModuleError(
                    f"No Datalog module {name!r}: no package {package!r}"
                ) from e
        elif self._importing:
            path = self._importing[-1].parent / name
        else:
            path = Path(name)
        if not path.is_file():
            raise DatalogModuleError(f"No Datalog module {str(path)!r}")
        return path.resolve()

    @add_match(Command)
    def _unknown_command(self, command):
        raise UnsupportedProgramError(
            f"The command statement {command} is not supported."
        )


def _unwrap_expr(expr):
    if isinstance(expr, Constant):
        return expr.value
    return expr.name


def _parse_module(path):
    """
    Return the statements of the Datalog module at *path*, refusing a
    module with a query.
    """
    from .frontend.datalog.standard_syntax import parser

    try:
        program = parser(path.read_text(encoding="utf-8"))
    except DatalogParserError as e:
        message = re.sub(
            r",? at line \d+,? col(umn)? \d+\.?$", "",
            str(e).strip().splitlines()[0],
        )
        where = "" if e.line is None else f", line {e.line}, column {e.column}"
        raise DatalogModuleError(
            f"In the Datalog module {path}{where}: {message}"
        ) from e
    if any(isinstance(f, Query) for f in program.formulas):
        raise DatalogModuleError(
            f"The Datalog module {path} has a query: a module defines "
            "facts and rules, the program importing it queries them."
        )
    return program
