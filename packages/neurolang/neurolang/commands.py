import re
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import AbstractSet, Dict, NamedTuple, Optional, Tuple

import pandas as pd
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
from .atlases import FETCHERS, Atlas
from .regions import ExplicitVBR
from .confine import ConfinedError, check_read
from .utils.nilearn_compat import fetch_files as _fetch_files

#: The kinds of ``.load_atlas`` that read a label image and a table.
ATLAS_FILE_KINDS = {
    "freesurfer": Atlas.from_freesurfer,
    "bids": Atlas.from_bids_dseg,
    "tsv": Atlas.from_tsv,
}

# The documentation of the predicates of a scope (see
# CommandsMixin.document).
PREDICATE_DOCUMENTATION = Symbol("__predicate_documentation__")


class PredicateDocumentation(NamedTuple):
    """
    The documentation of a predicate: its *description* and its
    *arguments*, ``(name, sort)`` pairs, the sort ``""`` when not given,
    or none.
    """

    description: str
    arguments: Tuple[Tuple[str, str], ...] = ()


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
            check_read(url, "table")
        except ConfinedError as e:
            raise InvalidCommandExpression(str(e)) from e
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
        Process the `.load_atlas` command. The load_atlas command loads an
        atlas, a label image and the names of its labels, into the relation
        `name(label, region)` of its regions, and records it in the
        `atlases` of the engine, by name (:class:`~neurolang.atlases.Atlas`),
        where the engine `tracts`' `.index_tracts` finds it.

        Usage
        -----
        The first positional argument is the name of the relation and of
        the atlas. The option `kind` says where the rest comes from:

        - no `kind` (or `kind="csv"`), the label image and the labels at
          a URL: `.load_atlas(name, atlas_url, labels_url[, base_url])`.
          If `base_url` is given, the atlas and labels urls are relative
          to it (an archive, uncompressed). The files are downloaded to
          `~/neurolang_data`; the first two columns of the labels are the
          value and the name of each label. Other keyword arguments are
          passed to pandas' `read_csv` method. Names are kept as they are.
        - `kind="freesurfer"` (a colour LUT), `"bids"` (a `dseg.tsv`) or
          `"tsv"` (columns `id`, `name`, optional `hemi`): local files,
          `.load_atlas(name, image, table, kind=..., space=...,
          background=0)`. Names are kept as they are.
        - `kind="destrieux"`, `"schaefer"` or `"tian"`: the atlases of the
          engines, fetched, `.load_atlas(name, kind=..., space="MNI152",
          data_dir=...)`; other options go to the fetcher (`n_rois=100`
          of `schaefer`, `scale=2` of `tian`).
        - `kind="nilearn"`: `nilearn.datasets.fetch_atlas_<fetcher>`,
          `.load_atlas(name, kind="nilearn", fetcher="destrieux_2009",
          space="MNI152", background=0)`; other options go to the fetcher.

        The names of the fetched atlases are normalised, `-` and `_` being
        spaces (:func:`~neurolang.atlases.label_name`), as the engines name
        them. The space is `"unspecified"` for files, `"MNI152"` for the
        fetched atlases, unless given.

        `.load_atlas(destrieux, "destrieux2009_rois_lateralized.nii.gz",
        "destrieux2009_rois_labels_lateralized.csv",
        "https://www.nitrc.org/frs/download.php/11942/destrieux2009.tgz")`

        will load Destrieux's atlas into the destrieux relation, as
        `.load_atlas(destrieux, kind="destrieux")` does from nilearn.

        Raises
        ------
        InvalidCommandExpression
            raised if command args are invalid or if an error occurs while
            loading the data
        """
        args = [_unwrap_expr(a) for a in command.args]
        options = {
            _unwrap_expr(k): _unwrap_expr(v) for k, v in command.kwargs
        }
        kind = options.pop("kind", "csv")
        # the files and addresses of the statement, the directory it
        # downloads to (see neurolang.confine)
        for value in [*args[1:], options.get("data_dir"),
                      options.get("data_home")]:
            if isinstance(value, (str, Path)):
                try:
                    check_read(value, "atlas")
                except ConfinedError as e:
                    raise InvalidCommandExpression(str(e)) from e
        if not args:
            raise InvalidCommandExpression(
                ".load_atlas needs the name of the atlas"
            )
        name, args = args[0], args[1:]
        try:
            atlas = self._load_atlas(name, kind, args, options)
        except (OSError, ValueError, NotImplementedError, ImportError) as e:
            raise InvalidCommandExpression(
                f"cannot load the atlas {name!r}: {e}"
            ) from e
        self.add_atlas(atlas)

    def _load_atlas(self, name, kind, args, options):
        def expect(n, what, allowed):
            if len(args) != n:
                raise InvalidCommandExpression(
                    f'.load_atlas of kind="{kind}" takes the name of the '
                    f"atlas{what}, not {len(args) + 1} arguments"
                )
            unknown = set(options) - set(allowed or options)
            if unknown:
                raise InvalidCommandExpression(
                    f'.load_atlas of kind="{kind}" has no option '
                    f"{sorted(unknown)}; its options are {sorted(allowed)}"
                )

        if kind == "csv":
            if len(args) not in (2, 3):
                expect(2, ", the atlas url and the labels url", None)
            image, labels = self._fetch_atlas_map_and_labels(
                *args[:2], args[2] if len(args) == 3 else None
            )
            return Atlas.from_csv(name, image, labels, read_csv=options)
        if kind in ATLAS_FILE_KINDS:
            expect(2, ", the image and the table", ("space", "background"))
            return ATLAS_FILE_KINDS[kind](
                name, *args, options.get("space", "unspecified"),
                background=int(options.get("background", 0)),
            )
        if kind in FETCHERS:
            expect(0, "", None)
            data_dir = options.pop("data_dir", Path.home() / "neurolang_data")
            return Atlas.fetch(name, kind, data_dir, **options)
        if kind == "nilearn":
            expect(0, "", None)
            if not options.get("fetcher"):
                raise InvalidCommandExpression(
                    'an atlas of kind="nilearn" needs fetcher=..., the name '
                    "of a nilearn fetcher, such as destrieux_2009"
                )
            if "background" in options:
                options["background"] = int(options["background"])
            return Atlas.from_nilearn(name, options.pop("fetcher"), **options)
        raise InvalidCommandExpression(
            f"the kind of an atlas is csv, {', '.join(ATLAS_FILE_KINDS)}, "
            f"{', '.join(FETCHERS)} or nilearn, not {kind!r}"
        )

    def _fetch_atlas_map_and_labels(self, atlas_url, labels_url, base_url):
        """Download the atlas and its labels to ``~/neurolang_data``."""
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

        return _fetch_files(data_dir, files)

    @property
    def atlases(self) -> Dict[str, Atlas]:
        """The atlases loaded (``.load_atlas``, ``add_atlas_set``, the
        ``atlases:`` of an engine), by name."""
        return self.__dict__.setdefault("_atlases", {})

    def add_atlas(self, atlas: Atlas, name: Optional[str] = None) -> None:
        """Add the relation ``name(label, region)`` of the regions of
        ``atlas`` (its name by default) and record it in :attr:`atlases`.

        A region is an :class:`~neurolang.regions.ExplicitVBR` of the
        voxels of a label of the table, but the background.
        """
        voxels, shape = atlas.label_voxels(), atlas.shape
        rows = [
            (name_, ExplicitVBR(voxels[i], atlas.affine, image_dim=shape))
            for i, name_ in atlas.region_names().items()
        ]
        type_ = Tuple[str, ExplicitVBR]
        self.add_extensional_predicate_from_tuples(
            Symbol[AbstractSet[type_]](name or atlas.name), rows, type_=type_
        )
        self.atlases[atlas.name] = atlas

    @add_match(Command(Symbol("document"), ..., ...))
    def document(self, command):
        """
        Process the `.document` command, which documents a predicate, as
        a ``%%`` block before its rule or fact does.

        Usage
        -----
        `.document(name, "description")` documents the predicate *name*;
        a third argument, a tuple of ``(name, sort)`` pairs, names its
        arguments. A predicate documented several times has its
        descriptions one after the other, each once, and the arguments
        given last. The documentation follows the scopes of the program,
        as its rules do (see :meth:`predicate_documentation`).
        """
        if not 2 <= len(command.args) <= 3 or command.kwargs:
            raise InvalidCommandExpression(
                'The .document command takes the name of a predicate and '
                'its description, as in .document(p, "description").'
            )
        name = _unwrap_expr(command.args[0])
        description = str(_unwrap_expr(command.args[1])).strip()
        arguments = (
            tuple(_unwrap_expr(command.args[2]))
            if len(command.args) == 3 else ()
        )
        documentation = self._documentation()
        old = documentation.get(name, PredicateDocumentation(""))
        if description and description not in old.description:
            description = "\n\n".join(
                d for d in (old.description, description) if d
            )
        else:
            description = old.description
        documentation[name] = PredicateDocumentation(
            description, tuple(map(tuple, arguments)) or old.arguments
        )
        self.symbol_table[PREDICATE_DOCUMENTATION] = Constant[Tuple](
            tuple(documentation.items()),
            auto_infer_type=False, verify_type=False,
        )

    def _documentation(self):
        """Return the documentation of the scope, of every predicate."""
        documentation = self.symbol_table.get(PREDICATE_DOCUMENTATION)
        return dict(documentation.value) if documentation is not None else {}

    def predicate_documentation(self):
        """
        Return the :class:`PredicateDocumentation` of the predicates of
        the scope, by name.

        Only the predicates the program defines are documented: those of
        a scope that was closed, or that were removed, are not.
        """
        return {
            name: documentation
            for name, documentation in self._documentation().items()
            if Symbol(name) in self.symbol_table
        }

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
        directory; not found there, in the directories an engine gives its
        modules (:attr:`module_directories`), in order.
        `.import("package:bundles/uf.dl")` imports a module shipped in the
        Python package ``package``.

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

    #: The directories where ``.import`` looks for a module not found
    #: relative to the importing one, such as those an engine ships.
    module_directories = ()

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
        else:
            base = self._importing[-1].parent if self._importing else Path()
            candidates = [base / name] + [
                Path(d) / name for d in self.module_directories
            ]
            path = next((p for p in candidates if p.is_file()), candidates[0])
            # a module of the engine's, or else one the statements may read
            shipped = any(path.resolve().is_relative_to(Path(d).resolve())
                          for d in self.module_directories)
            if not shipped:
                try:
                    check_read(path, "Datalog module")
                except ConfinedError as e:
                    raise DatalogModuleError(str(e)) from e
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
