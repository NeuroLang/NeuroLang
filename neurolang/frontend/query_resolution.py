r"""
Query Builder Base, Region and Neurosynth Mixins
================================================
Base classes to construct Datalog programs.
Capabilities to declare, manage and manipulate symbols.
Mixins provide capabilities related to brain volumes and
Neurosynth metadata manipulation
"""
from contextlib import contextmanager
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid1

import numpy as np
from nibabel.dataobj_images import DataobjImage

from .. import expressions as ir
from ..region_solver import Region
from ..regions import ExplicitVBR, ImplicitVBR, SphericalVolume
from ..type_system import Unknown, is_leq_informative
from ..typed_symbol_table import TypedSymbolTable
from .neurosynth_utils import NeuroSynthHandler, StudyID, TfIDf
from . import query_resolution_expressions as fe
from ..datalog import DatalogProgram
from ..logic import ExistentialPredicate, UniversalPredicate


class QueryBuilderBase:
    """
    Base class to construct Datalog programs.
    Capabilities to declare, manage and manipulate symbols.
    """

    def __init__(
        self, program_ir: DatalogProgram, logic_programming: bool = False
    ) -> "QueryBuilderBase":
        """
        Query Builder with symbol management capabilities

        Parameters
        ----------
        program_ir : DatalogProgram
            Datalog program's intermediate representation,
            usually blank
        logic_programming : bool, optional
            defines if symbols can be dynamically
            declared, by default False

        Returns
        -------
        QueryBuilderBase
            see description
        """
        self.program_ir = program_ir
        self.set_type = AbstractSet[self.program_ir.type]
        self.logic_programming = logic_programming

        for k, v in self.program_ir.included_functions.items():
            self.symbol_table[ir.Symbol[v.type](k)] = v

        self._symbols_proxy = QuerySymbolsProxy(self)

    def get_symbol(self, symbol_name: Union[str, fe.Expression]) -> fe.Symbol:
        """
        Retrieves symbol via its name, either providing a
        fe.Expression with the correct name or the name itself

        Parameters
        ----------
        symbol_name : Union[str, fe.Expression]
            name of the symbol to be retrieved. If of type fe.Expression,
            expression's name is used as the name

        Returns
        -------
        fe.Symbol
            symbol corresponding to the input name

        Raises
        ------
        ValueError
            if no symbol could be found with given name

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_symbol(3, "x")
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        """
        if isinstance(symbol_name, fe.Expression):
            symbol_name = symbol_name.expression.name
        if symbol_name not in self.symbol_table:
            raise ValueError(f"fe.Symbol {symbol_name} not defined")
        return fe.Symbol(self, symbol_name)

    def __getitem__(
        self, symbol_name: Union[fe.Symbol, str, fe.Expression]
    ) -> fe.Symbol:
        """
        Retrieves symbol via its name, either providing a
        fe.Expression with the correct name or the name itself
        Points towards .get_symbol method

        Parameters
        ----------
        symbol_name : Union[fe.Symbol, str, fe.Expression]
            if of type fe.Symbol, symol's name will be passed
            to the .get_symbol method

        Returns
        -------
        fe.Symbol
            output from .get_symbol method
        """
        if isinstance(symbol_name, fe.Symbol):
            symbol_name = symbol_name.symbol_name
        return self.get_symbol(symbol_name)

    def __contains__(self, symbol: fe.Symbol) -> bool:
        """
        Checks if symbol exists in current symbol_table

        Parameters
        ----------
        symbol : fe.Symbol
            symbol to check

        Returns
        -------
        bool
            does symbol exist in current symbol_table
        """
        return symbol in self.symbol_table

    def exists(
        self, quantified_variable: fe.Symbol, body: fe.Expression
    ) -> fe.Exists:
        """
        Existential predicate on the body. The predicate
        will be cosidered satisfied if any instance of
        `quantified_variable` satifies `body`.

        Parameters
        ----------
        quantified_variable : fe.Symbol
            existentially-quantified variable

        body: fe.Expression
            first order logic proposition which to be
            satisfied by the existential quatifier.

        Returns
        -------
        fe.Exists
            representation of the existential predicate.
        """
        existential_predicate = (
            fe.Exists(
                self,
                ExistentialPredicate(
                    quantified_variable.expression,
                    body.expression
                ),
                quantified_variable,
                body
            )
        )

        return existential_predicate

    def all(
        self, quantified_variable: fe.Symbol, body: fe.Expression
    ) -> fe.All:
        """
        Universal predicate on the body. The predicate
        will be cosidered satisfied if all instances of
        `quantified_variable` satify `body`.

        Parameters
        ----------
        quantified_variable : fe.Symbol
            universally-quantified variable

        body: fe.Expression
            first order logic proposition which to be
            satisfied by the universal quatifier.

        Returns
        -------
        fe.All
            representation of the universal predicate.
        """
        universal_predicate = (
            fe.All(
                self,
                UniversalPredicate(
                    quantified_variable.expression,
                    body.expression
                ),
                quantified_variable,
                body
            )
        )

        return universal_predicate

    @property
    def types(self) -> List[Type]:
        """
        Returns a list of the types of the symbols currently
        in the table (type can be Unknown)

        Returns
        -------
        List[Union[fe.Expression, Type]]
            List of types or Unknown if symbol has no known type
        """
        return self.symbol_table.types

    @property
    def symbol_table(self) -> TypedSymbolTable:
        """Projector to the program_ir's symbol_table"""
        return self.program_ir.symbol_table

    @property
    def symbols(self) -> Iterator[str]:
        """Iterator through the symbol's names"""
        return self._symbols_proxy

    @property
    @contextmanager
    def environment(self) -> "QuerySymbolsProxy":
        """
        Dynamic context that can be used to create
        symbols to write a Datalog program.
        Contrary to a scope, symbols stay in the symbol_table
        when exiting the environment context

        Yields
        -------
        QuerySymbolsProxy
            in dynamic mode, can be used to create symbols on-the-fly

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> with nl.environment as e:
        ...     e.x = 3
        >>> "x" in nl
        True
        >>> nl.symbols.x == 3
        True
        """
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        try:
            yield self._symbols_proxy
        finally:
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    @property
    @contextmanager
    def scope(self) -> "QuerySymbolsProxy":
        """
        Dynamic context that can be used to create
        symbols to write a Datalog program.
        Contrary to an environment, symbols disappear from the symbol_table
        when exiting the scope context

        Yields
        -------
        QuerySymbolsProxy
            in dynamic mode, can be used to create symbols on-the-fly

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> with nl.scope as e:
        ...     e.x = 3
        >>> "x" in nl
        False
        """
        old_dynamic_mode = self._symbols_proxy._dynamic_mode
        self._symbols_proxy._dynamic_mode = True
        self.program_ir.push_scope()
        try:
            yield self._symbols_proxy
        finally:
            self.program_ir.pop_scope()
            self._symbols_proxy._dynamic_mode = old_dynamic_mode

    def new_symbol(
        self,
        type_: Union[Any, Tuple[Any, ...], List[Any]] = Unknown,
        name: str = None,
    ) -> fe.Expression:
        """
        Creates a symbol and associated expression, optionally
        specifying it's type and/or name

        Parameters
        ----------
        type_ : Union[Any, Tuple[Any, ...], List[Any]], optional
            type of the created symbol, by default Unknown
            if Iterable, will be cast to a Tuple
        name : str, optional
            name of the created symbol, by default None

        Returns
        -------
        fe.Expression
            associated to the created symbol
        """
        if isinstance(type_, (tuple, list)):
            type_ = tuple(type_)
            type_ = Tuple[type_]

        if name is None:
            name = str(uuid1())
        return fe.Expression(self, ir.Symbol[type_](name))

    @property
    def functions(self) -> List[str]:
        """
        Returns the list of symbols corresponding to callables

        Returns
        -------
        List[str]
            list of symbols of type leq Callable

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> def f(x: int) -> int:
        ...     return x+2
        >>> nl.add_symbol(f, "f")
        f: typing.Callable[[int], int] = <function f at ...>
        >>> "f" in nl.functions
        True
        >>> nl.add_symbol(3, "x")
        x: <class 'int'> = 3
        >>> "x" in nl.functions
        False
        """
        return [
            s.name
            for s in self.symbol_table
            if is_leq_informative(s.type, Callable)
        ]

    def add_symbol(
        self, value: Union[fe.Expression, ir.Constant, Any], name: str = None
    ) -> fe.Symbol:
        """
        Creates a symbol with given value and adds it to the
        current symbol_table.
        Can typicaly be used to decorate callables, or add an
        ir.Constant to the program.

        Parameters
        ----------
        value : Union[fe.Expression, ir.Constant, Any]
            value of the symbol to add. If not an fe.Expression,
            will be cast as an ir.Constant
        name : str, optional
            overrides automatic naming of the symbol, by default None

        Returns
        -------
        fe.Symbol
            created symbol

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> @nl.add_symbol
        ... def g(x: int) -> int:
        ...     return x + 2
        >>> nl.get_symbol("g")
        g: typing.Callable[[int], int] = <function g at ...>
        >>> nl.add_symbol(3, "x")
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        """
        name = self._obtain_symbol_name(name, value)

        if isinstance(value, fe.Expression):
            value = value.expression
        elif isinstance(value, ir.Constant):
            pass
        else:
            value = ir.Constant(value)

        symbol = ir.Symbol[value.type](name)
        self.symbol_table[symbol] = value

        return fe.Symbol(self, name)

    def _obtain_symbol_name(self, name: Optional[str], value: Any) -> str:
        if name is not None:
            return name

        if not hasattr(value, "__qualname__"):
            return str(uuid1())

        if "." in value.__qualname__:
            ix = value.__qualname__.rindex(".")
            name = value.__qualname__[ix + 1 :]
        else:
            name = value.__qualname__

        return name

    def del_symbol(self, name: str) -> None:
        """
        Deletes the symbol with parameter name
        from the symbol_table

        Parameters
        ----------
        name : str
            Name of the symbol to delete

        Raises
        ------
        ValueError
            if no symbol could be found with given name

        Example
        -------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_symbol(3, "x")
        x: <class 'int'> = 3
        >>> nl.get_symbol("x")
        x: <class 'int'> = 3
        >>> nl.del_symbol("x")
        >>> nl.get_symbol("x")
        ValueError: Symbol x not defined
        """
        del self.symbol_table[name]

    def add_tuple_set(
        self,
        iterable: Union[Iterable, Iterable[Tuple[Any, ...]]],
        type_: Type = Unknown,
        name: str = None,
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Any, ...]] format (see examples).
        Typically used to create extensional facts from existing databases

        Parameters
        ----------
        iterable : Union[Iterable, Iterable[Tuple[Any, ...]]]
            typically a list of tuples of values, other formats will
            be interpreted as the latter (see examples)
        type_ : Type, optional
            type of elements for the tuples, if not specified
            will be inferred from the first element, by default Unknown
        name : str, optional
            name for the AbstractSet symbol, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderBase(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (3, 4)], name="l1")
        l1: typing.AbstractSet[typing.Tuple[int, int]] = \
            [(1, 2), (3, 4)]
        >>> nl.add_tuple_set([[1, 2, 3], (3, 4)], name="l2")
        l2: typing.AbstractSet[typing.Tuple[int, int, float]] = \
            [(1, 2, 3.0), (3, 4, nan)]
        >>> nl.add_tuple_set((1, 2, 3), name="l3")
        l3: typing.AbstractSet[typing.Tuple[int]] = \
            [(1,), (2,), (3,)]
        """
        if not isinstance(type_, tuple) or len(type_) == 1:
            if isinstance(type_, tuple) and len(type_) == 1:
                type_ = type_[0]
                iterable = (e[0] for e in iterable)

            set_type = AbstractSet[type_]
        else:
            type_ = tuple(type_)
            set_type = AbstractSet[Tuple[type_]]

        constant = self._add_tuple_set_elements(iterable, set_type)
        if name is None:
            name = str(uuid1())

        symbol = ir.Symbol[set_type](name)
        self.symbol_table[symbol] = constant

        return fe.Symbol(self, name)

    def _add_tuple_set_elements(self, iterable, set_type):
        element_type = set_type.__args__[0]
        new_set = []
        for e in iterable:
            if not (isinstance(e, fe.Symbol)):
                s, c = self._create_symbol_and_get_constant(e, element_type)
                self.symbol_table[s] = c
            else:
                s = e.neurolang_symbol
            new_set.append(s)

        return ir.Constant[set_type](self.program_ir.new_set(new_set))

    @staticmethod
    def _create_symbol_and_get_constant(element, element_type):
        symbol = ir.Symbol[element_type].fresh()
        if isinstance(element, ir.Constant):
            constant = element.cast(element_type)
        elif is_leq_informative(element_type, Tuple):
            constant = ir.Constant[element_type](
                tuple(ir.Constant(ee) for ee in element)
            )
        else:
            constant = ir.Constant[element_type](element)
        return symbol, constant


class RegionMixin:
    """
    Mixin complementing a QueryBuilderBase
    with methods specific to the manipulation
    of brain volumes: regions, atlases, etc...
    """

    @property
    def region_names(self) -> List[str]:
        """
        Returns the list of symbol names with Region type

        Returns
        -------
        List[str]
            list of symbol names from symbol_table
        """
        return [s.name for s in self.symbol_table.symbols_by_type(Region)]

    @property
    def region_set_names(self) -> List[str]:
        """
        Returns the list of symbol names with set_type

        Returns
        -------
        List[str]
            list of symbol names from symbol_table
        """
        return [
            s.name for s in self.symbol_table.symbols_by_type(self.set_type)
        ]

    def new_region_symbol(self, name: Optional[str] = None) -> fe.Symbol:
        """
        Returns symbol with type Region

        Parameters
        ----------
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        return self.new_symbol(type_=Region, name=name)

    def add_region(
        self, region: fe.Expression, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Adds region fe.Symbol to symbol_table

        Parameters
        ----------
        region : fe.Expression
            should be of the program_ir's type
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            symbol added to the symbol_table

        Raises
        ------
        ValueError
            if region is not of program_ir's type
        """
        if not isinstance(region, self.program_ir.type):
            raise ValueError(
                f"type mismatch between region and program_ir type:"
                f" {self.program_ir.type}"
            )

        return self.add_symbol(region, name)

    def add_region_set(
        self, region_set: Iterable, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Region]] format

        Parameters
        ----------
        region_set : Iterable
            Typically List[Tuple[Region]]
        name : Optional[str], optional
            symbol's name, if None will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        return self.add_tuple_set(region_set, name=name, type_=Region)

    @staticmethod
    def create_region(
        spatial_image: DataobjImage,
        label: int = 1,
        prebuild_tree: bool = False,
    ) -> ExplicitVBR:
        """
        Creates an ExplicitVBR out of the voxels of a dense spatial_image
        with specified label

        Parameters
        ----------
        spatial_image : DataobjImage
            contains the .dataobj array of interest
        label : int, optional
            selects which voxel to put in the output, by default 1
        prebuild_tree : bool, optional
            see ExplicitVBR, by default False

        Returns
        -------
        ExplicitVBR
            see description
        """
        voxels = np.transpose(
            (np.asanyarray(spatial_image.dataobj) == label).nonzero()
        )
        if len(voxels) == 0:
            return None
        region = ExplicitVBR(
            voxels,
            spatial_image.affine,
            image_dim=spatial_image.shape,
            prebuild_tree=prebuild_tree,
        )
        return region

    def add_atlas_set(
        self,
        name: str,
        atlas_labels: Dict[int, str],
        spatial_image: DataobjImage,
    ) -> fe.Symbol:
        """
        Creates an atlas set:
        1- for each region specified by a label and name in atlas_labels,
        creates associated ExplicitVBR and symbols
        Tuple[region_name: str, Region]
        2- groups regions in an AbstractSet[Tuple[str, Region]] symbol with
        specified name

        Parameters
        ----------
        name : str
            name for the output atlas symbol
        atlas_labels : Dict[int, str]
            specifies which voxels to select in the spatial_image,
            and the associated region name
        spatial_image : DataobjImage
            contains the .dataobj array ofinterest

        Returns
        -------
        fe.Symbol
            see description
        """
        atlas_set = list()
        for label_number, label_name in atlas_labels.items():
            region = self.create_region(spatial_image, label=label_number)
            if region is None:
                continue
            atlas_set.append((label_name, region))
        return self.add_tuple_set(
                atlas_set,
                type_=Tuple[str, ExplicitVBR], name=name
        )

    def sphere(
        self,
        center: Union[np.ndarray, Iterable[int]],
        radius: int,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Creates a Region symbol associated with the spherical
        volume described by its center and volume

        Parameters
        ----------
        center : Union[np.ndarray, Iterable[int]]
            3D center of the sphere
        radius : int
            radius of the sphere
        name : Optional[str], optional
            name of the output symbol, if None will be fresh,
            by default None

        Returns
        -------
        fe.Symbol
            see description
        """
        sr = SphericalVolume(center, radius)
        symbol = self.add_region(sr, name)
        return symbol

    def make_implicit_regions_explicit(self, affine, dim):
        """Raises NotImplementedError for now"""
        for region_symbol_name in self.region_names:
            region_symbol = self.get_symbol(region_symbol_name)
            region = region_symbol.value
            if isinstance(region, ImplicitVBR):
                self.add_region(
                    region.to_explicit_vbr(affine, dim), region_symbol_name
                )


class NeuroSynthMixin:
    """
    Neurosynth is a platform for large-scale, automated synthesis
    of functional magnetic resonance imaging (fMRI) data.
    see https://neurosynth.org/
    This Mixin complements a QueryBuilderBase with methods
    related to coordinate-based meta-analysis (CBMA) data loading.
    """

    def load_neurosynth_term_study_ids(
        self,
        term: str,
        name: Optional[str] = None,
        frequency_threshold: Optional[float] = 0.05,
    ) -> fe.Symbol:
        """
        Load study ids (PMIDs) of studies in the Neurosynth database that are
        related to a given term based on a hard thresholding of TF-IDF
        features.

        Parameters
        ----------
        term : str
            Term that studies must match.

        name : str (optional)
            Name of the symbol associated with the resulting set of studies.
            Randomly generated by default

        frequency_threshold : optional (default 0.05)
            Minimum value of the TF-IDF feature for studies to be considered
            related to the given term.

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        study_set = self.neurosynth_db.ns_study_id_set_from_term(
            term, frequency_threshold
        )
        return self.add_tuple_set(
            study_set.values, type_=Tuple[StudyID], name=name
        )

    def load_neurosynth_study_tfidf_feature_for_terms(
        self, terms: Iterable[str], name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load TF-IDF features measured in each study within the Neurosynth
        database for the given terms.

        Parameters
        ----------
        terms : iterable of str
            Terms for which the TF-IDF features should be obtained.

        name : str (optional)
            Name of the symbol associated with the resulting set of (study id,
            tfidf) tuples. Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_study_tfidf_feature_for_terms(terms)
        return self.add_tuple_set(
            result_set.values, type_=Tuple[StudyID, str, TfIDf], name=name
        )

    def load_neurosynth_study_ids(
        self, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load all study ids (PMIDs) that are part of the Neurosynth database.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of study ids.
            Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_study_ids()
        return self.add_tuple_set(result_set, type_=Tuple[StudyID], name=name)

    def load_neurosynth_reported_activations(
        self, name: Optional[str] = None
    ) -> fe.Symbol:
        """
        Load all activations reported in the Neurosynth database in the form of
        a set of (study id, voxel id) pairs.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of (study id,
            voxel id) tuples. Randomly generated by default

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_reported_activations()
        return self.add_tuple_set(
            result_set, type_=Tuple[StudyID, int], name=name
        )

    def load_neurosynth_term_study_associations(
        self,
        name: Optional[str] = None,
        threshold: Optional[float] = 1e-3,
        study_ids: Optional[List[int]] = None,
    ) -> fe.Symbol:
        """
        Load all term-to-study associations in the Neurosynth database based on
        a hard thresholding of the TF-IDF features measured from the abstracts
        of the studies.

        Parameters
        ----------
        name : str (optional)
            Name of the symbol associated with the resulting set of study ids.
            Randomly generated by default

        threshold : float (optional, default 1e-3)
            Threshold used to determine the association of a study to a given
            term.

        study_ids : iterable of int (optional)
            List of the IDs of the studies to consider. By default, all studies
            in the database are considered.

        Returns
        -------
        fe.Symbol
            fe.Symbol associated to the resulting set.

        """
        if not hasattr(self, "neurosynth_db"):
            self.neurosynth_db = NeuroSynthHandler()
        if not name:
            name = str(uuid1())
        result_set = self.neurosynth_db.ns_term_study_associations(
            threshold=threshold, study_ids=study_ids
        )
        return self.add_tuple_set(
            result_set, type_=Tuple[StudyID, str], name=name
        )


class QuerySymbolsProxy:
    """
    Class useful to create symbols on-the-fly
    Typically used in QueryBuilderBase contexts as the yielded value
    to write a program.
    Various methods are projectors to QueryBuilderBase methods
    """

    def __init__(self, query_builder):
        self._dynamic_mode = False
        self._query_builder = query_builder

    def __getattr__(self, name):
        """See QueryBuilderBase.get_symbol"""
        if name in self.__getattribute__("_query_builder"):
            return self._query_builder.get_symbol(name)

        try:
            return super().__getattribute__(name)
        except AttributeError:
            if self._dynamic_mode:
                return self._query_builder.new_symbol(Unknown, name=name)
            else:
                raise

    def __setattr__(self, name, value):
        """See QueryBuilderBase.add_symbol"""
        if name == "_dynamic_mode":
            return super().__setattr__(name, value)
        elif self._dynamic_mode:
            return self._query_builder.add_symbol(value, name=name)
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        """See QueryBuilderBase.del_symbol"""
        if self._dynamic_mode and name:
            self._query_builder.del_symbol(name)
        else:
            super().__delattr__(name)

    def __getitem__(self, attr):
        """See QueryBuilderBase.get_symbol"""
        return self._query_builder.get_symbol(attr)

    def __setitem__(self, key, value):
        """See QueryBuilderBase.add_symbol"""
        return self._query_builder.add_symbol(value, name=key)

    def __contains__(self, symbol):
        """See QueryBuilderBase.__contains__"""
        return symbol in self._query_builder.symbol_table

    def __iter__(self) -> List[str]:
        """
        Iterates through the names of the symbols
        currently in the symbol_table, ordered in ascending name

        Returns
        -------
        List[str]
            list of symbol names
        """
        return iter(
            sorted(set(s.name for s in self._query_builder.symbol_table))
        )

    def __len__(self) -> int:
        """
        Returns number of symbols currently in symbol_table

        Returns
        -------
        int
            see description
        """
        return len(self._query_builder.symbol_table)

    def __dir__(self):
        """Descibes self and lists symbols in current symbol_table"""
        init = object.__dir__(self)
        init += [symbol.name for symbol in self._query_builder.symbol_table]
        return init

    def __repr__(self):
        """Describes symbols currently in symbol_table"""
        init = [symbol.name for symbol in self._query_builder.symbol_table]
        return f"QuerySymbolsProxy with symbols {init}"
