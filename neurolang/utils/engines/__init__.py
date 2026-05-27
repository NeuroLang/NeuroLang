"""
Declarative engine definitions for the Neurolang-query CLI.

Engines are declared in ``engines.yaml``, which provides metadata
(description, predicates) and points to Python initialization scripts.
Each engine's init script is a Python module that exports::

    def init_engine(
        nl: NeurolangPDL,
        mask: nib.Nifti1Image,
        data_dir: Path,
    ) -> None:
        ...

The engine registry in :mod:`neurolang.utils.engine_registry` handles
loading the YAML, calling init scripts, and building fully-configured
:class:`~neurolang.frontend.NeurolangPDL` instances.
"""
