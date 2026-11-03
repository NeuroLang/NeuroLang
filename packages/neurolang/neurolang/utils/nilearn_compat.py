"""
Lazy access to nilearn's dataset and image modules.

``nilearn.datasets`` and ``nilearn.image`` take over a second to import
because they pull in scikit-learn. Importing them only when data is
downloaded or resampled keeps ``import neurolang`` and the
``neurolang-query`` listing options fast.
"""


from pathlib import Path


def nilearn_datasets():
    """
    Return the ``nilearn.datasets`` module, importing it on first use.

    Returns
    -------
    module
        ``nilearn.datasets``.

    """
    from nilearn import datasets

    return datasets


def nilearn_image():
    """
    Return the ``nilearn.image`` module, importing it on first use.

    Returns
    -------
    module
        ``nilearn.image``.

    """
    from nilearn import image

    return image


def fetch_files(*args, **kwargs):
    """
    Download files with nilearn's (private) file fetcher.

    Parameters
    ----------
    *args, **kwargs
        passed to nilearn's ``fetch_files``: the data directory, a list
        of ``(file name, url, options)`` tuples and options such as
        ``verbose``.

    Returns
    -------
    list of str
        paths of the downloaded (or already cached) files.

    Notes
    -----
    When every file is already in the data directory, and none is to be
    downloaded again, their paths are returned without importing nilearn,
    as it would, which takes seconds.

    """
    files = list(args[1] if len(args) > 1 else kwargs.get("files", ()))
    data_dir = args[0] if args else kwargs.get("data_dir")
    options = set(kwargs) - {"data_dir", "files", "verbose", "resume"}
    if data_dir is not None and not options and len(args) <= 2 and all(
        not opts.get("overwrite") and (Path(data_dir) / name).exists()
        for name, _, opts in files
    ):
        return [str(Path(data_dir) / name) for name, _, _ in files]
    try:
        from nilearn.datasets.utils import _fetch_files as fetcher
    except ImportError:
        # nilearn >= 0.11: _fetch_files was moved and renamed
        from nilearn.datasets._utils import fetch_files as fetcher
    return fetcher(*args, **kwargs)
