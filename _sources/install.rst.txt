Installing NeuroLang
====================

NeuroLang requires **Python ≥ 3.8**. The recommended way to install it is
with `uv <https://github.com/astral-sh/uv>`_ — a fast Python package manager —
or with standard ``pip``.

.. tab-set::

   .. tab-item:: Windows

      **1. Install uv**

      Open PowerShell and run:

      .. code-block:: powershell

         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      Or use pip directly if you already have Python installed:

      .. code-block:: text

         pip install uv

      **2. Install NeuroLang**

      .. code-block:: text

         uv pip install neurolang

      **3. Verify**

      Open a new terminal and run:

      .. code-block:: text

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: macOS

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install neurolang

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: Linux

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install neurolang

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: From source

      Clone the repository and install in editable mode with all dev and doc
      dependencies:

      .. code-block:: bash

         git clone https://github.com/NeuroLang/NeuroLang.git
         cd NeuroLang
         uv pip install -e ".[dev,doc]"

      To update your local copy:

      .. code-block:: bash

         git pull
         uv pip install -e ".[dev,doc]"

      Verify:

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"


Using pip without uv
---------------------

If you prefer plain pip:

.. code-block:: bash

   pip install neurolang


Dependencies
------------

NeuroLang requires the following libraries (installed automatically):

* `NumPy <https://numpy.org>`_
* `SciPy <https://scipy.org>`_
* `pandas <https://pandas.pydata.org>`_
* `nibabel <https://nipy.org/nibabel/>`_
* `nilearn <https://nilearn.github.io>`_
* `scikit-learn <https://scikit-learn.org>`_
* `matplotlib <https://matplotlib.org>`_
* `rdflib <https://rdflib.readthedocs.io>`_

Installing alternative backends for Neurolang
---------------------------------------------

By default, Neurolang uses the pandas library to manage its data.
It is however possible to use an alternative backend which relies
on `dask-sql <https://github.com/NeuroLang/dask-sql>`__.

Using the dask-sql backend allows Neurolang to benefit from some
query optimizations based on SQL syntax, as well as parallelism in
computation by using dask instead of pandas. In other words, using
the dask-sql backend might help improve the performance of specific
queries which are slow to resolve with the default installation of
Neurolang. It is not however guaranteed to be faster, as query
optimizations are very case specific...

To install Neurolang with the dask-sql backend, **it is required to
first install a version of the maven library along with a running
java installation with version >= 8** (dask-sql needs Java for
parsing the SQL queries). There are two ways to install maven along
with java:

On a linux machine, run the following command in your shell terminal:

.. container:: code

   ::

      sudo apt update && sudo apt install maven

Or, if you're using conda to manage your python environments, you can
run the following command in your active python environment:

.. container:: code

   ::

      conda install -c conda-forge maven

Once you've setup your machine with a working version of java and maven,
you can install Neurolang with the dask-sql backend by running the 
following command in your active python environment from a local copy of
the Neurolang repository:

.. container:: code

   ::

      pip install -U --user neurolang[dask]

Finally, to enable the dask-sql backend for Neurolang, you can either
edit Neurolang's configuration file located in your python environment
path under `config/config.ini`, or you can call

.. container:: code

   ::

      from neurolang.config import config
      config.set_query_backend("dask")

from your python shell.

**Note**: you need to call the `set_query_backend` method at the top
of your script, before you import other modules from Neurolang.
