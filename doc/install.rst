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
