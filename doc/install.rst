Installing Neurolang
====================

.. |dependencies| replace:: 
   Neurolang requires a Python installation and the following dependencies: 
   ipython, scipy, scikit-learn, joblib, matplotlib, nibabel, nilearn.


 

Windows
-------

First: download and install 64 bit Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend that you **install a complete 64 bit scientific Python
distribution
like** `Anaconda <%0A%20%20%20%20%20%20%20https://www.anaconda.com/download/>`__
. Since it meets all the requirements of neurolang, it will save you
time and trouble. You could also check
`PythonXY <http://python-xy.github.io/>`__ as an alternative.

|dependencies|

Second: open a Command Prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| **(Press "Win-R", type "cmd" and press "Enter". This will open the
   program cmd.exe, which is the command prompt)**
| Then type the following line and press "Enter"

.. container:: code

   ::

      pip install -U --user neurolang

Third: open IPython
~~~~~~~~~~~~~~~~~~~

| **(You can open it by writing "ipython" in the command prompt and
   pressing "Enter")**
| Then type in the following line and press "Enter":

.. container:: code

   ::

      In [1]: import neurolang

If no error occurs, you have installed neurolang correctly.


Mac
---


First: download and install 64 bit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Anaconda <https://www.anaconda.com/download/>`__

We recommend that you **install a complete 64 bit scientific Python
distribution
like** `Anaconda <https://www.anaconda.com/download/>`__. Since it
meets all the requirements of neurolang, it will save you time and
trouble.


Second: open a Terminal
~~~~~~~~~~~~~~~~~~~~~~~

| **(Navigate to /Applications/Utilities and double-click on
   Terminal)**
| Then type the following line and press "Enter"

.. container:: code

   ::

      pip install -U --user neurolang

Third: open IPython
~~~~~~~~~~~~~~~~~~~

| **(You can open it by writing "ipython" in the terminal and
   pressing "Enter")**
| Then type in the following line and press "Enter":

.. container:: code

   ::

      In [1]: import neurolang

If no error occurs, you have installed neurolang correctly.

Linux
-----

If you are using **Ubuntu or Debian**

First: Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install or ask your system administrator to install the following
packages using the distribution package manager: **ipython** ,
**scipy**, **scikit-learn** (sometimes called sklearn, or
python-sklearn), **joblib**, **matplotlib** (sometimes called
python-matplotlib) and **nibabel** (sometimes called
python-nibabel).

**If you do not have access to the package manager we recommend
that you install a complete 64 bit scientific Python distribution
like** `Anaconda <https://www.anaconda.com/download/>`__. Since
it meets all the requirements of neurolang, it will save you time
and trouble.

Second: open a Terminal
~~~~~~~~~~~~~~~~~~~~~~~

| **(Press ctrl+alt+t and a Terminal console will pop up)**
| Then type the following line and press "Enter"

.. container:: code

   ::

      pip install -U --user neurolang

Third: open IPython
~~~~~~~~~~~~~~~~~~~

| **(You can open it by writing "ipython" in the terminal and
   pressing "Enter")**
| Then type in the following line and press "Enter":

.. container:: code

   ::

      In [1]: import neurolang

If no error occurs, you have installed neurolang correctly.

To Install the development version
----------------------------------

**Use git as an alternative to using pip, to get the latest
neurolang version**

Simply run the following command (as a shell command, not a Python
command):

.. container:: code

   ::

      git clone https://github.com/neurolang/neurolang.git

In the future, you can readily update your copy of neurolang by
executing “git pull” in the neurolang root directory (as a shell
command).

If you really do not want to use git, you may still download the
latest development snapshot from the following link (unziping
required):
https://github.com/neurolang/neurolang/archive/master.zip

**Install in the neurolang directory created by the previous
steps, run (again, as a shell command):**

.. container:: code

   ::

      python setup.py develop --user

**Now to test everything is set up correctly, open IPython and
type in the following line:**

.. container:: code

   ::

      In [1]: import neurolang

If no error occurs, you have installed neurolang correctly.

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

      pip install -e .[dask]

Finally, to enable the dask-sql backend for Neurolang, you need to edit
Neurolang's configuration file located in `neurolang/utils/config/config.ini`.