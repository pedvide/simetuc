simetuc
=======

Simulating Energy Transfer and Upconversion

--------------

|License| |Python version| |Pypi version| |Anaconda version| |Build
Status| |Coverage Status|

Installation
------------

Python 3.5 or 3.6 is required. Installing
`Anaconda <https://www.continuum.io/downloads>`_ is recommended; it
works with Windows (64/32 bits), Linux (64/32 bits) and Mac (64 bits).

After installing Anaconda execute the following commands at the command
prompt (cmd.exe for Windows, shell for Linux and Mac):

::

    conda config --add channels conda-forge
    conda config --add channels pedvide
    conda install simetuc

(The first two commands add packages repositories with up-to-date
versions of all needed packages.)

or

::

    pip install simetuc

That will download and install all necessary files.

Note: Some OSX users report problems using conda, if after installing
you can't use the program (i.e., ``simetuc -h`` fails because simetuc
wasn't recognized as a command), use ``pip install simetuc``

Update
~~~~~~

If you installed it using conda, update with:

::

    conda update simetuc

If you installed it with pip, update with:

::

    pip install -U simetuc

Features
--------

-  Command line interface program.

   -  Run with: ``simetuc config_file.txt [options]``
   -  See all options below and with: ``simetuc -h``

-  The simulations are controlled by a configuration text file that the
   user can edit with the parameters adequate to its system of study. It
   includes:

   -  Information about the host lattice.
   -  Energy states labels.
   -  Absorption and excitation (including ESA).
   -  Decay (including branching ratios).
   -  Energy transfer.
   -  Other settings for the power and concentration dependence or
      optimization.

-  simetuc works with any sensitizer and activator ion kind.

   -  The examples are given for the Yb-Tm system.

-  All kinds of energy transfer processes are supported:

   -  Energy migration.
   -  Upconversion (ETU).
   -  Downconversion.
   -  Cross-relaxation.
   -  Cooperative processes.
   -  Energy transfer from sensitizers to activators.
   -  Back transfer from activators to sensitizers.

-  See the example `configuration file <https://github.com/pedvide/simetuc/blob/master/simetuc/config_file.cfg>`_ in
   the simetuc folder.
-  Add decay experimental data as two column text data, separated by
   tabs or spaces.
-  Different options:

   -  Create the lattice.
   -  Simulate the dynamics (rise and decay).
   -  Optimize the energy transfer parameters.

      -  Minimize the deviation between experiment and simulation.

   -  Simulate the steady state.
   -  Simulate the power dependence of each emission.
   -  Simulate the concentration dependence of the dynamics or the
      steady state.

-  All results are plotted and saved in the .hdf5 format.
-  For all options ``--average`` uses standard average rate equations
   instead of microscopic ones.

Documentation
-------------

See the `manual <https://github.com/pedvide/simetuc/blob/master/docs/manual/simetuc_user_manual.pdf>`_.

TODO
----

-  [ ] Add pressure dependence option: Change the distances of the
   lattice and simulate dynamics or steady-state.
-  [ ] Read experimental data in more formats.
-  [x] Add cooperative sensitization (work in progress).
-  [ ] Include pulse frequency for steady state simulations using a non
   cw laser

Bugs/Requests
-------------

Please use the `GitHub issue
tracker <https://github.com/pedvide/simetuc/issues>`__ to submit bugs or
request features.

Publications
------------

This software has been described and used in these publications:

-  Villanueva-Delgado, P.; Krämer, K. W. & Valiente, R. `Simulating
   Energy Transfer and Upconversion in β-NaYF4: Yb3+,
   Tm3+ <http://pubs.acs.org/doi/10.1021/acs.jpcc.5b06770>`__
-  Villanueva-Delgado, P.; Krämer, K. W.; Valiente, R.; de Jong, M. &
   Meijerink, A. `Modeling Blue to UV Upconversion in β-NaYF4:
   Tm3+ <http://pubs.rsc.org/en/Content/ArticleLanding/2016/CP/C6CP04347J#!divAbstract>`_

If you use this software in a scientific publication, please cite the
appropriate articles above.

Acknowledgments
---------------

The financial support of the EU FP7 ITN LUMINET (Grant agreement No.
316906) is gratefully acknowledged.

This work was started at the University of Cantabria under Prof. Rafael
Valiente and continued at the University of Bern under PD Dr. Karl
Krämer.

License
-------

Copyright Pedro Villanueva Delgado, 2016-2017.

Distributed under the terms of the `MIT <https://github.com/pedvide/simetuc/blob/master/LICENSE.txt>`_ license,
simetuc is free and open source software.

.. |License| image:: https://img.shields.io/github/license/pedvide/simetuc.svg
   :target: https://github.com/pedvide/simetuc/blob/master/LICENSE.txt
.. |Python version| image:: https://img.shields.io/pypi/pyversions/simetuc.svg
   :target: https://pypi.python.org/pypi/simetuc
.. |Pypi version| image:: https://img.shields.io/pypi/v/simetuc.svg
   :target: https://pypi.python.org/pypi/simetuc
.. |Anaconda version| image:: https://anaconda.org/pedvide/simetuc/badges/version.svg
   :target: https://anaconda.org/pedvide/simetuc
.. |Build Status| image:: https://travis-ci.org/pedvide/simetuc.svg?branch=master
   :target: https://travis-ci.org/pedvide/simetuc
.. |Coverage Status| image:: https://coveralls.io/repos/github/pedvide/simetuc/badge.svg?branch=master
   :target: https://coveralls.io/github/pedvide/simetuc?branch=master
