simetuc
=======

Simulating Energy Transfer and Upconversion

--------------

|License| |Python version| |Pypi version| |Anaconda version| |Build
Status| |Coverage Status|

Installation
------------

Python 3.5 is required (it may work with earlier versions, though).
Installing `Anaconda <https://anaconda.org/>`__ is recommended
(preferably 64 bits).

::

    $ conda install -c pedvide simetuc

or

::

    $ pip install simetuc

Features
--------

-  Command line interface program.

   -  Run with: ``simetuc config_file.txt [options]``
   -  See all options below and with: ``simetuc -h``

-  The simulations are controlled by a configuration textfile that
   includes:

   -  Information about the host lattice.
   -  Energy states.
   -  Absorption and excitation.
   -  Decay (including branching ratios).
   -  Energy transfer.
   -  Other setings for the power and concentration dependence or
      optimization.

-  See the example configuration file in the simetuc folder.
-  Add experimental data as two column text data, separated by tabs or
   spaces.
-  Different options:

   -  Lattice creation.
   -  Simulate the dynamics (rise and decay).
   -  Optimization of the energy transfer parameters.

      -  Minimize the deviation between experiment and simulation.

   -  Simulate the steady state.
   -  Simulate the power dependence of each emission.
   -  Simulate the concentration dependence of the dynamics or the
      steady state.

-  All results can be plotted and saved in .hdf5 format.

Documentation
-------------

See the powerpoint `presentation <docs/simetuc_presentation>`__.

TODO
----

-  [ ] Add pressure dependene option: Change the distances of the
   lattice and simulate dynamics or steady-state.
-  [ ] Read experimental data in more formats.

Bugs/Requests
-------------

Please use the `GitHub issue
tracker <https://github.com/pedvide/simetuc/issues>`__ to submit bugs or
request features.

Acknowledgements
----------------

The financial support of the EU FP7 ITN LUMINET (Grant agreement No.
316906) is gratefully acknowledged.

This work was started at the University of Cantabria under Prof. Rafael
Valiente and continued at the University of Bern under PD Dr. Karl
Krämer.

License
-------

Copyright Pedro Villanueva Delgado, 2016.

Distributed under the terms of the `MIT <LICENSE.txt>`__ license,
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
