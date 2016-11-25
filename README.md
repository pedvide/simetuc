# simetuc
Simulating Energy Transfer and Upconversion

------

[![License](https://img.shields.io/github/license/pedvide/simetuc.svg)](https://github.com/pedvide/simetuc/blob/master/LICENSE.txt)
[![Python version](https://img.shields.io/pypi/pyversions/simetuc.svg)](https://pypi.python.org/pypi/simetuc)
[![Pypi version](https://img.shields.io/pypi/v/simetuc.svg)](https://pypi.python.org/pypi/simetuc)
[![Anaconda version](https://anaconda.org/pedvide/simetuc/badges/version.svg)](https://anaconda.org/pedvide/simetuc)
[![Build Status](https://travis-ci.org/pedvide/simetuc.svg?branch=master)](https://travis-ci.org/pedvide/simetuc)
[![Coverage Status](https://coveralls.io/repos/github/pedvide/simetuc/badge.svg?branch=master)](https://coveralls.io/github/pedvide/simetuc?branch=master)


## Installation
--------
Python 3.5 is required (it may work with earlier versions, though).
Installing [Anaconda](https://anaconda.org/) is recommended (preferably 64 bits). 


    $ conda install -c pedvide simetuc

or

    $ pip install simetuc

## Features
-------------
* Command line interface program.
    * Run with: `simetuc config_file.txt [options]`
    * See all options below and with: `simetuc -h`
* The simulations are controlled by a configuration textfile that includes:
    * Information about the host lattice.
    * Energy states.
    * Absorption and excitation.
    * Decay (including branching ratios).
    * Energy transfer.
    * Other setings for the power and concentration dependence or optimization.
* See the example configuration file in the simetuc folder.
* Add experimental data as two column text data, separated by tabs or spaces.
* Different options:
    * Lattice creation.
    * Simulate the dynamics (rise and decay).
    * Optimization of the energy transfer parameters.
        * Minimize the deviation between experiment and simulation.
    * Simulate the steady state.
    * Simulate the power dependence of each emission.
    * Simulate the concentration dependence of the dynamics or the steady state.
* All results can be plotted and saved in .hdf5 format.

## Documentation
-------------
See the powerpoint [presentation](docs/simetuc_presentation).

## TODO
-------------
 - [ ] Add pressure dependene option: Change the distances of the lattice and simulate dynamics or steady-state.
 - [ ] Read experimental data in more formats.

## Bugs/Requests
-------------

Please use the [GitHub issue tracker](https://github.com/pedvide/simetuc/issues) to submit bugs or request features.

## License
-------

Copyright Pedro Villanueva Delgado, 2016.

Distributed under the terms of the [MIT](LICENSE.txt) license, simetuc is free and open source software.

[//]: # (convert .md (github) to .rst (pypi) use: pandoc --from=markdown --to=rst --output=README.rst README.md)
