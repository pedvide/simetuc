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

Python 3.5 is required.
Installing [Anaconda](https://www.continuum.io/downloads) is recommended; it works with Windows (64/32 bits), Linux (64/32 bits) and Mac (64 bits).


    $ conda install -c pedvide simetuc

or

    $ pip install simetuc

## Features

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
* For all options --average uses standard average rate equations instead of microscopic ones.

## Documentation

See the powerpoint [presentation](docs/simetuc_presentation).

## TODO

 - [ ] Add pressure dependence option: Change the distances of the lattice and simulate dynamics or steady-state.
 - [ ] Read experimental data in more formats.
 - [ ] Add cooperative sensitization.
 - [ ] Include pulse frequency for steady state simulations using a non cw laser

## Bugs/Requests

Please use the [GitHub issue tracker](https://github.com/pedvide/simetuc/issues) to submit bugs or request features.

## Publications

This software has been described and used in these publications:

 - Villanueva-Delgado, P.; Krämer, K. W. & Valiente, R. [Simulating Energy Transfer and Upconversion in β-NaYF<sub>4</sub>: Yb<sup>3+</sup>, Tm<sup>3+</sup>](http://pubs.acs.org/doi/10.1021/acs.jpcc.5b06770)
 - Villanueva-Delgado, P.; Krämer, K. W.; Valiente, R.; de Jong, M. & Meijerink, A. [Modeling Blue to UV Upconversion in β-NaYF<sub>4</sub>: Tm<sup>3+</sup>](http://pubs.rsc.org/en/Content/ArticleLanding/2016/CP/C6CP04347J#!divAbstract)

If you use this sofware in a scientific publication, please cite the appropiate articles above.

## Acknowledgements

The financial support of the EU FP7 ITN LUMINET (Grant agreement No. 316906) is gratefully acknowledged.

This work was started at the University of Cantabria under Prof. Rafael Valiente and continued at the University of Bern under PD Dr. Karl Krämer.

## License

Copyright Pedro Villanueva Delgado, 2016.

Distributed under the terms of the [MIT](LICENSE.txt) license, simetuc is free and open source software.

[//]: # (convert .md (github) to .rst (pypi) use: pandoc --from=markdown --to=rst --output=README.rst README.md)
