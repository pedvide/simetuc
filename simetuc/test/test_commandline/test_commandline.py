# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:35:56 2016

@author: Pedro
"""

import pytest
import os

import numpy as np

import simetuc.commandline as commandline
from simetuc.util import temp_config_filename
from simetuc.optimize import OptimSolution


config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_standard_config.cfg')

@pytest.fixture(scope='function')
def no_logging(mocker):
    mocker.patch('simetuc.commandline.logging')
    mocker.patch('os.makedirs')

def test_cli_help(no_logging):
    '''Test that the help works'''
    ext_args = ['-h']
    with pytest.raises(SystemExit) as excinfo:
        commandline.main(ext_args)
    assert excinfo.type == SystemExit

def test_cli_version(no_logging):
    '''Test that the version works'''
    ext_args = ['--version']
    with pytest.raises(SystemExit) as excinfo:
        commandline.main(ext_args)
    assert excinfo.type == SystemExit

def test_cli_verbose_quiet(mocker, no_logging):
    '''Test that the verbose and quiet flags work'''
    # mock the generate function of lattice so we don't do work
    mocked_generate = mocker.patch('simetuc.lattice.generate')

    ext_args = [config_file, '--no-plot', '-v', '-l']
    commandline.main(ext_args)
    assert mocked_generate.call_count == 1
    assert isinstance(mocked_generate.call_args[0][0], dict)

    ext_args = [config_file, '--no-plot', '-q', '-l']
    commandline.main(ext_args)
    assert mocked_generate.call_count == 2

option_list = ['-l', '-d', '-s', '-p', '-c', '-c d', '-o', '-o conc']
@pytest.mark.parametrize('option', option_list, ids=option_list)
def test_cli_main_options(option, mocker, no_logging):
    '''Test that the main options work'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    mocked_lattice = mocker.patch('simetuc.lattice.generate')
    mocked_opt = mocker.patch('simetuc.optimize.optimize_dynamics')
    mocked_opt_conc = mocker.patch('simetuc.optimize.optimize_concentrations')

    ext_args = [config_file, '--no-plot', option]
    commandline.main(ext_args)

    if option in ['-d', '-s', '-p', '-c', '-c d']:
        assert mocked_sim.call_count == 1
    elif option == '-o':
        assert mocked_opt.call_count == 1
    elif option == '-o conc':
        assert mocked_opt_conc.call_count == 1
    elif option == '-l':
        assert mocked_lattice.call_count == 1
    else:
        assert False

def test_cli_conc_dep_dyn(mocker, no_logging):
    '''Test that the concentration dependence of the dynamics works
        it can't be tested above because of the value d
    '''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    ext_args = [config_file, '--no-plot', '-c', 'd']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1

def test_cli_plot_dyn(mocker, no_logging):
    '''Test that not using no-plot works'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    ext_args = [config_file, '-d']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1

def test_cli_no_save(mocker, no_logging):
    '''Test that the --no-save works'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    mocked_save = mocker.patch('simetuc.simulations.DynamicsSolution.save')
    ext_args = [config_file, '--no-plot', '--no-save', '-d']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1
    assert mocked_save.call_count == 0

option_list = ['processes: [CR50]\n    method: SLSQP', '']
@pytest.mark.parametrize('option', option_list, ids=['method=SLSQP', 'no_processes'])
def test_cli_optim_options(mocker, no_logging, option):
    '''Test that the optimization works with the optimization method'''

    mocked_opt = mocker.patch('simetuc.optimize.optimize_dynamics')

    # add optim method to config file
    with open(config_file, 'rt') as file:
        config_content = file.read()
    data = config_content.replace('processes: [CR50]', option)

    with temp_config_filename(data) as new_config_file:
        ext_args = [new_config_file, '--no-plot', '-o']
        commandline.main(ext_args)
        assert mocked_opt.call_count == 1
