# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:35:56 2016

@author: Pedro
"""

import pytest
import os

import numpy as np

import simetuc.commandline as commandline


config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_standard_config.cfg')

def test_cli_help():
    '''Test that the help works'''
    ext_args = ['-h']
    with pytest.raises(SystemExit) as excinfo:
        commandline.main(ext_args)
    assert excinfo.type == SystemExit

def test_cli_version():
    '''Test that the version works'''
    ext_args = ['--version']
    with pytest.raises(SystemExit) as excinfo:
        commandline.main(ext_args)
    assert excinfo.type == SystemExit

def test_cli_verbose_quiet(mocker):
    '''Test that the verbose and quiet flags work'''
    # mock the generate function of lattice so we don't do work
    mocked_generate = mocker.patch('simetuc.lattice.generate')

    ext_args = [config_file, '--no-plot', '-v', '-l']
    commandline.main(ext_args)
    assert mocked_generate.call_count == 1
    assert type(mocked_generate.call_args[0][0]) == dict

    ext_args = [config_file, '--no-plot', '-q', '-l']
    commandline.main(ext_args)
    assert mocked_generate.call_count == 2

option_list = ['-l', '-d', '-s', '-p', '-c', '-o']
@pytest.mark.parametrize('option', option_list, ids=option_list)
def test_cli_main_options(option, mocker):
    '''Test that the main options work'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    mocked_opt = mocker.patch('simetuc.optimize.optimize_dynamics')
    mocked_opt.return_value = (np.array([1.0]), 0.0, 2.0)

    ext_args = [config_file, '--no-plot', option]
    commandline.main(ext_args)

    if option is ['-d', '-s', '-p', '-c']:
        assert mocked_sim.call_count == 1
    elif option == ['-o']:
        assert mocked_opt.call_count == 1

def test_cli_conc_dep_dyn(mocker):
    '''Test that theconcentration dependence of the dynamics works
        it can't be tested above because of the value d
    '''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    ext_args = [config_file, '--no-plot', '-c', 'd']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1

def test_cli_save(mocker):
    '''Test that the save works'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    ext_args = [config_file, '--no-plot', '--save', '-d']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1

def test_cli_save_txt(mocker):
    '''Test that the save works'''
    mocked_sim = mocker.patch('simetuc.simulations.Simulations')
    ext_args = [config_file, '--no-plot', '--save-txt', '-d']
    commandline.main(ext_args)
    assert mocked_sim.call_count == 1
