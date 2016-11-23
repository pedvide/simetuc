# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:35:56 2016

@author: Pedro
"""

import pytest
import os
#import numpy as np

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

def test_cli_verbose_quiet():
    '''Test that the verbose and quiet flags work'''
    ext_args = [config_file, '--no-plot', '-v', '-l']
    commandline.main(ext_args)

    ext_args = [config_file, '--no-plot', '-q', '-l']
    commandline.main(ext_args)

option_list = ['-l', '-d', '-s', '-p', '-c', '-o']
@pytest.mark.parametrize('option', option_list, ids=option_list)
def test_cli_main_options(option):
    '''Test that the main options work'''
    ext_args = [config_file, '--no-plot', option]
    commandline.main(ext_args)

def test_cli_conc_dep_dyn():
    '''Test that theconcentration dependence of the dynamics works
        it can't be tested above because of the value d'''
    ext_args = [config_file, '--no-plot', '-c', 'd']
    commandline.main(ext_args)

def test_cli_save():
    '''Test that the save works'''
    ext_args = [config_file, '--no-plot', '--save', '-d']
    commandline.main(ext_args)

def test_cli_save_txt():
    '''Test that the save works'''
    ext_args = [config_file, '--no-plot', '--save-txt', '-d']
    commandline.main(ext_args)
