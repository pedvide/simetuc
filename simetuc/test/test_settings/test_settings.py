# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 00:10:24 2016

@author: Villanueva
"""

import pytest
import os
import numpy as np
import yaml

import simetuc.settings as settings
from simetuc.util import temp_config_filename

test_folder_path = os.path.dirname(os.path.abspath(__file__))

def test_standard_config():
    cte = settings.load(os.path.join(test_folder_path, 'test_standard_config.txt'))

    cte_good = dict([
             ('lattice',
              dict([('name', 'bNaYF4'),
                           ('N_uc', 20),
                           ('S_conc', 0.3),
                           ('A_conc', 0.3),
                           ('spacegroup', 'P-6'),
                           ('sites_pos',
                            [(0.0, 0.0, 0.0),
                             (0.6666666666666666, 0.3333333333333333, 0.5)]),
                           ('sites_occ', [1.0, 0.5]),
                           ('cell_par',
                            [5.9738, 5.9738, 3.5297, 90.0, 90.0, 120.0])])),
             ('states',
              dict([('sensitizer_ion_label', 'Yb'),
                           ('sensitizer_states_labels', ['GS', 'ES']),
                           ('activator_ion_label', 'Tm'),
                           ('activator_states_labels',
                            ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2']),
                           ('sensitizer_states', 2),
                           ('activator_states', 7)])),
             ('excitations',
                  dict([('Vis_473',
                      dict([('active', True),
                                   ('power_dens', 1000000.0),
                                   ('t_pulse', 1e-08),
                                   ('process', ['Tm(3H6) -> Tm(1G4)']),
                                   ('degeneracy', [1.4444444444444444]),
                                   ('pump_rate', [0.00093]),
                                   ('init_state', [0]),
                                   ('final_state', [5]),
                                   ('ion_exc', ['A'])])),
                 ('NIR_1470',
                      dict([('active', False),
                                   ('power_dens', 10000000.0),
                                   ('t_pulse', 1e-08),
                                   ('process', ['Tm(1G4) -> Tm(1D2)']),
                                   ('degeneracy', [1.8]),
                                   ('pump_rate', [0.0002]),
                                   ('init_state', [5]),
                                   ('final_state', [6]),
                                   ('ion_exc', ['A'])])),
                 ('NIR_980',
                      dict([('active', False),
                                   ('power_dens', 10000000.0),
                                   ('t_pulse', 1e-08),
                                   ('process', ['Yb(GS)->Yb(ES)']),
                                   ('degeneracy', [1.3333333333333333]),
                                   ('pump_rate', [0.0044]),
                                   ('init_state', [0]),
                                   ('final_state', [1]),
                                   ('ion_exc', ['S'])])),
                 ('NIR_800',
                      dict([('active', False),
                                   ('power_dens', 10000000.0),
                                   ('t_pulse', 1e-08),
                                   ('process',
                                    ['Tm(3H6)->Tm(3H4)', 'Tm(3H5)->Tm(1G4)']),
                                   ('degeneracy',
                                    [1.4444444444444444, 1.2222222222222223]),
                                   ('pump_rate', [0.0044, 0.004]),
                                   ('init_state', [0, 2]),
                                   ('final_state', [3, 5]),
                                   ('ion_exc', ['A', 'A'])]))])),
             ('optimization_processes', ['CR50']),
             ('power_dependence', [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]),
             ('conc_dependence', [(0.0, 0.01), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)]),
             ('simulation_params', {'N_steps': 1000,
                                    'N_steps_pulse': 2,
                                    'atol': 1e-15,
                                    'rtol': 0.001}),
             ('decay',
              {'B_pos_value_A': [(1, 0, 1.0),
                (2, 1, 0.4),
                (3, 1, 0.3),
                (4, 3, 0.999),
                (5, 1, 0.15),
                (5, 2, 0.16),
                (5, 3, 0.04),
                (5, 4, 0.0),
                (6, 1, 0.43)],
               'B_pos_value_S': [(1, 0, 1.0)],
               'pos_value_A': [(1, 83.33333333333333),
                (2, 40000.0),
                (3, 500.0),
                (4, 500000.0),
                (5, 1315.7894736842104),
                (6, 14814.814814814814)],
               'pos_value_S': [(1, 400.0)]}),
             ('ET',
              dict([('CR50',
                            {'indices': [5, 0, 3, 2],
                             'mult': 6,
                             'type': 'AA',
                             'value': 887920884.0}),
                           ('ETU53',
                            {'indices': [5, 3, 6, 1],
                             'mult': 6,
                             'type': 'AA',
                             'value': 450220614.0}),
                           ('ETU55',
                            {'indices': [5, 5, 6, 4],
                             'mult': 6,
                             'type': 'AA',
                             'value': 0.0}),
                           ('ETU1',
                            {'indices': [1, 0, 0, 2],
                             'mult': 6,
                             'type': 'SA',
                             'value': 1e4}),
                           ('BackET',
                            {'indices': [3, 0, 0, 1],
                             'mult': 6,
                             'type': 'AS',
                             'value': 4502.20614}),
                           ('EM',
                            {'indices': [1, 0, 0, 1],
                             'mult': 6,
                             'type': 'SS',
                             'value': 45022061400.0})]))])

    assert cte == cte_good

def test_non_existing_config():
    with pytest.raises(settings.ConfigError) as excinfo:
        # load non existing file
        settings.load(os.path.join(test_folder_path, 'test_non_existing_config.txt'))
    assert excinfo.match(r"Error reading file")
    assert excinfo.type == settings.ConfigError

def test_empty_config():
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename('') as filename:
            settings.load(filename)
    assert excinfo.match(r"The settings file is empty or otherwise invalid")
    assert excinfo.type == settings.ConfigError

@pytest.mark.parametrize('bad_yaml_data', [':', '\t', 'key: value:',
                                           'label1:\n    key1:value1'+'label2:\n    key2:value2'],
                          ids=['colon', 'tab', 'bad colon', 'bad value'])
def test_yaml_error_config(bad_yaml_data):
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename(bad_yaml_data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Something went wrong while parsing the config file")
    assert excinfo.type == settings.ConfigError

def test_not_dict_config():
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename('vers') as filename:
            settings.load(filename)
    assert excinfo.match(r"The settings file is empty or otherwise invalid")
    assert excinfo.type == settings.ConfigError

def test_version_config():
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename('version: 2') as filename:
            settings.load(filename)
    assert excinfo.match(r"Version number must be 1!")
    assert excinfo.type == settings.ConfigError

def idfn(sections_data):
    '''Returns the name of the test according to the parameters'''
    num_l = len(sections_data.splitlines())
    return 'sections_{}'.format(num_l)
import itertools
import operator
# list of sections
data = '''version: 1
lattice: asd
states: asd
excitations: asd
sensitizer_decay: asd
activator_decay: asd
sensitizer_branching_ratios: asd
activator_branching_ratios: asd'''
# combinations of sections. At least 1 is missing
list_data = list(itertools.accumulate(data.splitlines(keepends=True)[:-1], operator.concat))
@pytest.mark.parametrize('sections_data', list_data, ids=idfn)
def test_sections_config(sections_data):
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename(sections_data) as filename:
            settings.load(filename)
    assert excinfo.match(r"The sections or values .* must be present")
    assert excinfo.type == settings.ConfigError

# should get a warning for an extra unrecognized section
def test_extra_sections_warning_config():
    data = data_all_mandatory_ok+'''extra_unknown_section: dsa'''
    with pytest.warns(settings.ConfigWarning):
        with temp_config_filename(data) as filename:
            settings.load(filename)

data_lattice = '''version: 1
lattice:
    name: bNaYF4
    N_uc: {}
    # concentration
    S_conc: {}
    A_conc: {}
    # unit cell
    # distances in Angstrom
    a: {}
    b: {}
    c: {}
    # angles in degree
    alpha: {}
    beta: {}
    gamma: {}
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: {}
    sites_occ: {}
states: asd
excitations: asd
sensitizer_decay: asd
activator_decay: asd
sensitizer_branching_ratios: asd
activator_branching_ratios: asd
'''
# list of tuples of values for N_uc, S_conc, A_conc, a,b,c, alpha,
lattice_values = [('dsa', 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # text instead of number
(0.3, 0.3, 0.3, 'dsa', 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # text instead of number
(0.3, 0.3, 0.3, -5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # negative number
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 370, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # angle too large
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[]', '[]'), # empty occupancies
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '', ''), # empty occupancies 2
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '0, 0, 0', '[1.1, 1/2]'), # occupancy pos not a list
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0]]', '[1]'), # sites_pos must be list of 3 numbers
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [-1/3, 1/3, 1/2]]', '[1, 1/2]'), # negative number
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [1/3, 1/3, 1/2]]', '[-1, 1/2]'), # negative number
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [1/3, 1/3, 1/2]]', '[1.1, 1/2]'), # occupancy larger than 1
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [4/3, 1/3, 1/2]]', '[1, 1/2]'), # sites_pos larger than 1
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [one/5, 1/3, 1/2]]', '[1, 1/2]'), # sites_pos string
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [1/3, 1/3, 1/2]]', '[1/2]'), # different number of occ.
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0]]', '[1/2, 0.75]')] # different number of occ. 2
ids=['text instead of number', 'text instead of number', 'negative number',\
'angle too large', 'empty occupancies', 'empty occupancies 2', 'occupancies pos not a list', 'sites_pos: list of 3 numbers', 'negative number',\
'negative number', 'occupancy larger than 1', 'sites_pos larger than 1', 'sites_pos string', 'different number of occ.', 'different number of occ. 2']
@pytest.mark.parametrize('lattice_values', lattice_values, ids=ids)
def test_lattice_config(lattice_values):
    data_format = data_lattice.format(*lattice_values)
    with pytest.raises(ValueError) as excinfo:
        with temp_config_filename(data_format) as filename:
            settings.load(filename)
    assert excinfo.type == ValueError

data_lattice_occ_ok = '''name: bNaYF4
N_uc: 8
# concentration
S_conc: 0.3
A_conc: 0.3
# unit cell
# distances in Angstrom
a: 5.9738
b: 5.9738
c: 3.5297
# angles in degree
alpha: 90.0
beta: 90.0
gamma: 120.0
# the number is also ok for the spacegroup
spacegroup: P-6
# info about sites
sites_pos: {}
sites_occ: {}
'''
# list of tuples of values for sites_pos and sites_occ
lattice_values = [('[0, 0, 0]', '1'), # one pos and occ
('[[0, 0, 0]]', '[1]')] # one pos and occ list of lists
ids=['one pos and occ', 'one pos and occ list of lists']
@pytest.mark.parametrize('lattice_values', lattice_values, ids=ids)
def test_lattice_config_ok_occs(lattice_values):
    data_format = data_lattice_occ_ok.format(*lattice_values)
    lattice_dict = yaml.load(data_format)
    parsed_lattice_dict = settings._parse_lattice(lattice_dict)
    for elem in ['name', 'spacegroup', 'N_uc', 'S_conc',
                 'A_conc', 'cell_par', 'sites_pos', 'sites_occ']:
        assert elem in parsed_lattice_dict

data_lattice_ok = '''version: 1
lattice:
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]
excitations: asd
sensitizer_decay: asd
activator_decay: asd
sensitizer_branching_ratios: asd
activator_branching_ratios: asd
'''
# error b/c section states is not a dictionary
def test_empty_states_config():
    data = '''version: 1
lattice:
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]
states: asd
excitations: asd
sensitizer_decay: asd
activator_decay: asd
sensitizer_branching_ratios: asd
activator_branching_ratios: asd
'''
    with pytest.raises(settings.ConfigError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Section \"states\" is empty")
    assert excinfo.type == settings.ConfigError

def test_states_config1():
    data = data_lattice_ok + '''states:
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(settings.ConfigError) as excinfo: # missing key
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"The sections or values")
    assert excinfo.match(r"sensitizer_ion_label")
    assert excinfo.type == settings.ConfigError

def test_states_config2():
    data = data_lattice_ok + '''states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels:
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(ValueError) as excinfo: # empty key
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"sensitizer_states_labels must not be empty")
    assert excinfo.type == ValueError

def test_states_config3():  # fractions in the state labels
    data = data_lattice_ok + '''states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [2F7/2, 2F5/2]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(TypeError) as excinfo: # it should fail in the excitations section
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"string indices must be integers")
    assert excinfo.type == TypeError

data_states_ok = '''version: 1
lattice:
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]
states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]
sensitizer_decay:
    ES: 2.5e-3
activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
sensitizer_branching_ratios: asd
activator_branching_ratios: asd
'''
def test_excitations_config1():
    data = data_states_ok + '''excitations:'''
    with pytest.raises(settings.ConfigError) as excinfo: # no excitations
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"At least one excitation is mandatory")
    assert excinfo.type == settings.ConfigError

def test_excitations_config2():
    data = data_states_ok + '''excitations:
    Vis_473:'''
    with pytest.raises(settings.ConfigError) as excinfo: # emtpy excitation
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Section .* is empty!")
    assert excinfo.type == settings.ConfigError

def test_excitations_config3():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: False
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(settings.ConfigError) as excinfo: # no active excitation
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"At least one excitation must be active")
    assert excinfo.type == settings.ConfigError

def test_excitations_config4():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        power_dens: dsa # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(ValueError) as excinfo: # power_dens is a string
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Invalid value for parameter \"power_dens\"")
    assert excinfo.type == ValueError

def test_excitations_config5():
    data = data_states_ok + '''excitations:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(settings.ConfigError) as excinfo: # label missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Section .* is empty!")
    assert excinfo.type == settings.ConfigError

def test_excitations_config7():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(settings.ConfigError) as excinfo: # missing power_dens
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"The sections or values")
    assert excinfo.match('power_dens')
    assert excinfo.type == settings.ConfigError

def test_excitations_config8():
    data_exc = '''Vis_473:
    active: True
    power_dens: 1e6 # power density W/cm^2
    process: Tm(3H6) -> Tm(1G4) # both ion labels are required
    degeneracy: 13/9
    pump_rate: 9.3e-4 # cm2/J
'''
    exc_dict = yaml.load(data_exc)
    settings._parse_excitations(exc_dict)

def test_abs_config1():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: T(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(ValueError) as excinfo: # wrong label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Incorrect ion label in excitation: T\(3H6\) -> Tm\(1G4\)')
    assert excinfo.type == ValueError

def test_abs_config2():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: T(3H6) -> T(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(ValueError) as excinfo: # both labels are wrong
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Incorrect ion label in excitation: T\(3H6\) -> T\(1G4\)')
    assert excinfo.type == ValueError

def test_abs_config3():
    data = data_states_ok + '''excitations:
    NIR_980:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Y(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J
'''
    with pytest.raises(ValueError) as excinfo: # ion labels is wrong
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match('Incorrect ion label in excitation')
    assert excinfo.type == ValueError

def test_abs_config4(): # good test
    data = data_states_ok + '''excitations:
    NIR_980:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J
'''
    with pytest.raises(AttributeError) as excinfo: # fails later b/c no branching ratios
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.type == AttributeError

def test_abs_config5(): # ok, ESA settings
    data = data_states_ok + '''excitations:
    NIR_800:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Tm(3H5)->Tm(1G4)]
        degeneracy: [13/9, 11/9]
        pump_rate: [4.4e-3, 2e-3] # cm2/J
'''
    with pytest.raises(AttributeError) as excinfo: # fails later b/c no branching ratios
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.type == AttributeError

def test_abs_config6():
    data = data_states_ok + '''excitations:
    NIR_800:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Tm(3H5)->Tm(1G4)]
        degeneracy: [13/9]
        pump_rate: [4.4e-3, 2e-3] # cm2/J
'''
    with pytest.raises(ValueError) as excinfo:  # degeneracy list too short
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("pump_rate, degeneracy, and process must have the same number of items")
    assert excinfo.type == ValueError

def test_abs_config7():
    data = data_states_ok + '''excitations:
    NIR_800:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Tm(3H5)->Tm(1G4)]
        degeneracy: [13/9, 11/9]
        pump_rate: [4.4e-3, -2e-3] # cm2/J
'''
    with pytest.raises(ValueError) as excinfo:  # pump rate must be positive
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("Negative value in list")
    assert excinfo.type == ValueError

def test_abs_config8():
    data = data_states_ok + '''excitations:
    NIR_800:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Yb(GS)->Yb(ES)]
        degeneracy: [13/9, 11/9]
        pump_rate: [4.4e-3, 2e-3] # cm2/J
'''
    with pytest.raises(ValueError) as excinfo:  # all ions must be the same
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("All processes must involve the same ion in")
    assert excinfo.type == ValueError


data_abs_ok = '''version: 1
lattice:
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]
states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]
excitations:
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
sensitizer_branching_ratios: asd
activator_branching_ratios: asd
'''

def test_decay_config1():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s
    ES: dsa

activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
    with pytest.raises(ValueError) as excinfo: # decay rate is string
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Invalid value for parameter")
    assert excinfo.type == ValueError

def test_decay_config2():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s
    ES: 1e-3

activator_decay:
'''
    with pytest.raises(settings.ConfigError) as excinfo: # all states must have a decay rate
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"All activator states must have a decay rate")
    assert excinfo.type == settings.ConfigError

def test_decay_config3():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s
    ES: 1e-3

activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
'''
    with pytest.raises(settings.ConfigError) as excinfo: # 1D2 state missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"All activator states must have a decay rate")
    assert excinfo.type == settings.ConfigError

def test_decay_config4():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s

activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
    with pytest.raises(settings.ConfigError) as excinfo: # ES state missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"All sensitizer states must have a decay rate")
    assert excinfo.type == settings.ConfigError

def test_decay_config5():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s
    ES: 1e-3

activator_decay:
# lifetimes in s
    34: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong state label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"is not a valid state label")
    assert excinfo.type == settings.LabelError


data_decay_ok = '''version: 1 # mandatory, only 1 is supported at the moment
lattice:
# all fields here are mandatory
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]

states:
# all fields here are mandatory,
# leave empty if necessary (i.e.: just "sensitizer_ion_label" on a line), but don't delete them
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]

excitations:
# the excitation label can be any text
# at this point, only one active excitation is suported
# the t_pulse value is only mandatory for the dynamics, it's ignored in the steady state
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
    NIR_980:
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J

sensitizer_decay:
# lifetimes in s
    ES: 2.5e-3

activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
def test_branch_config1(): # all ok
    data = data_decay_ok + '''sensitizer_branching_ratios:
    ES->GS: 1.0
activator_branching_ratios:
    3H5->3F4: 0.4
    3H4->3F4: 0.3
    3F3->3H4: 0.999
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    1D2->3F4: 0.43
'''
    with temp_config_filename(data) as filename:
        settings.load(filename)

def test_branch_config2():
    data = data_decay_ok + '''
activator_branching_ratios:
    3H5->3F4: 0.4
    3H4->3F4: 0.3
    3F3->3H4: 0.999
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    1D2->3F4: 0.43
'''
    with pytest.raises(settings.ConfigError) as excinfo: # missing sensitizer_branching_ratios
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"The sections or values .* must be present")
    assert excinfo.match(r"sensitizer_branching_ratios")
    assert excinfo.type == settings.ConfigError

def test_branch_config3():
    data = data_decay_ok + '''sensitizer_branching_ratios:
activator_branching_ratios:
    3H->3F4: 0.4
    3H4->3F4: 0.3
    3F3->3H4: 0.999
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    1D2->3F4: 0.43
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong state label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"is not a valid state label")
    assert excinfo.type == settings.LabelError

def test_branch_config4():
    data = data_decay_ok + '''sensitizer_branching_ratios:
activator_branching_ratios:
    3H5->3F4: 1.4
    3H4->3F4: 0.3
    3F3->3H4: 0.999
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    1D2->3F4: 0.43
'''
    with pytest.raises(ValueError) as excinfo: # value above 1.0
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'"3H5->3F4" is not between 0 and 1.')
    assert excinfo.type == ValueError


data_all_mandatory_ok = data_branch_ok = '''version: 1 # mandatory, only 1 is supported at the moment
lattice:
# all fields here are mandatory
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]

states:
# all fields here are mandatory,
# leave empty if necessary (i.e.: just "sensitizer_ion_label" on a line), but don't delete them
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]

excitations:
# the excitation label can be any text
# at this point, only one active excitation is suported
# the t_pulse value is only mandatory for the dynamics, it's ignored in the steady state
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
    NIR_980:
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J

sensitizer_decay:
# lifetimes in s
    ES: 2.5e-3

activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6

sensitizer_branching_ratios:
# nothing. This section is still mandatory, though

activator_branching_ratios:
    # 3H5 and 3H4 to 3F4
    3H5->3F4: 0.4
    3H4->3F4: 0.3
    # 3F3 to 3H4
    3F3->3H4: 0.999
    # 1G4 to 3F4, 3H5, 3H4 and 3F3
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    # 1D2 to 3F4
    1D2->3F4: 0.43
'''
def test_ET_config1():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: T(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ion label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"is not a valid ion label")
    assert excinfo.type == settings.LabelError

def test_ET_config2():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(34) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong state label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"is not a valid state label")
    assert excinfo.type == settings.LabelError

def test_ET_config3():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: fds
        strength: 8.87920884e+08
'''
    with pytest.raises(ValueError) as excinfo: # wrong multipolarity
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Invalid value for parameter \"multipolarity\"")
    assert excinfo.type == ValueError

def test_ET_config4():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Yb(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
'''
    with pytest.raises(settings.LabelError) as excinfo: # initial ion label should be the same
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"is not a valid state label")
    assert excinfo.type == settings.LabelError

def test_ET_config5():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Yb(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
    CR50:
        process: Yb(ES) + Tm(3H6) -> Yb(GS) + Tm(3H5)
        multipolarity: 8
        strength: 1e3
'''
    with pytest.raises(settings.LabelError) as excinfo: # duplicate labels
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Duplicate label")
    assert excinfo.type == settings.LabelError

def test_ET_config6():
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
'''
    with pytest.raises(settings.ConfigError) as excinfo: # strength missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("Missing parameter \"strength\"")
    assert excinfo.type == settings.ConfigError

def test_ET_config7():
    data = data_branch_ok + '''enery_transfer:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
'''
    with pytest.raises(settings.ConfigError) as excinfo: # label missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Label missing in")
    assert excinfo.type == settings.ConfigError

def test_ET_config8(): # ok
    data = data_branch_ok + '''enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 1e3
        strength_avg: 1e1
'''
    with temp_config_filename(data) as filename:
            settings.load(filename)

data_ET_ok = '''version: 1 # mandatory, only 1 is supported at the moment
lattice:
# all fields here are mandatory
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]

states:
# all fields here are mandatory,
# leave empty if necessary (i.e.: just "sensitizer_ion_label" on a line), but don't delete them
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]

excitations:
# the excitation label can be any text
# at this point, only one active excitation is suported
# the t_pulse value is only mandatory for the dynamics, it's ignored in the steady state
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
    NIR_980:
        active: False
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J

sensitizer_decay:
# lifetimes in s
    ES: 2.5e-3

activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6

sensitizer_branching_ratios:
# nothing. This section is still mandatory, though

activator_branching_ratios:
    # 3H5 and 3H4 to 3F4
    3H5->3F4: 0.4
    3H4->3F4: 0.3
    # 3F3 to 3H4
    3F3->3H4: 0.999
    # 1G4 to 3F4, 3H5, 3H4 and 3F3
    1G4->3F4: 0.15
    1G4->3H5: 0.16
    1G4->3H4: 0.04
    1G4->3F3: 0.00
    # 1D2 to 3F4
    1D2->3F4: 0.43

enery_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
    ETU53:
        process:  Tm(1G4) + Tm(3H4) -> Tm(1D2) + Tm(3F4)
        multipolarity: 6
        strength: 4.50220614e+08
    ETU55:
        process:  Tm(1G4) + Tm(1G4) -> Tm(1D2) + Tm(3F3)
        multipolarity: 6
        strength: 0 # 4.50220614e+7
    ETU1:
        process:  Yb(ES) + Tm(3H6) -> Yb(GS) + Tm(3H5)
        multipolarity: 6
        strength: 1e4
    BackET:
        process:  Tm(3H4) + Yb(GS) -> Tm(3H6) + Yb(ES)
        multipolarity: 6
        strength: 4.50220614e+3
    EM:
        process:  Yb(ES) + Yb(GS) -> Yb(GS) + Yb(ES)
        multipolarity: 6
        strength: 4.50220614e+10
'''
#@pytest.mark.skip # not used anymore
#def test_expData_config1():
#    data = data_ET_ok + '''experimental_data:
#    T(1D2): tau_Tm_1D2_exc_Vis_473.txt
#    Tm(1G4): tau_Tm_1G4_exc_Vis_473.txt
#    Tm(3H4): tau_Tm_3H4_exc_Vis_473.txt
#    Tm(3F4): tau_Tm_3F4_exc_Vis_473.txt
#'''
#    with pytest.raises(settings.LabelError) as excinfo: # wrong ion label
#        with temp_config_filename(data) as filename:
#            settings.load(filename)
#    assert excinfo.match(r"is not a valid ion label")
#    assert excinfo.type == settings.LabelError
#@pytest.mark.skip
#def test_expData_config2():
#    data = data_ET_ok + '''experimental_data:
#    Tm(1D2): tau_Tm_1D2_exc_Vis_473.txt
#    Tm(1G4): tau_Tm_1G4_exc_Vis_473.txt
#    Tm(3H4): tau_Tm_3H4_exc_Vis_473.txt
#    Tm(34): tau_Tm_3F4_exc_Vis_473.txt
#'''
#    with pytest.raises(settings.LabelError) as excinfo: # wrong state label
#        with temp_config_filename(data) as filename:
#            settings.load(filename)
#    assert excinfo.match(r"is not a valid state label")
#    assert excinfo.type == settings.LabelError


# test optimization_processes
def test_optim_config1():
    data = data_ET_ok + '''optimization_processes: [ETU_does_no_exist]
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ET process label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Wrong labels in optimization_processes!")
    assert excinfo.type == settings.LabelError

def test_optim_config2(): # ok
    data = data_ET_ok + '''optimization_processes: [ETU53]
'''
    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert cte['optimization_processes'] == ['ETU53']

# test simulation params
data_sim_params = '''simulation_params:
    rtol: 1e-3
    atol: 1e-15
    N_steps_pulse: 100
    N_steps: 1000
'''
def test_sim_params_config1(): # ok
    data = data_ET_ok + data_sim_params

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert cte['simulation_params'] == dict([('rtol', 1e-3),
                                            ('atol', 1e-15),
                                            ('N_steps_pulse', 100),
                                            ('N_steps', 1000)])

def test_sim_params_config2(recwarn): # ok
    data = data_ET_ok + data_sim_params + 'wrong: label'

    with pytest.warns(settings.ConfigWarning): # 'wrong: label' in simulation_params
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(settings.ConfigWarning)
    assert issubclass(warning.category, settings.ConfigWarning)
    assert 'Some values or sections should not be present in the file.' in str(warning.message)


def test_pow_dep_config1(): # ok
    data = data_ET_ok + '''power_dependence: [1e0, 1e7, 8]'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert np.alltrue(cte['power_dependence'] == np.array([1.00000000e+00, 1.00000000e+01, 1.00000000e+02,
                                                1.00000000e+03, 1.00000000e+04, 1.00000000e+05,
                                                1.00000000e+06, 1.00000000e+07]))

def test_pow_dep_config2(): # not present
    data = data_ET_ok + ''''''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'power_dependence' not in cte

def test_pow_dep_config3(): # empty
    data = data_ET_ok + '''power_dependence: []'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'power_dependence' in cte
        assert cte['power_dependence'] == []

def test_pow_dep_config4(): # text instead numbers
    data = data_ET_ok + '''power_dependence: [asd, 1e7, 8]'''

    with pytest.raises(ValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Invalid value in list")
    assert excinfo.type == ValueError


def test_conc_dep_config1(): # ok
    data = data_ET_ok + '''concentration_dependence: [[0, 1, 2], [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]]'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert cte['conc_dependence'] == [(0.0, 0.01), (1.0, 0.01), (2.0, 0.01), (0.0, 0.1),
                                      (1.0, 0.1), (2.0, 0.1), (0.0, 0.2), (1.0, 0.2),
                                      (2.0, 0.2), (0.0, 0.3), (1.0, 0.3), (2.0, 0.3),
                                      (0.0, 0.4), (1.0, 0.4), (2.0, 0.4), (0.0, 0.5),
                                      (1.0, 0.5), (2.0, 0.5)]

def test_conc_dep_config2(): # not present
    data = data_ET_ok + ''''''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'conc_dependence' not in cte

def test_conc_dep_config3(): # ok, but empty
    data = data_ET_ok + '''concentration_dependence: []'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'conc_dependence' in cte
        assert cte['conc_dependence'] == []

def test_conc_dep_config4(): # negative number
    data = data_ET_ok + '''concentration_dependence: [[0, 1, 2], [-0.01, 0.1, 0.2, 0.3, 0.4, 0.5]]'''

    with pytest.raises(ValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Negative value in list")
    assert excinfo.type == ValueError

def test_optim_method(): # ok
    data = data_ET_ok + '''optimize_method: COBYLA'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)

    assert cte['optimize_method'] == 'COBYLA'


# test extra value in section lattice
def test_extra_value(recwarn):
    extra_data = '''version: 1
lattice:
    name: bNaYF4
    N_uc: 8
    # concentration
    S_conc: 0.3
    A_conc: 0.3
    # unit cell
    # distances in Angstrom
    a: 5.9738
    b: 5.9738
    c: 3.5297
    # angles in degree
    alpha: 90
    beta: 90
    gamma: 120
    # the number is also ok for the spacegroup
    spacegroup: P-6
    # info about sites
    sites_pos: [[0, 0, 0], [2/3, 1/3, 1/2]]
    sites_occ: [1, 1/2]
    extra_value: 3
states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]
excitations:
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
sensitizer_decay:
# lifetimes in s
    ES: 2.5e-3
activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
sensitizer_branching_ratios:
activator_branching_ratios:
'''
    with pytest.warns(settings.ConfigWarning): # "extra_value" in lattice section
        with temp_config_filename(extra_data) as filename:
            settings.load(filename)
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(settings.ConfigWarning)
    assert issubclass(warning.category, settings.ConfigWarning)
    assert 'Some values or sections should not be present in the file.' in str(warning.message)

