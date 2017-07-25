# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 00:10:24 2016

@author: Villanueva
"""

import pytest
import os
import numpy as np
import ruamel_yaml as yaml
import copy

from settings_parser import Settings, SettingsFileError, SettingsValueError, SettingsExtraValueWarning

import simetuc.settings as settings
from simetuc.util import temp_config_filename, Excitation, LabelError
from simetuc.util import DecayTransition, IonType, EneryTransferProcess, Transition

test_folder_path = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope='function')
def setup_cte():

    cte_good = dict([
             ('lattice', {'A_conc': 0.3,
                         'N_uc': 20,
                         'S_conc': 0.3,
                         'a': 5.9738,
                         'alpha': 90.0,
                         'b': 5.9738,
                         'beta': 90.0,
                         'c': 3.5297,
                         'gamma': 120.0,
                         'd_max': 100.0,
                         'd_max_coop': 25.0,
                         'name': 'bNaYF4',
                         'sites_occ': [1.0, 0.5],
                         'sites_pos': [(0.0, 0.0, 0.0), (2/3, 1/3, 0.5)],
                         'spacegroup': 'P-6'}),
             ('states',
              dict([('sensitizer_ion_label', 'Yb'),
                           ('sensitizer_states_labels', ['GS', 'ES']),
                           ('activator_ion_label', 'Tm'),
                           ('activator_states_labels',
                            ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2']),
                           ('sensitizer_states', 2),
                           ('activator_states', 7)])),
             ('excitations', {
                  'NIR_1470': [Excitation(IonType.A, 5, 6, False, 9/5, 2e-4, 1e7, 1e-8)],
                  'NIR_800': [Excitation(IonType.A, 0, 3, False, 13/9, 0.0044, 1e7, 1e-8),
                             Excitation(IonType.A, 2, 5, False, 11/9, 0.004, 1e7, 1e-8)],
                  'NIR_980': [Excitation(IonType.S, 0, 1, False, 4/3, 0.0044, 1e7, 1e-8)],
                  'Vis_473': [Excitation(IonType.A, 0, 5, True, 13/9, 0.00093, 1e6, 1e-8)]}
             ),
             ('optimization', {'method': 'SLSQP', 'processes': [EneryTransferProcess([Transition(IonType.A, 5, 3),
                                                                                      Transition(IonType.A, 0, 2)],
                                                                                      mult=6, strength=2893199540.0)],
                               'options': {}}),
             ('power_dependence', [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]),
             ('concentration_dependence', [(0.0, 0.01), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)]),
             ('simulation_params', {'N_steps': 1000,
                                    'N_steps_pulse': 2,
                                    'atol': 1e-15,
                                    'rtol': 0.001}),
             ('decay',
              {'branching_A': {DecayTransition(IonType.A, 2, 1, branching_ratio=0.4),
                                DecayTransition(IonType.A, 3, 1, branching_ratio=0.3),
                                DecayTransition(IonType.A, 4, 3, branching_ratio=0.999),
                                DecayTransition(IonType.A, 5, 1, branching_ratio=0.15),
                                DecayTransition(IonType.A, 5, 2, branching_ratio=0.16),
                                DecayTransition(IonType.A, 5, 3, branching_ratio=0.04),
                                DecayTransition(IonType.A, 5, 4, branching_ratio=0.0),
                                DecayTransition(IonType.A, 6, 1, branching_ratio=0.43)},
               'branching_S': set(),
               'decay_A': {DecayTransition(IonType.A, 1, 0, decay_rate=83.33333333333333),
                            DecayTransition(IonType.A, 2, 0, decay_rate=40000.0),
                            DecayTransition(IonType.A, 3, 0, decay_rate=500.0),
                            DecayTransition(IonType.A, 4, 0, decay_rate=500000.0),
                            DecayTransition(IonType.A, 5, 0, decay_rate=1315.7894736842104),
                            DecayTransition(IonType.A, 6, 0, decay_rate=14814.814814814814)},
               'decay_S': {DecayTransition(IonType.S, 1, 0, decay_rate=400.0)}}),
             ('energy_transfer', {
              'CR50': EneryTransferProcess([Transition(IonType.A, 5, 3),
                                            Transition(IonType.A, 0, 2)],
                                           mult=6, strength=2893199540.0),
              'ETU53': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                             Transition(IonType.A, 3, 1)],
                                            mult=6, strength=2.5e8),
              'ETU55': EneryTransferProcess([Transition(IonType.A, 5, 6),
                                             Transition(IonType.A, 5, 4)],
                                            mult=6, strength=0.0),
              'BackET': EneryTransferProcess([Transition(IonType.A, 3, 0),
                                              Transition(IonType.S, 0, 1)],
                                             mult=6, strength=4502.0),
              'EM': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                          Transition(IonType.S, 0, 1)],
                                         mult=6, strength=45022061400.0),
              'ETU1': EneryTransferProcess([Transition(IonType.S, 1, 0),
                                            Transition(IonType.A, 0, 2)],
                                           mult=6, strength=10000.0)}
             )])


    return cte_good

@pytest.fixture(scope='function')
def setup_cte_full_S(setup_cte):
    copy_cte = copy.deepcopy(setup_cte)
    copy_cte.states['sensitizer_states_labels'] = ['GS', '1ES', '2ES', '3ES']
    copy_cte.states['sensitizer_states'] = 4
    copy_cte.decay['branching_S'] = [DecayTransition(IonType.S, 1, 0, 1.0),
                                        DecayTransition(IonType.S, 2, 1, 0.5),
                                        DecayTransition(IonType.S, 3, 1, 0.01)]
    copy_cte.optimization['processes'] = ['CR50', DecayTransition(IonType.S, 2, 1, 0.5)]

    return copy_cte

data_all_mandatory_ok = data_branch_ok = '''version: 1
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

def test_standard_config(setup_cte):
    ''''Test that the returned Settings instance for a know config file is correct.'''
    filename = os.path.join(test_folder_path, 'test_standard_config.txt')
    cte = settings.load(filename)

    with open(filename, 'rt') as file:
        config_file = file.read()
    setup_cte['config_file'] = config_file

    assert cte.lattice == setup_cte['lattice']
    assert cte.states == setup_cte['states']
    assert cte.excitations == setup_cte['excitations']
    assert cte.energy_transfer == setup_cte['energy_transfer']
    assert cte.optimization == setup_cte['optimization']
    assert cte.power_dependence == setup_cte['power_dependence']
    assert cte.concentration_dependence == setup_cte['concentration_dependence']


def test_non_existing_file():
    with pytest.raises(SettingsFileError) as excinfo:
        # load non existing file
        settings.load(os.path.join(test_folder_path, 'test_non_existing_config.txt'))
    assert excinfo.match(r"Error reading file")
    assert excinfo.type == SettingsFileError

def test_empty_file():
    with pytest.raises(SettingsFileError) as excinfo:
        with temp_config_filename('') as filename:
            settings.load(filename)
    assert excinfo.match(r"The settings file is empty or otherwise invalid")
    assert excinfo.type == SettingsFileError

@pytest.mark.parametrize('bad_yaml_data', [':', '\t', 'key: value:',
                                           'label1:\n    key1:value1'+'label2:\n    key2:value2'],
                          ids=['colon', 'tab', 'bad colon', 'bad value'])
def test_yaml_error_config(bad_yaml_data):
    with pytest.raises(SettingsFileError) as excinfo:
        with temp_config_filename(bad_yaml_data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Error while parsing the config file")
    assert excinfo.type == SettingsFileError

def test_not_dict_config():
    with pytest.raises(SettingsFileError) as excinfo:
        with temp_config_filename('vers') as filename:
            settings.load(filename)
    assert excinfo.match(r"The settings file is empty or otherwise invalid")
    assert excinfo.type == SettingsFileError

def test_version_config():
    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data_all_mandatory_ok.replace('version: 1',
                                                                'version: 2')) as filename:
            settings.load(filename)
    assert excinfo.match(r"cannot be larger than 1")
    assert excinfo.type == SettingsValueError

def idfn(sections_data):
    '''Returns the name of the test according to the parameters'''
    num_l = len(sections_data.splitlines())
    return 'sections_{}'.format(num_l)
import itertools
import operator
# list of mandatory sections
data = '''version: 1
lattice: asd
states: asd
excitations: asd
sensitizer_decay: asd
activator_decay: asd'''
# combinations of sections. At least 1 is missing
list_data = list(itertools.accumulate(data.splitlines(keepends=True)[:-1], operator.concat))
@pytest.mark.parametrize('sections_data', list_data, ids=idfn)
def test_sections_config(sections_data):
    with pytest.raises(SettingsFileError) as excinfo:
        with temp_config_filename(sections_data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Those sections must be present")
    assert excinfo.match(r"Sections that are needed but not present in the file")
    assert excinfo.type == SettingsFileError

# should get a warning for an extra unrecognized section
def test_extra_sections_warning_config():
    data = data_all_mandatory_ok+'''extra_unknown_section: dsa'''
    with pytest.warns(SettingsExtraValueWarning) as warnings:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert len(warnings) == 2 # one warning
    warning = warnings.pop(SettingsExtraValueWarning)
    assert issubclass(warning.category, SettingsExtraValueWarning)
    assert 'Some values or sections should not be present in the file' in str(warning.message)

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
'''
# list of tuples of values for N_uc, S_conc, A_conc, a,b,c, alpha,
lattice_values = [('dsa', 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # text instead of number
(0.3, 0.3, 0.3, 'dsa', 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [2/3, 1/3, 1/2]]', '[1, 1/2]'), # text instead of number
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[]', '[]'), # empty occupancies
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '', ''), # empty occupancies 2
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '0, 0, 0', '[1.1, 1/2]'), # occupancy pos not a list
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0]]', '[1]'), # sites_pos must be list of 3 numbers
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [one/5, 1/3, 1/2]]', '[1, 1/2]'), # sites_pos string
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0], [1/3, 1/3, 1/2]]', '[1/2]'), # different number of occ.
(0.3, 0.3, 0.3, 5.9, 5.9, 3.5, 90, 90, 120, '[[0, 0, 0]]', '[1/2, 0.75]')] # different number of occ. 2
ids=['text instead of number', 'text instead of number',\
'empty occupancies', 'empty occupancies 2', 'occupancies pos not a list', 'sites_pos: list of 3 numbers',\
'sites_pos string', 'different number of occ.', 'different number of occ. 2']
@pytest.mark.parametrize('lattice_values', lattice_values, ids=ids)
def test_lattice_config(lattice_values):
    data_format = data_lattice.format(*lattice_values)
    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data_format) as filename:
            settings.load(filename)
    assert excinfo.type == SettingsValueError

data_lattice_occ_ok = '''lattice:
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

    confs = Settings({'lattice': settings.configs.settings['lattice']})
    with temp_config_filename(data_format) as filename:
        confs.validate(filename)

    for elem in ['name', 'spacegroup', 'N_uc', 'S_conc',
                 'A_conc', 'sites_pos', 'sites_occ']:
        assert elem in confs.lattice

data_lattice_full = '''version: 1
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
'''
def test_lattice_dmax():
    data = data_lattice_full + '''    d_max: 100.0
    d_max_coop: 25.0'''

    confs = Settings({'lattice': settings.configs.settings['lattice']})
    with temp_config_filename(data) as filename:
        confs.validate(filename)

    assert confs.lattice['d_max'] == 100.0

def test_lattice_radius():
    data = data_lattice_full + '''states:
    asd: dsa
excitations:
        asd: dsa
sensitizer_decay:
        asd: dsa
activator_decay:
        asd: dsa'''
    data = data.replace('N_uc: 8', 'radius: 100.0')
    with pytest.raises(SettingsValueError) as excinfo: # ok, error later
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Error validating section "states"')
    assert excinfo.match(r'"sensitizer_ion_label" not in dictionary')
    assert excinfo.type == SettingsValueError

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
'''
    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Error validating section "states"')
    assert excinfo.type == SettingsValueError

def test_states_no_states_labels():
    data = data_lattice_ok + '''states:
        sensitizer_states_labels: [GS, ES]
        activator_ion_label: Tm
        activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(SettingsValueError) as excinfo: # missing key
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Error validating section "states"')
    assert excinfo.match(r'"sensitizer_ion_label" not in dictionary')
    assert excinfo.type == SettingsValueError

def test_states_no_list():
    data = data_lattice_ok + '''states:
        sensitizer_ion_label: Yb
        sensitizer_states_labels:
        activator_ion_label: Tm
        activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(SettingsValueError) as excinfo: # empty S labels
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"sensitizer_states_labels")
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_states_empty_list():  # empty list for S states
    data = data_lattice_ok + '''states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [GS, ES]
    activator_ion_label: Tm
    activator_states_labels: []'''
    with pytest.raises(SettingsValueError) as excinfo: # empty S labels list
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Length of activator_states_labels \(0\) cannot be smaller than 1.')
    assert excinfo.type == SettingsValueError

def test_states_fractions():  # fractions in the state labels
    data = data_lattice_ok + '''states:
    sensitizer_ion_label: Yb
    sensitizer_states_labels: [2F7/2, 2F5/2]
    activator_ion_label: Tm
    activator_states_labels: [3H6, 3F4, 3H5, 3H4, 3F3, 1G4, 1D2]'''
    with pytest.raises(SettingsValueError) as excinfo: # it should fail in the excitations section
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Setting "excitations"')
    assert excinfo.match(r'does not have the right type')
    assert excinfo.type == SettingsValueError

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
'''
def test_excitations_config1():
    data = data_states_ok + '''excitations:'''
    with pytest.raises(SettingsValueError) as excinfo: # no excitations
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"excitations")
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_excitations_config2():
    data = data_states_ok + '''excitations:
    Vis_473:'''
    with pytest.raises(SettingsValueError) as excinfo: # emtpy excitation
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

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
    with pytest.raises(SettingsValueError) as excinfo: # no active excitation
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"At least one excitation must be present and active")
    assert excinfo.type == SettingsValueError

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
    with pytest.raises(SettingsValueError) as excinfo: # power_dens is a string
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_excitations_config5():
    data = data_states_ok + '''excitations:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(SettingsValueError) as excinfo: # label missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_excitations_config7():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        t_pulse: 1e-8 # pulse width, seconds
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(SettingsValueError) as excinfo: # missing power_dens
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match('power_dens')
    assert excinfo.type == SettingsValueError

def test_excitations_parse_excitations():
    data_exc ='''Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        process: Tm(3H6) -> Tm(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    exc_dict = yaml.load(data_exc)

    states_dict = {'activator_states_labels': ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2'],
                   'sensitizer_states_labels': ['GS', 'ES'],
                   'activator_ion_label': 'Tm',
                   'sensitizer_ion_label': 'Yb'}
    exc = settings._parse_excitations(states_dict, exc_dict)
    excitation = exc['Vis_473'][0]

    assert excitation.transition.ion == IonType.A
    assert excitation.transition.state_i == 0
    assert excitation.transition.state_f == 5

def test_abs_config_wrong_ions_labels():
    data = data_states_ok + '''excitations:
    Vis_473:
        active: True
        power_dens: 1e6 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: T(3H6) -> T(1G4) # both ion labels are required
        degeneracy: 13/9
        pump_rate: 9.3e-4 # cm2/J
'''
    with pytest.raises(LabelError) as excinfo: # both labels are wrong
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'Incorrect ion label in excitation: T\(3H6\) -> T\(1G4\)')
    assert excinfo.type == LabelError

def test_abs_config_wrong_ion_label():
    data = data_states_ok + '''excitations:
    NIR_980:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Y(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J
'''
    with pytest.raises(LabelError) as excinfo: # ion labels is wrong
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match('Incorrect ion label in excitation')
    assert excinfo.type == LabelError

def test_abs_config_ok(): # good test
    data = data_states_ok + '''excitations:
    NIR_980:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: Yb(GS)->Yb(ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J
'''
    with temp_config_filename(data) as filename:
        settings.load(filename)

def test_abs_config_ok_ESA(): # ok, ESA settings
    data = data_states_ok + '''excitations:
    NIR_800:
        active: True
        power_dens: 1e7 # power density W/cm^2
        t_pulse: 1e-8 # pulse width, seconds
        process: [Tm(3H6)->Tm(3H4), Tm(3H5)->Tm(1G4)]
        degeneracy: [13/9, 11/9]
        pump_rate: [4.4e-3, 2e-3] # cm2/J
'''
    with temp_config_filename(data) as filename:
        settings.load(filename)

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
    with pytest.raises(SettingsValueError) as excinfo:  # degeneracy list too short
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("pump_rate, degeneracy, and process must have the same number of items")
    assert excinfo.type == SettingsValueError

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
    with pytest.raises(SettingsValueError) as excinfo:  # pump rate must be positive
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("cannot be smaller than 0.")
    assert excinfo.type == SettingsValueError

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
    with pytest.raises(SettingsValueError) as excinfo:  # all ions must be the same
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match("All processes must involve the same ion in")
    assert excinfo.type == SettingsValueError


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
'''

def test_decay_ok():
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
    with pytest.raises(SettingsValueError) as excinfo: # decay rate is string
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_not_all_states_decay():
    data = data_abs_ok + '''sensitizer_decay:
# lifetimes in s
    ES: 1e-3

activator_decay:
'''
    with pytest.raises(SettingsValueError) as excinfo: # all states must have a decay rate
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"activator_decay")
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

    data = data_abs_ok + '''sensitizer_decay:
activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
    with pytest.raises(SettingsValueError) as excinfo: # all states must have a decay rate
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"sensitizer_decay")
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_decay_missing_A_state():
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
    with pytest.raises(SettingsValueError) as excinfo: # 1D2 state missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"All activator states must have a decay rate")
    assert excinfo.type == SettingsValueError

def test_decay_missing_S_state():
    data = data_abs_ok + '''sensitizer_decay:
    1ES: 2e-3

activator_decay:
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6
'''
    data = data.replace('sensitizer_states_labels: [GS, ES]',
                        'sensitizer_states_labels: [GS, 1ES, 2ES]')
    with pytest.raises(SettingsValueError) as excinfo: # 2ES state missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"All sensitizer states must have a decay rate")
    assert excinfo.type == SettingsValueError

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
def test_branch_config_ok(): # all ok
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

def test_branch_config_wrong_label():
    data = data_decay_ok + '''
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

def test_branch_config_value_above_1():
    data = data_decay_ok + '''
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
    with pytest.raises(SettingsValueError) as excinfo: # value above 1.0
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r'cannot be larger than 1')
    assert excinfo.type == SettingsValueError


def test_ET_config_wrong_ion_label():
    data = data_branch_ok + '''energy_transfer:
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

def test_ET_config_wrong_state_label():
    data = data_branch_ok + '''energy_transfer:
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

def test_ET_config_wrong_multipolarity():
    data = data_branch_ok + '''energy_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: fds
        strength: 8.87920884e+08
'''
    with pytest.raises(SettingsValueError) as excinfo: # wrong multipolarity
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_ET_config_wrong_initial_final_ion_label():
    data = data_branch_ok + '''energy_transfer:
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

def test_ET_config_duplicate_ET_labels():
    data = data_branch_ok + '''energy_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Yb(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 8.87920884e+08
    CR50:
        process: Yb(ES) + Tm(3H6) -> Yb(GS) + Tm(3H5)
        multipolarity: 8
        strength: 1e3
'''
    with pytest.raises(SettingsValueError) as excinfo: # duplicate labels
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Duplicate label")
    assert excinfo.type == SettingsValueError

def test_ET_config_missing_strength():
    data = data_branch_ok + '''energy_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
'''
    with pytest.raises(SettingsValueError) as excinfo: # strength missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match('"strength" not in dictionary')
    assert excinfo.type == SettingsValueError

def test_ET_config_missing_ETlabel():
    data = data_branch_ok + '''energy_transfer:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
'''
    with pytest.raises(SettingsValueError) as excinfo: # label missing
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError

def test_ET_ok(): # ok
    data = data_branch_ok + '''energy_transfer:
    CR50:
        process: Tm(1G4) + Tm(3H6) -> Tm(3H4) + Tm(3H5)
        multipolarity: 6
        strength: 1e3
        strength_avg: 1e1
'''
    with temp_config_filename(data) as filename:
            settings.load(filename)

def test_ET_coop_ok(): # ok
    data = data_branch_ok + '''energy_transfer:
    CR50:
        process: Yb(ES) + Yb(ES) + Tm(3H6) -> Yb(GS) + Yb(GS) + Tm(1G4)
        multipolarity: 6
        strength: 1e3
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

energy_transfer:
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

data_ET_ok_full_S = '''version: 1 # mandatory, only 1 is supported at the moment
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
    sensitizer_states_labels: [GS, 1ES, 2ES, 3ES]
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
        process: Yb(GS)->Yb(1ES)
        degeneracy: 4/3
        pump_rate: 4.4e-3 # cm2/J

sensitizer_decay:
# lifetimes in s
    1ES: 2.5e-3
    2ES: 2.5e-4
    3ES: 2.5e-5

activator_decay:
# lifetimes in s
    3F4: 12e-3
    3H5: 25e-6
    3H4: 2e-3
    3F3: 2e-6
    1G4: 760e-6
    1D2: 67.5e-6

sensitizer_branching_ratios:
    2ES -> 1ES: 0.5
    3ES -> 1ES: 0.01

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

energy_transfer:
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
        process:  Yb(1ES) + Tm(3H6) -> Yb(GS) + Tm(3H5)
        multipolarity: 6
        strength: 1e4
    BackET:
        process:  Tm(3H4) + Yb(GS) -> Tm(3H6) + Yb(1ES)
        multipolarity: 6
        strength: 4.50220614e+3
    EM:
        process:  Yb(1ES) + Yb(GS) -> Yb(GS) + Yb(1ES)
        multipolarity: 6
        strength: 4.50220614e+10
'''

# test optimization processes
def test_optim_wrong_proc():
    '''Wrong ET optimization process'''
    data = data_ET_ok + '''optimization:
        processes: [ETU_does_no_exist]
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ET process label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Wrong labels in optimization: processes")
    assert excinfo.type == settings.LabelError

def test_optim_wrong_proc_2():
    '''Wrong ET optimization process'''
    data = data_ET_ok + '''optimization:
        processes: [ETU53, ETU_does_no_exist]
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ET process label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Wrong labels in optimization: processes")
    assert excinfo.type == settings.LabelError

def test_optim_wrong_B_proc():
    '''Wrong branching ration optimization process'''
    data = data_ET_ok + '''optimization:
        processes: [3F3->3H5]
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ET process label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Wrong labels in optimization: processes")
    assert excinfo.type == settings.LabelError

def test_optim_wrong_B_proc_label():
    '''Wrong branching ration optimization process'''
    data = data_ET_ok + '''optimization:
        processes: [3H145->3F4]
'''
    with pytest.raises(settings.LabelError) as excinfo: # wrong ET process label
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Wrong labels in optimization: processes")
    assert excinfo.type == settings.LabelError

@pytest.mark.parametrize('data_proc', [(data_ET_ok, '3H5->3F4',  Transition(IonType.A, 2, 1)),
                                       (data_ET_ok_full_S, '3ES->1ES',  Transition(IonType.S, 3, 1))])
def test_optim_ok_proc(data_proc): # ok
    data = data_proc[0] + '''optimization:
        processes: [ETU53, {}]
'''.format(data_proc[1])
    with temp_config_filename(data) as filename:
        cte = settings.load(filename)

    ETU53 = EneryTransferProcess([Transition(IonType.A, 5, 6), Transition(IonType.A, 3, 1)],
                                 mult=6, strength=2.5e8, name='ETU53')
    assert cte.optimization['processes'] == [ETU53, data_proc[2]]

def test_optim_method(): # ok
    data = data_ET_ok + '''optimization:
        method: COBYLA'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)

    assert cte.optimization['method'] == 'COBYLA'

def test_optim_excitations(): # ok
    data = data_ET_ok + '''optimization:
        excitations: [Vis_473, NIR_980]'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert cte.optimization['excitations'] ==  ['Vis_473', 'NIR_980']

def test_optim_wrong_excitations(): # ok
    data = data_ET_ok + '''optimization:
        excitations: [Vis_473, wrong_label]'''

    with pytest.raises(settings.LabelError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"in optimization: excitations")
    assert excinfo.match(r"not found in excitations section above!")
    assert excinfo.type == settings.LabelError

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
        print(filename)
        cte = settings.load(filename)
    assert cte.simulation_params == dict([('rtol', 1e-3),
                                            ('atol', 1e-15),
                                            ('N_steps_pulse', 100),
                                            ('N_steps', 1000)])


def test_pow_dep_config1(): # ok
    data = data_ET_ok + '''power_dependence: [1e0, 1e7, 8]'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert np.alltrue(cte.power_dependence == np.array([1.00000000e+00, 1.00000000e+01, 1.00000000e+02,
                                                1.00000000e+03, 1.00000000e+04, 1.00000000e+05,
                                                1.00000000e+06, 1.00000000e+07]))

def test_pow_dep_config2(): # not present
    data = data_ET_ok + ''''''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'power_dependence' not in cte

def test_pow_dep_config3(): # empty
    data = data_ET_ok + '''power_dependence: []'''

    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Length of power_dependence")
    assert excinfo.match(r"cannot be smaller than 3")
    assert excinfo.type == SettingsValueError

def test_pow_dep_config4(): # text instead numbers
    data = data_ET_ok + '''power_dependence: [asd, 1e7, 8]'''

    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"does not have the right type")
    assert excinfo.type == SettingsValueError


def test_conc_dep_config1(): # ok
    data = data_ET_ok + '''concentration_dependence: [[0, 1, 2], [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]]'''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
    assert cte.concentration_dependence == [(0.0, 0.01), (1.0, 0.01), (2.0, 0.01), (0.0, 0.1),
                                      (1.0, 0.1), (2.0, 0.1), (0.0, 0.2), (1.0, 0.2),
                                      (2.0, 0.2), (0.0, 0.3), (1.0, 0.3), (2.0, 0.3),
                                      (0.0, 0.4), (1.0, 0.4), (2.0, 0.4), (0.0, 0.5),
                                      (1.0, 0.5), (2.0, 0.5)]

def test_conc_dep_config2(): # not present
    data = data_ET_ok + ''''''

    with temp_config_filename(data) as filename:
        cte = settings.load(filename)
        assert 'concentration_dependence' not in cte

def test_conc_dep_config3(): # ok, but empty
    data = data_ET_ok + '''concentration_dependence: []'''

    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"Length of concentration_dependence")
    assert excinfo.match(r"cannot be smaller than 2")
    assert excinfo.type == SettingsValueError

def test_conc_dep_config4(): # negative number
    data = data_ET_ok + '''concentration_dependence: [[0, 1, 2], [-0.01, 0.1, 0.2, 0.3, 0.4, 0.5]]'''

    with pytest.raises(SettingsValueError) as excinfo:
        with temp_config_filename(data) as filename:
            settings.load(filename)
    assert excinfo.match(r"cannot be smaller than 0")
    assert excinfo.type == SettingsValueError


# test extra value in section lattice
def test_extra_value():
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
    with pytest.warns(SettingsExtraValueWarning) as warnings: # "extra_value" in lattice section
        with temp_config_filename(extra_data) as filename:
            settings.load(filename)
    assert len(warnings) == 1 # one warning
    warning = warnings.pop(SettingsExtraValueWarning)
    assert issubclass(warning.category, SettingsExtraValueWarning)
    assert 'Some values or sections should not be present in the file' in str(warning.message)


