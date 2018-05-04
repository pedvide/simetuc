# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:57 2016

@author: Pedro
"""
# pylint: disable=E1101

import re
#import sys
import logging
# nice debug printing of settings
#import pprint
import os
from pkg_resources import resource_filename
from typing import Dict, List, Tuple, Any, Set

import numpy as np

from settings_parser import Settings, SettingsValueError, SettingsExtraValueWarning

from simetuc.util import LabelError, log_exceptions_warnings, temp_config_filename
from simetuc.util import Transition, Excitation, DecayTransition, IonType, EneryTransferProcess
import simetuc.settings_config as configs


@log_exceptions_warnings
def _parse_lattice(parsed_settings: Settings) -> Dict:
    '''Add cell_par, and right values for d_max, d_max_coop and N_uc if radius is given.'''
    lattice = parsed_settings.lattice.copy()
    lattice['cell_par'] = []
    for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
        lattice['cell_par'].append(lattice[key])

    lattice['d_max'] = lattice.get('d_max', np.inf)
    lattice['d_max_coop'] = lattice.get('d_max_coop', np.inf)

    radius = lattice.get('radius', None)
    if radius:
        # use enough unit cells for the radius
        min_param = min(lattice['cell_par'][0:3])
        lattice['N_uc'] = int(np.ceil(lattice['radius']/min_param))

    return lattice

@log_exceptions_warnings
def _parse_excitations(dict_states: Dict, dict_excitations: Dict) -> Dict:
    '''Parses the excitation section
        Returns the parsed excitations dict'''
    sensitizer_labels = dict_states['sensitizer_states_labels']
    activator_labels = dict_states['activator_states_labels']
    parsed_dict = {}  # type: Dict

    # for each excitation
    for exc_label, exc_dict in dict_excitations.items() or dict().items():
#            # make a list if they aren't already
        if not isinstance(exc_dict['degeneracy'], list):
            exc_dict['degeneracy'] = [exc_dict['degeneracy']]
        if not isinstance(exc_dict['pump_rate'], list):
            exc_dict['pump_rate'] = [exc_dict['pump_rate']]
        if not isinstance(exc_dict['process'], list):
            exc_dict['process'] = [exc_dict['process']]

        # all three must have the same length
        if not (len(exc_dict['degeneracy']) ==
                len(exc_dict['pump_rate']) ==
                len(exc_dict['process'])):
            msg = ('pump_rate, degeneracy, and process ' +
                   'must have the same number of items in {}.').format(exc_label)
            raise SettingsValueError(msg)

        exc_dict['init_state'] = []
        exc_dict['final_state'] = []
        exc_dict['ion_exc'] = []

        parsed_dict[exc_label] = []

        # for each process in the excitation
        for num, process in enumerate(exc_dict['process']):
            # get the ion and state labels of the process
            ion_state_list = _get_ion_and_state_labels(process)

            init_ion = ion_state_list[0][0]
            final_ion = ion_state_list[1][0]

            init_state = ion_state_list[0][1]
            final_state = ion_state_list[1][1]

            if init_ion != final_ion:
                msg = 'Incorrect ion label in excitation: {}'.format(process)
                raise LabelError(msg)

            if init_ion == dict_states['sensitizer_ion_label']:  # SENSITIZER
                init_state_num = _get_state_index(sensitizer_labels, init_state,
                                                  section='excitation process')
                final_state_num = _get_state_index(sensitizer_labels, final_state,
                                                   section='excitation process')

                exc_dict['ion_exc'].append('S')
            elif init_ion == dict_states['activator_ion_label']:  # ACTIVATOR
                init_state_num = _get_state_index(activator_labels, init_state,
                                                  section='excitation process')
                final_state_num = _get_state_index(activator_labels, final_state,
                                                   section='excitation process')

                exc_dict['ion_exc'].append('A')
            else:
                msg = 'Incorrect ion label in excitation: {}'.format(process)
                raise LabelError(msg)

            # add to list
            exc_dict['init_state'].append(init_state_num)
            exc_dict['final_state'].append(final_state_num)

            ion = IonType.S if exc_dict['ion_exc'][-1] is 'S' else IonType.A
            exc_trans = Excitation(ion, init_state_num, final_state_num,
                                   exc_dict['active'], exc_dict['degeneracy'][num],
                                   exc_dict['pump_rate'][num], exc_dict['power_dens'],
                                   exc_dict.get('t_pulse', None),
                                   label_ion=init_ion,
                                   label_i=init_state, label_f=final_state)
            parsed_dict[exc_label].append(exc_trans)

        # all ions must be the same in the list!
        ion_list = exc_dict['ion_exc']
        if not all(ion == ion_list[0] for ion in ion_list):
            msg = 'All processes must involve the same ion in {}.'.format(exc_label)
            raise SettingsValueError(msg)

        dict_excitations[exc_label] = exc_dict

    # at least one excitation must exist and be active
    if not any(dict_excitations[label]['active'] for label in dict_excitations or []):
        msg = 'At least one excitation must be present and active'
        raise SettingsValueError(msg)

    return parsed_dict

@log_exceptions_warnings
def _parse_decay_rates(dict_states: Dict,
                       parsed_settings: Dict) -> Dict[str, Set[DecayTransition]]:
    '''Parse the decay rates and return two lists with the state index and decay rate'''
    sensitizer_labels = dict_states['sensitizer_states_labels']
    sensitizer_ion_label = dict_states['sensitizer_ion_label']
    activator_labels = dict_states['activator_states_labels']
    activator_ion_label = dict_states['activator_ion_label']

    # the number of user-supplied lifetimes must be the same as
    # the number of energy states (minus the GS)
    if parsed_settings['sensitizer_decay'] is None or\
            len(parsed_settings['sensitizer_decay']) != len(sensitizer_labels)-1:
        msg = 'All sensitizer states must have a decay rate.'
        raise SettingsValueError(msg)
    if parsed_settings['activator_decay'] is None or\
            len(parsed_settings['activator_decay']) != len(activator_labels)-1:
        msg = 'All activator states must have a decay rate.'
        raise SettingsValueError(msg)

    parsed_S_decay = parsed_settings['sensitizer_decay']
    parsed_A_decay = parsed_settings['activator_decay']
    try:
        # list of tuples of state and decay rate
        decay_S_state = [(_get_state_index(sensitizer_labels, key, section='decay rate',
                                           process=key, num=num), 1/tau, key)
                         for num, (key, tau) in enumerate(parsed_S_decay.items())]
        decay_S = {DecayTransition(IonType.S, state_i, 0, decay_rate=val,
                                   label_ion=sensitizer_ion_label,
                                   label_i=label, label_f=sensitizer_labels[0])
                   for state_i, val, label in decay_S_state}

        decay_A_state = [(_get_state_index(activator_labels, key, section='decay rate',
                                           process=key, num=num), 1/tau, key)
                         for num, (key, tau) in enumerate(parsed_A_decay.items())]
        decay_A = {DecayTransition(IonType.A, state_i, 0, decay_rate=val,
                                   label_ion=activator_ion_label,
                                   label_i=label, label_f=activator_labels[0])
                   for state_i, val, label in decay_A_state}
    except LabelError:
        raise

    parsed_dict = {}
    parsed_dict['decay_S'] = decay_S
    parsed_dict['decay_A'] = decay_A

    return parsed_dict

@log_exceptions_warnings
def _parse_branching_ratios(parsed_settings: Settings) -> Tuple[Set[DecayTransition],
                                                                Set[DecayTransition]]:
    '''Parse the branching ratios'''
    dict_states = parsed_settings.states
    sensitizer_labels = dict_states['sensitizer_states_labels']
    sensitizer_ion_label = dict_states['sensitizer_ion_label']
    activator_labels = dict_states['activator_states_labels']
    activator_ion_label = dict_states['activator_ion_label']

    branch_ratios_S = parsed_settings.get('sensitizer_branching_ratios', None)
    branch_ratios_A = parsed_settings.get('activator_branching_ratios', None)
    if branch_ratios_S:
        states_val = [(*_get_branching_ratio_indices(process, sensitizer_labels), value)
                      for process, value in branch_ratios_S.items()]
        branch_trans_S = {DecayTransition(IonType.S, state_i, state_f, branching_ratio=val,
                                          label_ion=sensitizer_ion_label,
                                          label_i=sensitizer_labels[state_i],
                                          label_f=sensitizer_labels[state_f])
                          for state_i, state_f, val in states_val
                          if state_f != 0}
    else:
        branch_trans_S = set()
    if branch_ratios_A:
        states_val = [(*_get_branching_ratio_indices(process, activator_labels), value)
                      for process, value in branch_ratios_A.items()]
        branch_trans_A = {DecayTransition(IonType.A, state_i, state_f, branching_ratio=val,
                                          label_ion=activator_ion_label,
                                          label_i=activator_labels[state_i],
                                          label_f=activator_labels[state_f])
                          for state_i, state_f, val in states_val
                          if state_f != 0}
    else:
        branch_trans_A = set()

    return (branch_trans_S, branch_trans_A)

@log_exceptions_warnings
def _parse_ET(parsed_settings: Settings) -> Dict:
    '''Parse the energy transfer processes'''
    dict_states = parsed_settings.states
    sensitizer_ion_label = dict_states['sensitizer_ion_label']
    activator_ion_label = dict_states['activator_ion_label']
    list_ion_label = [sensitizer_ion_label, activator_ion_label]

    sensitizer_labels = dict_states['sensitizer_states_labels']
    activator_labels = dict_states['activator_states_labels']
    tuple_state_labels = (sensitizer_labels, activator_labels)

    # ET PROCESSES.
    ET_dict = {}  # type: Dict
    for num, (name, et_subdict) in enumerate(parsed_settings['energy_transfer'].items()):

        process = et_subdict['process']
        mult = et_subdict['multipolarity']
        strength = et_subdict['strength']
        strength_avg = et_subdict.get('strength_avg', None)

        # get the ions and states labels involved
        list_init_final = _get_ion_and_state_labels(process)
        list_ions_num = [_get_ion_index(list_ion_label, ion) for ion, label in list_init_final]
        list_indices = [_get_state_index(tuple_state_labels[ion_num], label,
                                         section='ET process',
                                         process=process, num=num)
                        for ion_num, (ion_label, label) in zip(list_ions_num, list_init_final)]

        # list with all information about this ET process
        # tuples with ion and states labels and numbers
        list_ion_states = [(ion_label, state_label, ion_num, state_num)
                           for (ion_label, state_label), ion_num, state_num
                           in zip(list_init_final, list_ions_num, list_indices)]
        # fold the list of ion, state labels in two
        # so that each tuple has two tuples with the states belonging to the same transition
        folded_lst = list(zip(list_ion_states[:len(list_ion_states)//2],
                              list_ion_states[len(list_ion_states)//2:]))

        # store the data
        trans_lst = [Transition(IonType(tuple_i[2]), tuple_i[3], tuple_f[3],
                                label_ion=tuple_i[0], label_i=tuple_i[1], label_f=tuple_f[1])
                     for tuple_i, tuple_f in folded_lst]
#            print(trans_lst)
        ET_dict[name] = EneryTransferProcess(trans_lst, mult=mult, strength=strength,
                                             strength_avg=strength_avg, name=name)

    return ET_dict

@log_exceptions_warnings
def _parse_optimization(settings: Settings) -> Dict[str, Any]:
    '''Parse the optional optimization settings.'''
    optim_dict = {}
    optim_settings = settings.get('optimization', {})
    if 'processes' in optim_settings:
        optim_dict['processes'] = _parse_optim_params(settings, optim_settings['processes'])
    else:
        optim_dict['processes'] = list(settings.energy_transfer.values())

    if 'method' in optim_settings:
        optim_dict['method'] = optim_settings['method']

    if 'excitations' in optim_settings:
        optim_dict['excitations'] = optim_settings['excitations']
    for label in optim_settings.get('excitations', []):
        if label not in settings.excitations:
            msg = ('Label "{}" in optimization: excitations '.format(label)
                   + 'not found in excitations section above!')
            raise LabelError(msg)

    optim_dict['options'] = optim_settings.get('options', {})

    return optim_dict

@log_exceptions_warnings
def _match_branching_ratio(settings: Settings, process: str) -> DecayTransition:
    '''Gets a branching ratio process and returns the Transition.
        Raises an exception if it doesn't exist.'''
    for branch_proc in settings.decay['branching_A'] | settings.decay['branching_S']:
        if process in repr(branch_proc):
            return branch_proc

    msg = 'Wrong labels in optimization: processes. ({}).'.format(process)
    raise LabelError(msg)

@log_exceptions_warnings
def _parse_optim_params(settings: Settings, dict_optim: Dict) -> List:
    '''Parse the optional list of parameters to optimize.
       Some of the params are ET, other are branching ratios'''
    # requested params
    set_params = set(dict_optim)

    # ET params that the user has defined before
    set_known_ET_params = set(settings.energy_transfer.keys())

    # set of ET params to optimize
    set_ET_params = set_params.intersection(set_known_ET_params)
    lst_ET_params = [settings.energy_transfer[proc_label] for proc_label in set_ET_params]

    # other params should be branching ratios
    set_other_params = set_params.difference(set_ET_params)
    # list of transitions
    branch_transitions = [_match_branching_ratio(settings, process) for process in set_other_params]

    return lst_ET_params + branch_transitions

@log_exceptions_warnings
def _parse_simulation_params(settings: Settings) -> Dict:
    '''Parse the optional simulation parameters
        If some are not given, the default values are used
    '''
    # use the file located where the package is installed
    _log_config_file = 'settings.cfg'
#     resource_string opens the file and gets it as a string. Works inside .egg too
#    if __name__ != '__main__':
    _log_config_location = resource_filename(__name__, os.path.join('config', _log_config_file))
#    else:
#        _log_config_location = os.path.join('config', _log_config_file)
#    print(_log_config_location)

    user_settings = settings.get('simulation_params', None)

    default_settings = Settings({'simulation_params': configs.settings['simulation_params']})
    default_settings.validate(_log_config_location)
    default_settings = default_settings['simulation_params']

    if user_settings is None:
        return default_settings

    new_settings = dict(default_settings)
    new_settings.update(user_settings)

    return new_settings

@log_exceptions_warnings
def _parse_power_dependence(user_list: Tuple[float, float, int]) -> List[float]:
    '''Parses the power dependence list with the minimum, maximum and number of points.'''
    min_power = user_list[0]
    max_power = user_list[1]
    num_points = int(user_list[2])

    power_list = np.logspace(np.log10(min_power), np.log10(max_power), num_points)
    return list(power_list)

@log_exceptions_warnings
def _parse_conc_dependence(conc_dep_d: Dict, N_uc: int) -> Dict:
    '''Parses the concentration dependence list with
        the minimum, maximum and number of points.'''
    parsed_dict = {}

    # get the lists of concentrations from the user
    # if empty, set to 0.0
    user_list = conc_dep_d['concentrations']
    S_conc_l = user_list[0]
    A_conc_l = user_list[1]

    # make a regular grid of values
    conc_grid = np.meshgrid(S_conc_l, A_conc_l)
    conc_grid[0].shape = (conc_grid[0].size, 1)
    conc_grid[1].shape = (conc_grid[0].size, 1)
    conc_list = [((float(a), float(b))) for a, b in zip(conc_grid[0], conc_grid[1])]
    parsed_dict['concentrations'] = conc_list

    N_uc_list = conc_dep_d.get('N_uc_list', [N_uc]*len(conc_list))

    # if it's smaller extend with the last element
    if len(N_uc_list) < len(conc_list):
        N_uc_list.extend([N_uc_list[-1]]*(len(conc_list) - len(N_uc_list)))
    # if it's too big, raise an error
    elif len(N_uc_list) > len(conc_list):
        msg = 'N_uc_list has more elements than concentrations.'
        raise SettingsValueError(msg)

    parsed_dict['N_uc_list'] = N_uc_list

    return parsed_dict

@log_exceptions_warnings
def load_file(filename: str) -> None:
    ''' Load filename and extract the settings for the simulations
        If mandatory values are missing, errors are logged
        and exceptions are raised
        Warnings are logged if extra settings are found'''
    logger = logging.getLogger(__name__)
    logger.info('Reading settings file (%s)...', filename)

    settings = Settings(configs.settings)
    settings.validate(filename)

    # store original configuration file
    with open(filename, 'rt') as file:
        settings['config_file'] = file.read()

    # LATTICE
    settings.lattice = _parse_lattice(settings)
    # NUMBER OF STATES
    settings.states['sensitizer_states'] = len(settings.states['sensitizer_states_labels'])
    settings.states['activator_states'] = len(settings.states['activator_states_labels'])

    # EXCITATIONS
    settings.excitations = _parse_excitations(settings.states, settings.excitations)

    # DECAY RATES
    settings.decay = _parse_decay_rates(settings.states, settings)
    if 'activator_decay' in settings:
        del settings.activator_decay
    if 'sensitizer_decay' in settings:
        del settings.sensitizer_decay

    # BRANCHING RATIOS (from 0 to 1)
    branching_S, branching_A = _parse_branching_ratios(settings)
    settings.decay['branching_S'] = branching_S
    settings.decay['branching_A'] = branching_A
    if 'activator_branching_ratios' in settings:
        del settings.activator_branching_ratios
    if 'sensitizer_branching_ratios' in settings:
        del settings.sensitizer_branching_ratios

    # ET PROCESSES.
    # not mandatory -> check
    if 'energy_transfer' in settings:
        settings.energy_transfer = _parse_ET(settings)
    else:
        settings.energy_transfer = dict()

    # OPTIMIZATION
    settings.optimization = _parse_optimization(settings)

    # SIMULATION PARAMETERS
    # not mandatory -> check
    settings.simulation_params = _parse_simulation_params(settings)

    # POWER DEPENDENCE LIST
    # not mandatory -> check
    if 'power_dependence' in settings:
        settings.power_dependence = _parse_power_dependence(settings.power_dependence)

    # CONCENTRATION DEPENDENCE LIST
    # not mandatory -> check
    if 'concentration_dependence' in settings:
        settings.concentration_dependence = _parse_conc_dependence(
            settings.concentration_dependence,
            settings.lattice['N_uc'])


    settings['no_console'] = False
    settings['no_plot'] = False

    # log read and parsed settings
    # use pretty print
#    logger.debug('Settings dump:')
#    logger.debug('File dict (config_cte):')
#    logger.debug(pprint.pformat(settings.settings))
#    logger.debug('Validated dict (cte):')
#    logger.debug(repr(settings))

    logger.info('Settings loaded!')

    return settings


def _get_ion_and_state_labels(string: str) -> List[Tuple[str, str]]:
    ''' Returns a list of tuples (ion_label, state_label)'''
    state_re = r'[\w/]+'  # match any letter, including forward slash '/'
    ion_re = r'\w+'  # match any letter
    return re.findall(r'\s*(' + ion_re + r')\s*\(\s*(' + state_re + r')\s*\)', string)


@log_exceptions_warnings
def _get_state_index(list_labels: List[str], state_label: str,
                     section: str = '', process: str = None, num: int = None) -> int:
    ''' Returns the index of the state label in the list_labels
        Print error and exit if it doesn't exist
    '''
    if process is None:
        process = state_label
    try:
        index = list_labels.index(state_label)
    except ValueError as err:  # print an error and exit
        num_msg = ' (number {})'.format(num) if num else ''
        msg1 = 'Incorrect {}{}: {} .'.format(section, num_msg, process)
        msg2 = err.args[0].replace('in list', 'a valid state label')
        raise LabelError(msg1 + msg2) from err
    return index


@log_exceptions_warnings
def _get_ion_index(list_ion_labels: List[str], ion_label: str,
                   section: str = '', process: str = None, num: int = None) -> int:
    ''' Returns the index of the ion label in the list_ion_labels
        Print error and exit if it doesn't exist
    '''
    if process is None:
        process = ion_label
    try:
        index = list_ion_labels.index(ion_label)
    except ValueError as err:  # print an error and exit
        num_msg = ' (number {})'.format(num) if num else ''
        msg1 = 'Incorrect {}{}: {} .'.format(section, num_msg, process)
        msg2 = err.args[0].replace('in list', 'a valid ion label')
        raise LabelError(msg1 + msg2) from err
    return index


def _get_branching_ratio_indices(process: str, label: List[str]) -> Tuple[int, int]:
    '''Return the initial and final state indices involved in a branching ratio process.
       The ion has the given label states'''
    states_list = ''.join(process.split()).split('->')
    state_i, state_f = (_get_state_index(label, s, section='branching ratio',
                                         process=process) for s in states_list)
    return (state_i, state_f)


def load(filename: str) -> Settings:
    '''Creates a new Settings instance and loads the configuration file.
        Returns the Settings instance (dict-like).'''
#    print(filename)
    settings = load_file(filename)
    return settings

def load_from_text(text_data: str) -> Settings:
    '''Creates a new Settings instance and loads the configuration file.
        Returns the Settings instance (dict-like).'''
    with temp_config_filename(text_data) as filename:
        settings = load_file(filename)
    return settings


#if __name__ == "__main__":
##    cte_std = settings.load('test/test_settings/test_standard_config.txt')
#    cte = load('config_file.cfg')
