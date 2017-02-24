# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:57 2016

@author: Pedro
"""
# pylint: disable=E1101

### TODO: separate the checking of the values to the respective modules.
# This module shouldn't deal with information that concerns other modules
# It's too coupled now.

from collections import OrderedDict
import re
from fractions import Fraction
#import sys
import logging
# nice debug printing of settings
import pprint
import warnings
import os
from pkg_resources import resource_string
from typing import Dict, List, Tuple, Any, Callable

import numpy as np

import yaml


class LabelError(ValueError):
    '''A label in the configuration file is not correct'''
    pass


class ConfigError(SyntaxError):
    '''Something in the configuration file is not correct'''
    pass


class ConfigWarning(UserWarning):
    '''Something in the configuration file is not correct'''
    pass


def _load_yaml_file(filename: str, direct_file: bool = False) -> Dict:
    '''Open a yaml filename and loads it into a dictionary
        Exceptions are raised if the file doesn't exist or is invalid.
        If direct_file=True, filename is actually a file and not a path to one
    '''
    logger = logging.getLogger(__name__)

    cte = {}  # type: Dict
    try:
        if not direct_file:
            with open(filename) as file:
                # load data as ordered dictionaries so the ET processes are in the right order
                cte = _ordered_load(file, yaml.SafeLoader)
        else:
            cte = _ordered_load(filename, yaml.SafeLoader)
    except OSError as err:
        logger.error('Error reading file!')
        logger.error(str(err.args), exc_info=True)
        raise ConfigError('Error reading file ({})!'.format(filename)) from err
    except yaml.YAMLError as exc:
        logger.error('Error while parsing the config file: %s!', filename, exc_info=True)
        if hasattr(exc, 'problem_mark'):
            logger.error(str(exc.problem_mark).strip())
            if exc.context is not None:
                logger.error(str(exc.problem).strip() + ' ' + str(exc.context).strip())
            else:
                logger.error(str(exc.problem).strip())
            logger.error('Please correct data and retry.')
        else:  # pragma: no cover
            logger.error('Something went wrong while parsing the config file (%s):', filename)
            logger.error(exc)
        raise ConfigError('Something went wrong while parsing ' +
                          'the config file ({})!'.format(filename)) from exc

    return cte


def _ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    '''Load data as ordered dictionaries so the ET processes are in the right order
    # not necessary any more, but still used
        http://stackoverflow.com/a/21912744
    '''

    class OrderedLoader(Loader):
        '''Load the yaml file use an OderedDict'''
        pass

    def no_duplicates_constructor(loader, node, deep=False):
        """Check for duplicate keys."""
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                msg = "Duplicate label {}!".format(key)
                raise LabelError(msg)
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value

        # Load the yaml file use an OderedDict
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        no_duplicates_constructor)

    return yaml.load(stream, OrderedLoader)


def _check_values(needed_values_l: List[str], present_values_dict: Dict,
                  section: str = None, optional_values_l: List[str] = None,
                  exclusive_values_l: List[str] = None) -> None:
    ''' Check that the needed keys are present in section
        Print error if not
        Print warning in there are extra keys in section
        Print error if mutually exclusive values are present
        Returns None
    '''
    logger = logging.getLogger(__name__)

    # check section is a dictionary
    try:
        present_values = set(present_values_dict.keys())
        needed_values = set(needed_values_l)
    except (AttributeError, TypeError) as err:
        msg = 'Section "{}" is empty!'.format(section)
        logger.error(msg, exc_info=True)
        raise ConfigError(msg) from err

    if optional_values_l is None:
        optional_values = set()  # type: Set
    else:
        optional_values = set(optional_values_l)

    if exclusive_values_l is None:
        exclusive_values = set()  # type: Set
    else:
        exclusive_values = set(exclusive_values_l)

    # if present values don't include all needed values
    if not present_values.issuperset(needed_values):
        set_needed_not_present = needed_values - present_values
        if section is not None:
            logger.error('The following values in section "%s" are needed ' +
                         'but not present in the file:', section)
        else:
            logger.error('Sections that are needed but not present in the file:')
        logger.error(str(set(set_needed_not_present)))
        raise ConfigError('The sections or values ' +
                          '{} must be present.'.format(set_needed_not_present))

    set_extra = present_values - needed_values
    # if there are extra values and they aren't optional
    if set_extra and not set_extra.issubset(optional_values):
        set_not_optional = set_extra - optional_values
        if section is not None:
            logger.warning('WARNING! The following values in section "%s" ' +
                           'are not recognized:', section)
        else:
            logger.warning('These sections should not be present in the file:')
        logger.warning(str(set_not_optional))
        warnings.warn('Some values or sections should not be present in the file.', ConfigWarning)

    # check exclusive values.
    # this only works for one set of exclusive values. change if more are needed
    # so far only lattice has that (for N_uc and radius)
    if exclusive_values and exclusive_values.issubset(present_values):
        if section is not None:
            logger.error('The following values in section "%s" are mutually exclusive: ', section)
        else:
            logger.error('The following values are mutually exclusive: ')
        logger.error(str(exclusive_values))
        raise ConfigError('Only one of the values {} must be present.'.format(exclusive_values))


def _get_ion_and_state_labels(string: str) -> List[Tuple[str, str]]:
    ''' Returns a list of tuples (ion_label, state_label)'''
    state_re = r'[\w/]+'  # match any letter, including forward slash '/'
    ion_re = r'\w+'  # match any letter
    return re.findall(r'\s*(' + ion_re + r')\s*\(\s*(' + state_re + r')\s*\)', string)


def _get_state_index(list_labels: List[str], state_label: str,
                     section: str = '', process: str = None, num: int = None) -> int:
    ''' Returns the index of the state label in the list_labels
        Print error and exit if it doesn't exist
    '''
    logger = logging.getLogger(__name__)
    if process is None:
        process = state_label
    try:
        index = list_labels.index(state_label)
    except ValueError as err:  # print an error and exit
        msg = err.args[0].replace('in list', 'a valid state label')
        num_msg = ' (number {})'.format(num) if num else ''
        logger.error('Incorrect %s%s: %s', section, num_msg, process, exc_info=True)
        logger.error(msg)
        raise LabelError(msg) from err
    return index


def _get_ion(list_ion_labels: List[str], ion_label: str,
             section: str = '', process: str = None, num: int = None) -> int:
    ''' Returns the index of the ion label in the list_ion_labels
        Print error and exit if it doesn't exist
    '''
    logger = logging.getLogger(__name__)
    if process is None:
        process = ion_label
    try:
        index = list_ion_labels.index(ion_label)
    except ValueError as err:  # print an error and exit
        msg = err.args[0].replace('in list', 'a valid ion label')
        num_msg = ' (number {})'.format(num) if num else ''
        logger.error('Incorrect %s%s: %s', section, num_msg, process, exc_info=True)
        logger.error(msg)
        raise LabelError(msg) from err
    return index


def _get_value(dictionary: Dict, value: str, val_type: Callable) -> Any:
    '''Gets the value from the dictionary and makes sure it has the val_type
        Otherwise raises a ValueError exception.
        If the value or label doen't exist, it raises a ConfigError exception
    '''
    logger = logging.getLogger(__name__)

    try:
        val = val_type(dictionary[value])
    except ValueError as err:
        msg = 'Invalid value for parameter "{}"'.format(value)
        logger.error(msg, exc_info=True)
        logger.error(str(err.args))
        raise ValueError(msg) from err
    except KeyError as err:
        msg = 'Missing parameter "{}"'.format(value)
        logger.error(msg, exc_info=True)
        logger.error(str(err.args))
        raise ConfigError(msg) from err
    except TypeError as err:
        msg = 'Label missing in "{}"?'.format(dictionary)
        logger.error(msg, exc_info=True)
        logger.error(str(err.args))
        raise ConfigError(msg) from err

    return val


def _get_list(dictionary: Dict, value: str, val_type: Callable) -> List[Any]:
    '''Returns a list of positive floats, it converts it into a Fraction first.
        Raises ValuError if it's not a float or if negative'''
    logger = logging.getLogger(__name__)

    try:
        list_vals = dictionary[value]
        lst = [val_type(elem) for elem in list_vals]
    except ValueError as err:
        msg = 'Invalid value in list "{}"'.format(list_vals)
        logger.error(msg, exc_info=True)
        logger.error(str(err.args))
        raise ValueError(msg) from err

    return lst


# forcing values to be positive or negative is better dealt with at each module.
#def _get_positive_value(dictionary: Dict, value: str, val_type: Callable) -> Union[int, float]:
#    '''Gets a value from the dict and raises a ValueError if it's negative'''
#    logger = logging.getLogger(__name__)
#
#    val = _get_value(dictionary, value, val_type)
#    if val < 0:
#        msg = '"{}" is a negative value'.format(value)
#        logger.error(msg, exc_info=True)
#        raise ValueError(msg)
#
#    return val


def _get_int_value(dictionary: Dict, value: str) -> int:
    '''Gets an int from the dictionary'''
    return int(_get_value(dictionary, value, int))


def _get_float_value(dictionary: Dict, value: str) -> float:
    '''Gets a float from the dictionary, it converts it into a Fraction first'''
    return float(_get_value(dictionary, value, Fraction))


def _get_normalized_float_value(dictionary: Dict, value: str) -> float:
    '''Gets a normalized float from the dictionary, it converts it into a Fraction first.
        Value must be between 0 and 1, otherwise a ValueError is raised.
    '''
    logger = logging.getLogger(__name__)
    val = float(_get_value(dictionary, value, Fraction))
    if 0 <= val <= 1.0:
        return val
    else:
        msg = '"{}" is not between 0 and 1.'.format(value)
        logger.error(msg, exc_info=True)
        raise ValueError(msg)


def _get_string_value(dictionary: Dict, value: str) -> str:
    '''Gets a string from the dictionary'''
    return _get_value(dictionary, value, str)


def _get_list_strings(dictionary: Dict, value: str) -> List[str]:
    '''Gets a string from the dictionary'''
    return _get_list(dictionary, value, str)


def _get_list_floats(list_vals: List) -> List[float]:
    '''Returns a list of positive floats, it converts it into a Fraction first.
        Raises ValuError if it's not a float or if negative'''
    logger = logging.getLogger(__name__)

    try:
        lst = [float(Fraction(elem)) for elem in list_vals]  # type: ignore
    except ValueError as err:
        msg = 'Invalid value in list "{}"'.format(list_vals)
        logger.error(msg, exc_info=True)
        logger.error(str(err.args))
        raise ValueError(msg) from err

    if any(elem < 0 for elem in lst):
        msg = 'Negative value in list "{}"'.format(list_vals)
        logger.error(msg, exc_info=True)
        raise ValueError(msg)

    return lst


def _parse_lattice(dict_lattice: Dict) -> Dict:
    '''Parses the lattice section of the settings.
        Returns the parsed lattice dict'''
    logger = logging.getLogger(__name__)

    # LATTICE
    needed_keys = ['name', 'S_conc', 'A_conc', 'a', 'b', 'c',
                   'alpha', 'beta', 'gamma', 'spacegroup',
                   'sites_pos', 'sites_occ']
    exclusive_keys = ['N_uc', 'radius']
    optional_keys = ['d_max', 'd_max_coop'] + exclusive_keys

    # check that all keys are in the file
    _check_values(needed_keys, dict_lattice, 'lattice',
                  optional_values_l=optional_keys, exclusive_values_l=exclusive_keys)

    parsed_dict = {}  # type: Dict

    parsed_dict['name'] = _get_string_value(dict_lattice, 'name')
    parsed_dict['spacegroup'] = _get_string_value(dict_lattice, 'spacegroup')

    parsed_dict['S_conc'] = _get_float_value(dict_lattice, 'S_conc')
    parsed_dict['A_conc'] = _get_float_value(dict_lattice, 'A_conc')

    a_param = _get_float_value(dict_lattice, 'a')
    b_param = _get_float_value(dict_lattice, 'b')
    c_param = _get_float_value(dict_lattice, 'c')
    alpha_param = _get_float_value(dict_lattice, 'alpha')
    beta_param = _get_float_value(dict_lattice, 'beta')
    gamma_param = _get_float_value(dict_lattice, 'gamma')
    parsed_dict['cell_par'] = [a_param, b_param, c_param,
                               alpha_param, beta_param, gamma_param]

    if 'N_uc' in dict_lattice:
        parsed_dict['N_uc'] = _get_int_value(dict_lattice, 'N_uc')
    elif 'radius' in dict_lattice:
        parsed_dict['radius'] = _get_float_value(dict_lattice, 'radius')
        # use enough unit cells for the radius
        min_param = min(parsed_dict['cell_par'][0:3])
        parsed_dict['N_uc'] = int(np.ceil(parsed_dict['radius']/min_param))

    if 'd_max' in dict_lattice:
        d_max = _get_float_value(dict_lattice, 'd_max')
    else:
        d_max = np.inf
    parsed_dict['d_max'] = d_max
    if 'd_max_coop' in dict_lattice:
        d_max_coop = _get_float_value(dict_lattice, 'd_max_coop')
    else:
        d_max_coop = np.inf
    parsed_dict['d_max_coop'] = d_max_coop

    # deal with the sites positions and occupancies
    list_sites_pos = dict_lattice['sites_pos']
    if not isinstance(list_sites_pos, (list, tuple)) or len(list_sites_pos) < 1:
        msg = 'At least one site is required.'
        logger.error(msg)
        raise ValueError(msg)
    # make sure it's a list of lists
    if len(list_sites_pos) == 0 or not isinstance(list_sites_pos[0], (list, tuple)):
        list_sites_pos = [dict_lattice['sites_pos']]
    sites_pos = [tuple(_get_list_floats(tuple_val)) for tuple_val in list_sites_pos]
    parsed_dict['sites_pos'] = sites_pos

    if not all(len(row) == 3 for row in np.array(sites_pos)):
        msg = 'The sites positions are lists of 3 numbers.'
        logger.error(msg)
        raise ValueError(msg)

    list_sites_occ = dict_lattice['sites_occ']
    if not isinstance(list_sites_occ, (list, tuple)):
        list_sites_occ = [dict_lattice['sites_occ']]
    sites_occ = _get_list_floats(list_sites_occ)
    parsed_dict['sites_occ'] = sites_occ

    if not len(sites_pos) == len(sites_occ):
        msg = 'The number of sites must be the same in sites_pos and sites_occ.'
        logger.error(msg)
        raise ValueError(msg)

    return parsed_dict


def _parse_states(dict_states: Dict) -> Dict:
    '''Parses the states section of the settings.
        Returns the parsed states dict'''
    logger = logging.getLogger(__name__)

    needed_keys = ['sensitizer_ion_label', 'sensitizer_states_labels',
                   'activator_ion_label', 'activator_states_labels']
    # check that all keys are in the file
    _check_values(needed_keys, dict_states, 'states')
    # check that no value is None or empty
    for key, value in dict_states.items():
        if value is None or not value:
            msg = '{} must not be empty'.format(key)
            logger.error(msg)
            raise ValueError(msg)
    # store values
    parsed_dict = {}  # type: Dict
    parsed_dict['sensitizer_ion_label'] = _get_string_value(dict_states, 'sensitizer_ion_label')
    parsed_dict['sensitizer_states_labels'] = _get_list_strings(dict_states,
                                                                'sensitizer_states_labels')
    parsed_dict['activator_ion_label'] = _get_string_value(dict_states, 'activator_ion_label')
    parsed_dict['activator_states_labels'] = _get_list_strings(dict_states,
                                                               'activator_states_labels')
    # store number of states
    parsed_dict['sensitizer_states'] = len(parsed_dict['sensitizer_states_labels'])
    parsed_dict['activator_states'] = len(parsed_dict['activator_states_labels'])

    return parsed_dict

def _parse_excitations(dict_excitations: Dict) -> Dict:
    '''Parses the excitation section
        Returns the parsed excitations dict'''
    logger = logging.getLogger(__name__)

    # EXCITATIONS
    needed_keys = ['active', 'power_dens', 'process',
                   'degeneracy', 'pump_rate']
    optional_keys = ['t_pulse']

    # at least one excitation must exist
    if dict_excitations is None:
        msg = 'At least one excitation is mandatory'
        logger.error(msg)
        raise ConfigError(msg)

    parsed_dict = {}  # type: Dict

    # for each excitation
    for excitation in dict_excitations:
        exc_dict = dict_excitations[excitation]
        # parsed values go here
        parsed_dict[excitation] = {}
        # check that all keys are in each excitation
        _check_values(needed_keys, exc_dict,
                      'excitations {}'.format(excitation),
                      optional_values_l=optional_keys)

        # process values and check they are correct
        # if ESA: process, degeneracy and pump_rate are lists

        # make a list if they aren't already
        list_deg = exc_dict['degeneracy']
        list_pump = exc_dict['pump_rate']
        list_proc = exc_dict['process']
        if not isinstance(list_deg, (list, tuple)):
            list_deg = [exc_dict['degeneracy']]
        if not isinstance(list_pump, (list, tuple)):
            list_pump = [exc_dict['pump_rate']]
        if not isinstance(list_proc, (list, tuple)):
            list_proc = [exc_dict['process']]

        # all three must have the same length
        if not len(list_pump) == len(list_deg) == len(list_proc):
            msg = ('pump_rate, degeneracy, and process ' +
                   'must have the same number of items in {}.').format(excitation)
            logger.error(msg)
            raise ValueError(msg)

        if 't_pulse' in dict_excitations[excitation]:
            parsed_dict[excitation]['t_pulse'] = _get_float_value(exc_dict, 't_pulse')
        parsed_dict[excitation]['power_dens'] = _get_float_value(exc_dict, 'power_dens')

        # transform into floats, make sure they are positive
        parsed_dict[excitation]['degeneracy'] = _get_list_floats(list_deg)
        parsed_dict[excitation]['pump_rate'] = _get_list_floats(list_pump)

        parsed_dict[excitation]['process'] = list_proc  # processed in _parse_absorptions
        parsed_dict[excitation]['active'] = exc_dict['active']

    # at least one excitation must be active
    if not any(dict_excitations[label]['active'] for label in dict_excitations.keys()):
        msg = 'At least one excitation must be active'
        logger.error(msg)
        raise ConfigError(msg)

    return parsed_dict


def _parse_absorptions(dict_states: Dict, dict_excitations: Dict) -> None:
    '''Parse the absorption and add to the excitation label the ion that is excited and
        the inital and final states. It makes changes to the argument dictionaries
    '''
    logger = logging.getLogger(__name__)

    sensitizer_labels = dict_states['sensitizer_states_labels']
    activator_labels = dict_states['activator_states_labels']

    # absorption
    # for each excitation
    for excitation in dict_excitations:
        dict_excitations[excitation]['init_state'] = []
        dict_excitations[excitation]['final_state'] = []
        dict_excitations[excitation]['ion_exc'] = []

        # for each process in the excitation
        for process in dict_excitations[excitation]['process']:
            # get the ion and state labels of the process
            ion_state_list = _get_ion_and_state_labels(process)

            init_ion = ion_state_list[0][0]
            final_ion = ion_state_list[1][0]

            init_state = ion_state_list[0][1]
            final_state = ion_state_list[1][1]

            if init_ion != final_ion:
                msg = 'Incorrect ion label in excitation: {}'.format(process)
                logger.error(msg)
                raise ValueError(msg)
            if init_ion == dict_states['sensitizer_ion_label']:  # SENSITIZER
                init_ion_num = _get_state_index(sensitizer_labels, init_state,
                                                section='excitation process')
                final_ion_num = _get_state_index(sensitizer_labels, final_state,
                                                 section='excitation process')

                dict_excitations[excitation]['ion_exc'].append('S')
            elif init_ion == dict_states['activator_ion_label']:  # ACTIVATOR
                init_ion_num = _get_state_index(activator_labels, init_state,
                                                section='excitation process')
                final_ion_num = _get_state_index(activator_labels, final_state,
                                                 section='excitation process')

                dict_excitations[excitation]['ion_exc'].append('A')
            else:
                msg = 'Incorrect ion label in excitation: {}'.format(process)
                logger.error(msg)
                raise ValueError(msg)
            # add to list
            dict_excitations[excitation]['init_state'].append(init_ion_num)
            dict_excitations[excitation]['final_state'].append(final_ion_num)

        # all ions must be the same in the list!
        ion_list = dict_excitations[excitation]['ion_exc']
        if not all(ion == ion_list[0] for ion in ion_list):
            msg = 'All processes must involve the same ion in {}.'.format(excitation)
            logger.error(msg)
            raise ValueError(msg)


def _parse_decay_rates(cte: Dict) -> Tuple[List[Tuple[int, float]],
                                           List[Tuple[int, float]]]:
    '''Parse the decay rates and return two lists with the state index and decay rate'''
    logger = logging.getLogger(__name__)

    # DECAY RATES in inverse seconds

    sensitizer_labels = cte['states']['sensitizer_states_labels']
    activator_labels = cte['states']['activator_states_labels']

    # the number of user-supplied lifetimes must be the same as
    # the number of energy states (minus the GS)
    if cte['sensitizer_decay'] is None or len(cte['sensitizer_decay']) != len(sensitizer_labels)-1:
        msg = 'All sensitizer states must have a decay rate.'
        logger.error(msg)
        raise ConfigError(msg)
    if cte['activator_decay'] is None or len(cte['activator_decay']) != len(activator_labels)-1:
        msg = 'All activator states must have a decay rate.'
        logger.error(msg)
        raise ConfigError(msg)

    try:
        # list of tuples of state and decay rate
        pos_value_S = [(_get_state_index(sensitizer_labels, key, section='decay rate',
                                         process=key, num=num),
                        1/_get_float_value(cte['sensitizer_decay'], key)) for num, key in
                       enumerate(cte['sensitizer_decay'].keys())]
        pos_value_A = [(_get_state_index(activator_labels, key, section='decay rate',
                                         process=key, num=num),
                        1/_get_float_value(cte['activator_decay'], key)) for num, key in
                       enumerate(cte['activator_decay'].keys())]
    except ValueError as err:
        logger.error('Invalid value for parameter in decay rates.')
        logger.error(str(err.args))
        raise

    return (pos_value_S, pos_value_A)


def _get_branching_ratio_indices(process: str, label: List[str]) -> Tuple[int, int]:
    '''Return the initial and final state indices involved in a branching ratio process.
       The ion has the given label states'''
    states_list = ''.join(process.split()).split('->')
    state_i, state_f = (_get_state_index(label, s, section='branching ratio',
                                         process=process) for s in states_list)
    return (state_i, state_f)


def _parse_branching_ratios(cte: Dict) -> Tuple[List[Tuple[int, int, float]],
                                                List[Tuple[int, int, float]]]:
    '''Parse the branching ratios'''
    logger = logging.getLogger(__name__)

    sensitizer_labels = cte['states']['sensitizer_states_labels']
    activator_labels = cte['states']['activator_states_labels']

    try:
        branch_ratios_S = cte.get('sensitizer_branching_ratios', None)
        branch_ratios_A = cte.get('activator_branching_ratios', None)
        if branch_ratios_S is not None:
            # list of tuples of states and decay rate
            B_pos_value_S = [(*_get_branching_ratio_indices(process, sensitizer_labels),
                              _get_normalized_float_value(branch_ratios_S, process))
                             for process in branch_ratios_S]
        else:
            B_pos_value_S = []
        if branch_ratios_A is not None:
            B_pos_value_A = [(*_get_branching_ratio_indices(process, activator_labels),
                              _get_normalized_float_value(branch_ratios_A, process))
                             for process in branch_ratios_A]
        else:
            B_pos_value_A = []
    except ValueError as err:
        logger.error('Invalid value for parameter in branching ratios.')
        logger.error(str(err.args))
        raise

    return (B_pos_value_S, B_pos_value_A)


def _parse_ET(cte: Dict) -> Dict:
    '''Parse the energy transfer processes'''
    logger = logging.getLogger(__name__)

    sensitizer_ion_label = cte['states']['sensitizer_ion_label']
    activator_ion_label = cte['states']['activator_ion_label']
    list_ion_label = [sensitizer_ion_label, activator_ion_label]

    sensitizer_labels = cte['states']['sensitizer_states_labels']
    activator_labels = cte['states']['activator_states_labels']
    tuple_state_labels = (sensitizer_labels, activator_labels)

    # ET PROCESSES.
    ET_dict = OrderedDict()  # type: Dict
    for num, (key, value) in enumerate(cte['enery_transfer'].items()):
        # make sure all three parts are present and of the right type
        name = key
        process = _get_string_value(value, 'process')
        mult = _get_int_value(value, 'multipolarity')
        strength = _get_float_value(value, 'strength')
        # if it doesn't exist, set to 0
        if 'strength_avg' in value:
            strength_avg = _get_float_value(value, 'strength_avg')
        else:
            strength_avg = None

        # get the ions and states labels involved
        # find all patterns of "spaces,letters,spaces(spaces,letters,spaces)"
        # and get the "letters", spaces may not exist
        list_init_final = _get_ion_and_state_labels(process)
        list_ions_num = [_get_ion(list_ion_label, ion) for ion, label in list_init_final]
        list_indices = [_get_state_index(tuple_state_labels[ion_num], label,  # type: ignore
                                         section='ET process',
                                         process=process, num=num)
                        for ion_num, (ion_label, label) in zip(list_ions_num, list_init_final)]

        # get process type: S-S, A-A, S-A, A-S, S-S-A, or A-A-S
        list_ions = [ion for ion, label in list_init_final]
        first_ion = list_ions[0]
        second_ion = list_ions[1]
        if len(list_ions) == 6:
            third_ion = list_ions[2]
        else:
            third_ion = None
        ET_type = None
        if first_ion == activator_ion_label and second_ion == activator_ion_label:
            if third_ion == sensitizer_ion_label:
                ET_type = 'AAS'
            else:
                ET_type = 'AA'

        elif first_ion == sensitizer_ion_label and second_ion == sensitizer_ion_label:
            if third_ion == activator_ion_label:
                ET_type = 'SSA'
            else:
                ET_type = 'SS'
        elif first_ion == sensitizer_ion_label and second_ion == activator_ion_label:
            ET_type = 'SA'
        elif first_ion == activator_ion_label and second_ion == sensitizer_ion_label:
            ET_type = 'AS'
        else:  # pragma: no cover
            msg = 'Ions must be either activators or sensitizers in ET process.'
            logger.error(msg)
            raise ValueError(msg)

        # store the data
        ET_dict[name] = {}
        ET_dict[name]['indices'] = list_indices
        ET_dict[name]['mult'] = mult
        ET_dict[name]['value'] = strength
        if strength_avg is not None:
            ET_dict[name]['value_avg'] = strength_avg
        ET_dict[name]['type'] = ET_type

    return ET_dict

#def _parse_exp_data(cte):  # pragma: no cover
#    '''Parse the experimental data'''
#
#    sensitizer_labels = cte['states']['sensitizer_states_labels']
#    activator_labels = cte['states']['activator_states_labels']
#    tuple_state_labels = (sensitizer_labels, activator_labels)
#
#    sensitizer_ion_label = cte['states']['sensitizer_ion_label']
#    activator_ion_label = cte['states']['activator_ion_label']
#    list_ion_label = [sensitizer_ion_label, activator_ion_label]
#
#    # EXPERIMENTAL DATA
#    # get the ion and state labels
#    ion_label_list = [(_get_ion_and_state_labels(ion_state)[0], filename)
#                      for ion_state, filename in cte['experimental_data'].items()]
#
#    list_ions_num = [_get_ion(list_ion_label, ion) for (ion, state), filename in ion_label_list]
#    # add sensitizer_states to activators so they are in the right position
#    offset = cte['states']['sensitizer_states']
#    offset_list = [offset if num == 1  else 0 for num in list_ions_num]
#    # get state numbers and make sure they exist
#    list_indices = [_get_state_index(tuple_state_labels[ion_num], state_label,
#                                     section='experimental data')
#                    for ion_num, ((ion_label, state_label), filename)
#                    in zip(list_ions_num, ion_label_list)]
#
#    # add the offset to activators
#    ion_number_list = [a+b for a, b in zip(list_indices, offset_list)]
#    # list of filenames, None if the user didn't supply it
#    filenames = (len(sensitizer_labels)+len(activator_labels))*[None]
#    for num, state in enumerate(ion_number_list):
#        filenames[state] = ion_label_list[num][1]
#
#    return filenames


def _parse_optim_params(dict_optim: Dict, dict_ET: Dict, dict_decay: Dict, dict_states: Dict) -> List:
    '''Parse the optional list of parameters to optimize.
       Some of the params are ET, other are branching ratios'''

    logger = logging.getLogger(__name__)

    # ET params that the user has defined
    set_good_ET_params = set(dict_ET.keys())
    # branching ratio params that user has defined
    set_good_B_params_S = set((i, f) for (i, f, v) in dict_decay['B_pos_value_S'])
    set_good_B_params_A = set((i, f) for (i, f, v) in  dict_decay['B_pos_value_A'])

    set_params = set(dict_optim)

    # set of ET params to optimize
    set_ET_params = set_params.intersection(set_good_ET_params)

    # other params should be branching ratios, we need to parse them into (i, f) tuples
    set_other_params = set_params.difference(set_ET_params)

    sensitizer_labels = dict_states['sensitizer_states_labels']
    activator_labels = dict_states['activator_states_labels']

    try:
        if set_other_params:
            # list of tuples of states and decay rate
            B_pos_value_A = [_get_branching_ratio_indices(process, activator_labels)
                             for process in set_other_params]
            # make sure the branching ratio was defined before
#            print(B_pos_value_A)
#            print(set_good_B_params_A | set_good_B_params_S)
            if not set(B_pos_value_A).issubset(set_good_B_params_A | set_good_B_params_S):
                raise ValueError('Unrecognized parameters.')
        else:
             B_pos_value_A = []
    except ValueError as err:
        msg = 'Wrong labels in optimization_processes!'
        logger.error(msg)
        logger.error(str(err.args))
        raise LabelError(msg)

    return list(set_ET_params) + B_pos_value_A


def _parse_simulation_params(user_settings: Dict = None) -> Dict:
    '''Parse the optional simulation parameters
        If some are not given, the default values are used
    '''

    # use the file located where the package is installed
    _log_config_file = 'settings.cfg'
    # resource_string opens the file and gets it as a string. Works inside .egg too
    _log_config_location = resource_string(__name__, os.path.join('config', _log_config_file))
    default_settings = _load_yaml_file(_log_config_location, direct_file=True)
    default_settings = default_settings['simulation_params']

    if user_settings is None:
        user_settings = default_settings

    optional_keys = ['rtol', 'atol',
                     'N_steps_pulse', 'N_steps']
    # check that only recognized keys are in the file, warn user otherwise
    _check_values([], user_settings, 'simulation_params', optional_values_l=optional_keys)

    # type of the parameters
    params_types = [float, float, int, int]  # type: List[Callable]

    new_settings = dict(default_settings)
    for num, setting_key in enumerate(optional_keys):
        new_settings[setting_key] = _get_value(user_settings, setting_key, params_types[num])

    return new_settings


def _parse_power_dependence(user_list: List = None) -> List[float]:
    '''Parses the power dependence list with the minimum, maximum and number of points.'''
    if user_list is None or user_list == []:
        return []

    items = _get_list_floats(user_list)
    min_power = items[0]
    max_power = items[1]
    num_points = int(items[2])

    power_list = np.logspace(np.log10(min_power), np.log10(max_power), num_points)

    return list(power_list)


def _parse_conc_dependence(user_list: Tuple[List[float], List[float]] = None
                          ) -> List[Tuple[float, float]]:
    '''Parses the concentration dependence list with the minimum, maximum and number of points.'''
    if user_list is None or user_list == []:
        return []

    # get the lists of concentrations from the user
    # if empty, set to 0.0
    S_conc_l = _get_list_floats(user_list[0]) or [0.0]
    A_conc_l = _get_list_floats(user_list[1]) or [0.0]

    # make a regular grid of values
    conc_grid = np.meshgrid(S_conc_l, A_conc_l)
    conc_grid[0].shape = (conc_grid[0].size, 1)
    conc_grid[1].shape = (conc_grid[0].size, 1)
    conc_list = [((float(a), float(b))) for a, b in zip(conc_grid[0], conc_grid[1])]

    return conc_list


def _parse_optim_method(optim_val: str) -> str:
    '''Parse and return the optimization method.'''
    return str(optim_val)


def load(filename: str) -> Dict:
    ''' Load filename and extract the settings for the simulations
        If mandatory values are missing, errors are logged
        and exceptions are raised
        Warnings are logged if extra settings are found
    '''
    logger = logging.getLogger(__name__)
    logger.info('Reading settings file (%s)...', filename)

    # load file into cte dictionary.
    # the function checks that the file exists and that there are no errors
    config_cte = _load_yaml_file(filename)

    if config_cte is None or not isinstance(config_cte, dict):
        msg = 'The settings file is empty or otherwise invalid ({})!'.format(filename)
        logger.error(msg)
        raise ConfigError(msg)

    # check version
    if 'version' not in config_cte or config_cte['version'] != 1:
        logger.error('Error in configuration file (%s)!', filename)
        logger.error('Version number must be 1!')
        raise ConfigError('Version number must be 1!')
    del config_cte['version']

    # MAIN CHECK
    # check that all needed sections are in the file
    # and warn the user if there are extra ones
    needed_sections = ['lattice', 'states', 'excitations',
                       'sensitizer_decay', 'activator_decay']
    optional_sections = ['sensitizer_branching_ratios', 'activator_branching_ratios',
                         'optimization_processes',
                         'enery_transfer', 'simulation_params', 'power_dependence',
                         'concentration_dependence', 'optimize_method']
    _check_values(needed_sections, config_cte, optional_values_l=optional_sections)

    cte = {}  # type: Dict

    # store original configuration file
    with open(filename, 'rt') as file:
        cte['config_file'] = file.read()

    # LATTICE
    # parse lattice params
    cte['lattice'] = _parse_lattice(config_cte['lattice'])

    # NUMBER OF STATES
    cte['states'] = _parse_states(config_cte['states'])

    # EXCITATIONS
    cte['excitations'] = _parse_excitations(config_cte['excitations'])

    # ABSORPTIONS
    _parse_absorptions(cte['states'], cte['excitations'])

    # DECAY RATES
    pos_value_S, pos_value_A = _parse_decay_rates(config_cte)
    cte['decay'] = {}
    cte['decay']['pos_value_S'] = pos_value_S
    cte['decay']['pos_value_A'] = pos_value_A

    # BRANCHING RATIOS (from 0 to 1)
    B_pos_value_S, B_pos_value_A = _parse_branching_ratios(config_cte)
    cte['decay']['B_pos_value_S'] = B_pos_value_S
    cte['decay']['B_pos_value_A'] = B_pos_value_A

    # ET PROCESSES.
    # not mandatory -> check
    if 'enery_transfer' in config_cte:
        cte['ET'] = _parse_ET(config_cte)
    else:
        cte['ET'] = {}

    # EXPERIMENTAL DATA # not used anymore
    # not mandatory -> check
#    if 'experimental_data' in config_cte:
#        cte['experimental_data'] = _parse_exp_data(config_cte)

    # OPTIMIZATION PARAMETERS
    # not mandatory -> check
    if 'optimization_processes' in config_cte:
        cte['optimization_processes'] = _parse_optim_params(config_cte['optimization_processes'],
                                                            cte['ET'], cte['decay'], cte['states'])

    # SIMULATION PARAMETERS
    # not mandatory -> check
    if 'simulation_params' in config_cte:
        cte['simulation_params'] = _parse_simulation_params(config_cte['simulation_params'])
    else:
        cte['simulation_params'] = _parse_simulation_params()

    # POWER DEPENDENCE LIST
    # not mandatory -> check
    if 'power_dependence' in config_cte:
        cte['power_dependence'] = _parse_power_dependence(config_cte['power_dependence'])

    # CONCENTRATION DEPENDENCE LIST
    # not mandatory -> check
    if 'concentration_dependence' in config_cte:
        cte['conc_dependence'] = _parse_conc_dependence(config_cte['concentration_dependence'])

    # OPTIMIZE METHOD
    # not mandatory -> check
    if 'optimize_method' in config_cte:
        cte['optimize_method'] = _parse_optim_method(config_cte['optimize_method'])

    # log cte
    # use pretty print
    logger.debug('Settings dump:')
    logger.debug('File dict (config_cte):')
    logger.debug(pprint.pformat(config_cte))
    logger.debug('Parsed dict (cte):')
    logger.debug(pprint.pformat(cte))

    logger.info('Settings loaded!')
    return cte


#if __name__ == "__main__":
#    import simetuc.settings as settings
#    cte = settings.load('config_file_simple.cfg')
