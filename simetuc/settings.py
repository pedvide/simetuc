# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:57 2016

@author: Pedro
"""
# pylint: disable=E1101

from collections import OrderedDict
import re
from fractions import Fraction
#import sys
import logging
# nice debug printing of settings
import pprint
import warnings
import os
import pkg_resources

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


def _load_yaml_file(filename):
    '''Open a yaml filename and loads it into a dictionary
        Exceptions are raised if the file doesn't exist or is invalid
    '''
    logger = logging.getLogger(__name__)

    cte = {}
    try:
        with open(filename) as file:
            # load data as ordered dictionaries so the ET processes are in the right order
            cte = _ordered_load(file, yaml.SafeLoader)
    except OSError as err:
        logger.error('Error reading file!')
        logger.error(err.args)
        raise ConfigError('Error reading file ({})!'.format(filename)) from err
    except yaml.YAMLError as exc:
        logger.error('Error while parsing the config file: %s!', filename)
        if hasattr(exc, 'problem_mark'):
            logger.error(str(exc.problem_mark).strip())
            if exc.context != None:
                logger.error(str(exc.problem).strip() + ' ' + str(exc.context).strip())
            else:
                logger.error(str(exc.problem).strip())
            logger.error('Please correct data and retry.')
        else:
            logger.error('Something went wrong while parsing the config file (%s):', filename)
            logger.error(exc)
        raise ConfigError('Something went wrong while parsing '+
                          'the config file ({})!'.format(filename)) from exc

    return cte

def _ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    '''Load data as ordered dictionaries so the ET processes are in the right order
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


def _check_values(needed_values, present_values_dict, section=None, optional_values=None):
    ''' Check that the needed keys are present in section
        Print error and exit if not
        Print warning in there are extra keys in section
    '''
    logger = logging.getLogger(__name__)

    # check section is a dictionary
    try:
        present_values = set(present_values_dict.keys())
        needed_values = set(needed_values)
    except (AttributeError, TypeError) as err:
        if section is None: # main check for all sections
            msg = 'File has no sections!'
        else:
            msg = 'Section "{}" is empty!'.format(section)
        logger.error(msg)
        raise ConfigError(msg) from err

    if optional_values is None:
        optional_values = set()
    else:
        optional_values = set(optional_values)

    # if present values don't include all needed values
    if not present_values.issuperset(needed_values):
        set_needed_not_present = needed_values - present_values
        if len(set_needed_not_present) > 0:
            if section is not None:
                logger.error('The following values in section "%s" are needed ' +
                             'but not present in the file:', section)
            else:
                logger.error('Sections that are needed but not present in the file:')
            logger.error(set(set_needed_not_present))
            raise ConfigError('The sections or values \
                              {} must be present.'.format(set_needed_not_present))

    set_extra = present_values - needed_values
    # if there are extra values and they aren't optional
    if len(set_extra) > 0 and not set_extra.issubset(optional_values):
        set_not_optional = set_extra - optional_values
        if section is not None:
            logger.warning('''WARNING! The following values in section "%s"
                              are not recognized:''', section)
        else:
            logger.warning('These sections should not be present in the file:')
        logger.warning(set_not_optional)
        warnings.warn('Some values or sections should not be present in the file.', ConfigWarning)

def _get_ion_and_state_labels(string):
    ''' Returns a list of tuples (ion_label, state_label)
    '''
    return re.findall(r'\s*(\w+)\s*\(\s*(\w+)\s*\)', string)

def _get_state_index(list_labels, state_label, section='', process=None, num=None):
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
        logger.error('Incorrect %s%s: %s', section, num_msg, process)
        logger.error(msg)
        raise LabelError(msg) from err
    return index

def _get_ion(list_ion_labels, ion_label, section='', process=None, num=None):
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
        logger.error('Incorrect %s%s: %s', section, num_msg, process)
        logger.error(msg)
        raise LabelError(msg) from err
    return index


def _get_value(dictionary, value, val_type):
    '''Gets the value from the dictionary and makes sure it has the val_type
        Otherwise raises a ValueError exception.
        If the value or label doen't exist, it raises a ConfigError exception
    '''
    logger = logging.getLogger(__name__)

    try:
        val = val_type(dictionary[value])
    except ValueError as err:
        msg = 'Invalid value for parameter "{}"'.format(value)
        logger.error(msg)
        logger.error(err.args)
        raise ValueError(msg) from err
    except KeyError as err:
        msg = 'Missing parameter "{}"'.format(value)
        logger.error(msg)
        logger.error(err.args)
        raise ConfigError(msg) from err
    except TypeError as err:
        msg = 'Label missing in "{}"?'.format(dictionary)
        logger.error(msg)
        logger.error(err.args)
        raise ConfigError(msg) from err

    return val

def _get_positive_value(dictionary, value, val_type):
    '''Gets a value from the dict and raises a ValueError if it's negative'''
    logger = logging.getLogger(__name__)

    val = _get_value(dictionary, value, val_type)
    if val < 0:
        msg = '"{}" is a negative value'.format(value)
        logger.error(msg)
        raise ValueError(msg)

    return val

def _get_int_value(dictionary, value):
    '''Gets an int from the dictionary'''
    return _get_positive_value(dictionary, value, int)

def _get_float_value(dictionary, value):
    '''Gets a float from the dictionary, it converts it into a Fraction first'''
    return float(_get_positive_value(dictionary, value, Fraction))

def _get_string_value(dictionary, value):
    '''Gets a string from the dictionary'''
    return _get_value(dictionary, value, str)

def _get_list_floats(list_vals):
    '''Returns a list of positive floats, it converts it into a Fraction first.
        Raises ValuError if it's not a float or if negative'''
    logger = logging.getLogger(__name__)

    try:
        lst = [float(Fraction(elem)) for elem in list_vals]
    except ValueError as err:
        msg = 'Invalid value in list "{}"'.format(list_vals)
        logger.error(msg)
        logger.error(err.args)
        raise ValueError(msg) from err

    if any(elem < 0 for elem in lst):
        msg = 'Negative value in list "{}"'.format(list_vals)
        logger.error(msg)
        raise ValueError(msg)

    return lst

def _parse_lattice(dict_lattice):
    '''Parses the lattice section of the settings.
        Returns the parsed lattice dict'''
    logger = logging.getLogger(__name__)

    # LATTICE
    needed_keys = ['name', 'N_uc', 'S_conc', 'A_conc', 'a', 'b', 'c',
                   'alpha', 'beta', 'gamma', 'spacegroup',
                   'sites_pos', 'sites_occ']
    # check that all keys are in the file
    _check_values(needed_keys, dict_lattice, 'lattice')

    parsed_dict = {}

    parsed_dict['name'] = _get_string_value(dict_lattice, 'name')
    parsed_dict['spacegroup'] = _get_string_value(dict_lattice, 'spacegroup')
    parsed_dict['N_uc'] = _get_int_value(dict_lattice, 'N_uc')
    parsed_dict['S_conc'] = _get_float_value(dict_lattice, 'S_conc')
    parsed_dict['A_conc'] = _get_float_value(dict_lattice, 'A_conc')
    a_param = _get_float_value(dict_lattice, 'a')
    b_param = _get_float_value(dict_lattice, 'b')
    c_param = _get_float_value(dict_lattice, 'c')
    alpha_param = _get_float_value(dict_lattice, 'alpha')
    beta_param = _get_float_value(dict_lattice, 'beta')
    gamma_param = _get_float_value(dict_lattice, 'gamma')

    # angles should have reasonable values
    if alpha_param > 360 or beta_param > 360 or gamma_param > 360:
        msg = 'The angles must be below 360°.'
        logger.error(msg)
        raise ValueError(msg)

    # calculate cell parameters
    parsed_dict['cell_par'] = [a_param, b_param, c_param,
                               alpha_param, beta_param, gamma_param]

    # deal with the sites positions and occupancies
    sites_pos = [tuple(_get_list_floats(tuple_val)) for tuple_val in dict_lattice['sites_pos']]
    parsed_dict['sites_pos'] = sites_pos

    if not all(len(row) == 3 for row in np.array(sites_pos)):
        msg = 'The sites positions are lists of 3 numbers.'
        logger.error(msg)
        raise ValueError(msg)
    if not np.alltrue(np.array(sites_pos) >= 0) or not np.alltrue(np.array(sites_pos) <= 1):
        msg = 'The sites positions must be between 0 and 1.'
        logger.error(msg)
        raise ValueError(msg)

    sites_occ = _get_list_floats(dict_lattice['sites_occ'])
    parsed_dict['sites_occ'] = sites_occ
    if not all(0 <= val <= 1 for val in sites_occ):
        msg = 'Occupancies must be positive numbers less or equal than 1.0.'
        logger.error(msg)
        raise ValueError(msg)

    if len(sites_pos) < 1:
        msg = 'At least one site is required.'
        logger.error(msg)
        raise ValueError(msg)

    if not len(sites_pos) == len(sites_occ):
        msg = 'The number of sites must be the same in sites_pos and sites_occ.'
        logger.error(msg)
        raise ValueError(msg)

    return parsed_dict

def _parse_excitations(dict_excitations):
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

    parsed_dict = {}

    # for each excitation
    for excitation in dict_excitations:
        exc_dict = dict_excitations[excitation]
        # parsed values go here
        parsed_dict[excitation] = {}
        # check that all keys are in each excitation
        _check_values(needed_keys, exc_dict,
                      'excitations {}'.format(excitation),
                      optional_values=optional_keys)

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

        parsed_dict[excitation]['process'] = list_proc # processed in _parse_absorptions
        parsed_dict[excitation]['active'] = exc_dict['active']

    # at least one excitation must be active
    if not any(dict_excitations[label]['active'] for label in dict_excitations.keys()):
        msg = 'At least one excitation must be active'
        logger.error(msg)
        raise ConfigError(msg)

    return parsed_dict


def _parse_absorptions(dict_states, dict_excitations):
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
            if init_ion == dict_states['sensitizer_ion_label']: # SENSITIZER
                init_ion_num = _get_state_index(sensitizer_labels, init_state,
                                                section='excitation process')
                final_ion_num = _get_state_index(sensitizer_labels, final_state,
                                                 section='excitation process')

                dict_excitations[excitation]['ion_exc'].append('S')
            elif init_ion == dict_states['activator_ion_label']: # ACTIVATOR
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


def _parse_decay_rates(cte):
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
                        1/float(value)) for num, (key, value) in\
                                        enumerate(cte['sensitizer_decay'].items())]
        pos_value_A = [(_get_state_index(activator_labels, key, section='decay rate',
                                         process=key, num=num),
                        1/float(value)) for num, (key, value) in\
                                        enumerate(cte['activator_decay'].items())]
    except ValueError as err:
        logger.error('Invalid value for parameter in decay rates.')
        logger.error(err.args)
        raise

    return (pos_value_S, pos_value_A)

def _parse_branching_ratios(cte):
    '''Parse the branching ratios'''
    logger = logging.getLogger(__name__)

    sensitizer_labels = cte['states']['sensitizer_states_labels']
    activator_labels = cte['states']['activator_states_labels']

    try:
        # list of tuples of states and decay rate
        B_pos_value_S = []
        B_pos_value_A = []
        if cte['sensitizer_branching_ratios'] is not None:
            for num, (key, value) in enumerate(cte['sensitizer_branching_ratios'].items()):
                states_list = ''.join(key.split()).split('->')
                state_i, state_f = (_get_state_index(sensitizer_labels, s,
                                                     section='branching ratio',
                                                     process=key, num=num) for s in states_list)
                B_pos_value_S.append((state_i, state_f, float(value)))
        if cte['activator_branching_ratios'] is not None:
            for num, (key, value) in enumerate(cte['activator_branching_ratios'].items()):
                states_list = ''.join(key.split()).split('->')
                state_i, state_f = (_get_state_index(activator_labels, s,
                                                     section='branching ratio',
                                                     process=key, num=num) for s in states_list)
                B_pos_value_A.append((state_i, state_f, float(value)))
    except ValueError as err:
        logger.error('Invalid value for parameter in branching ratios.')
        logger.error(err.args)
        raise

    return (B_pos_value_S, B_pos_value_A)


def _parse_ET(cte):
    '''Parse the energy transfer processes'''
    logger = logging.getLogger(__name__)

    sensitizer_ion_label = cte['states']['sensitizer_ion_label']
    activator_ion_label = cte['states']['activator_ion_label']
    list_ion_label = [sensitizer_ion_label, activator_ion_label]

    sensitizer_labels = cte['states']['sensitizer_states_labels']
    activator_labels = cte['states']['activator_states_labels']
    tuple_state_labels = (sensitizer_labels, activator_labels)

    # ET PROCESSES.
    # OrderedDict so when we go over the processes we always get them in the same order
    ET_dict = OrderedDict()
    for num, (key, value) in enumerate(cte['enery_transfer'].items()):
        # make sure all three parts are present and of the right type
        name = key
        process = _get_string_value(value, 'process')
        mult = _get_int_value(value, 'multipolarity')
        strength = _get_float_value(value, 'strength')

        # get the ions and states labels involved
        # find all patterns of "spaces,letters,spaces(spaces,letters,spaces)"
        # and get the "letters", spaces may not exist
        list_init_final = _get_ion_and_state_labels(process)
        list_ions_num = [_get_ion(list_ion_label, ion) for ion, label in list_init_final]
        list_indices = [_get_state_index(tuple_state_labels[ion_num], label,
                                         section='ET process',
                                         process=process, num=num)
                        for ion_num, (ion_label, label) in zip(list_ions_num, list_init_final)]

        # get process type: S-S, A-A, S-A, A-S
        list_ions = [ion for ion, label in list_init_final]
        first_ion = list_ions[0]
        second_ion = list_ions[1]
        ET_type = None
        if first_ion == activator_ion_label and second_ion == activator_ion_label:
            ET_type = 'AA'
        elif first_ion == sensitizer_ion_label and second_ion == sensitizer_ion_label:
            ET_type = 'SS'
        elif first_ion == sensitizer_ion_label and second_ion == activator_ion_label:
            ET_type = 'SA'
        elif first_ion == activator_ion_label and second_ion == sensitizer_ion_label:
            ET_type = 'AS'
        else:
            msg = 'Ions must be either activators or sensitizers in ET process.'
            logger.error(msg)
            raise ValueError(msg)

        # store the data
        ET_dict[name] = {}
        ET_dict[name]['indices'] = list_indices
        ET_dict[name]['mult'] = mult
        ET_dict[name]['value'] = strength
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

def _parse_optim_params(dict_optim, dict_ET):
    '''Parse the optional list of ET parameters to optimize'''

    logger = logging.getLogger(__name__)

    list_params = dict_optim
    list_good_params = dict_ET.keys()

    if not set(list_good_params).issuperset(set(list_params)):
        msg = 'Wrong labels in optimization_processes!'
        logger.error(msg)
        raise LabelError(msg)

    return dict_optim

def _parse_simulation_params(user_settings):
    '''Parse the optional simulation parameters
        If some are not given, the default values are used
    '''
    path = pkg_resources.get_distribution('simetuc').location
    full_path = os.path.join(path, 'simetuc', 'config', 'settings.cfg')
    default_settings = _load_yaml_file(full_path)
    default_settings = default_settings['simulation_params']

    optional_keys = ['rtol', 'atol',
                     'N_steps_pulse', 'N_steps']
    # check that only recognized keys are in the file, warn user otherwise
    _check_values([], user_settings, 'simulation_params', optional_values=optional_keys)

    params_types = [float, float, int, int] # type of the parameters

    new_settings = dict(default_settings)
    for num, setting_key in enumerate(optional_keys):
        new_settings[setting_key] = _get_value(user_settings, setting_key, params_types[num])


    return new_settings

def load(filename):
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
                       'sensitizer_decay', 'activator_decay',
                       'sensitizer_branching_ratios', 'activator_branching_ratios']
    optional_sections = ['experimental_data', 'optimization_processes',
                         'enery_transfer', 'simulation_params']
    _check_values(needed_sections, config_cte, optional_values=optional_sections)

    cte = {}

    # LATTICE
    # parse lattice params
    cte['lattice'] = _parse_lattice(config_cte['lattice'])


    # NUMBER OF STATES
    needed_keys = ['sensitizer_ion_label', 'sensitizer_states_labels',
                   'activator_ion_label', 'activator_states_labels']
    # check that all keys are in the file
    _check_values(needed_keys, config_cte['states'], 'states')
    # check that no value is None or empty
    for key, value in config_cte['states'].items():
        if value is None or not value:
            msg = '{} must not be empty'.format(key)
            logger.error(msg)
            raise ValueError(msg)
    # store values
    cte['states'] = {}
    cte['states']['sensitizer_ion_label'] = config_cte['states']['sensitizer_ion_label']
    cte['states']['sensitizer_states_labels'] = config_cte['states']['sensitizer_states_labels']
    cte['states']['activator_ion_label'] = config_cte['states']['activator_ion_label']
    cte['states']['activator_states_labels'] = config_cte['states']['activator_states_labels']
    # store number of states
    cte['states']['sensitizer_states'] = len(config_cte['states']['sensitizer_states_labels'])
    cte['states']['activator_states'] = len(config_cte['states']['activator_states_labels'])


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


    # EXPERIMENTAL DATA # not used anymore
    # not mandatory -> check
#    if 'experimental_data' in config_cte:
#        cte['experimental_data'] = _parse_exp_data(config_cte)


    # OPTIMIZATION PARAMETERS
    # not mandatory -> check
    if 'optimization_processes' in config_cte:
        cte['optimization_processes'] = _parse_optim_params(config_cte['optimization_processes'],
                                                            cte['ET'])


    # SIMULATION PARAMETERS
    # not mandatory -> check
    if 'simulation_params' in config_cte:
        cte['simulation_params'] = _parse_simulation_params(config_cte['simulation_params'])


    # log cte
    # use pretty print
    logger.debug('Settings dump:')
    logger.debug(pprint.pformat(config_cte))

    logger.info('Settings loaded!')
    return cte

if __name__ == "__main__": # pragma: no cover
    cte = load('test/test_settings/test_standard_config.txt')
#    cte = load('config_file.txt')
