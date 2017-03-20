# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:57 2016

@author: Pedro
"""
# pylint: disable=E1101

from collections import OrderedDict
import re
#import sys
import logging
# nice debug printing of settings
import pprint
import warnings
import os
import copy
from pkg_resources import resource_string
from typing import Dict, List, Tuple, Any, Union, cast

import numpy as np

import yaml

from simetuc.util import LabelError, ConfigError, ConfigWarning
from simetuc.value import Value
import simetuc.settings_config as configs


class Settings(Dict):
    '''Contains all settings for the simulations,
        along with methods to load and parse settings files.'''

    def __init__(self, cte_dict: Dict = None) -> None:  # pylint: disable=W0231

        if cte_dict is None:
            cte_dict = dict()
        else:
            # make own copy
            cte_dict = copy.deepcopy(cte_dict)

        self.config_file = cte_dict.get('config_file', '')
        self.lattice = cte_dict.get('lattice', {})
        self.states = cte_dict.get('states', {})
        self.excitations = cte_dict.get('excitations', {})
        self.decay = cte_dict.get('decay', {})
        # get either energy_transfer or ET
        self.energy_transfer = cte_dict.get('energy_transfer', cte_dict.get('ET', {}))

        if 'optimization' in cte_dict:
            self.optimization = cte_dict.get('optimization')
        if 'simulation_params' in cte_dict:
            self.simulation_params = cte_dict.get('simulation_params')
        if 'power_dependence' in cte_dict:
            self.power_dependence = cte_dict.get('power_dependence')
        if 'conc_dependence' in cte_dict:
            self.conc_dependence = cte_dict.get('conc_dependence')

        if 'ions' in cte_dict:
            self.ions = cte_dict.get('ions')

        self.no_console = cte_dict.get('no_console', False)
        self.no_plot = cte_dict.get('no_plot', False)

    def __getitem__(self, key: str) -> Any:
        '''Implements Settings[key].'''
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(str(err))

    def get(self, key: str, default: Any = None) -> Any:
        '''Implements settings.get(key, default).'''
        if key in self:
            return self[key]
        else:
            return default

    def __setitem__(self, key: str, value: Any) -> Any:
        '''Implements Settings[key] = value.'''
        setattr(self, key, value)

    def __delitem__(self, key: str) -> Any:
        '''Implements Settings[key] = value.'''
        delattr(self, key)

    def __contains__(self, key: Any) -> bool:
        '''Returns True if the settings contains the key'''
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __bool__(self) -> bool:
        '''Instance is True if all its data structures have been filled out'''
        for var in vars(self).keys():
#            print(var)
            # If the var is not literally False, but empty
            if getattr(self, var) is not False and not getattr(self, var):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        '''Two settings are equal if all their attributes are equal.'''
        if not isinstance(other, Settings):
            return NotImplemented
        for attr in ['config_file', 'lattice', 'states', 'excitations', 'decay']:
            if self[attr] != other[attr]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        '''Define a non-equality test'''
        if not isinstance(other, Settings):
            return NotImplemented
        return not self == other

    def __repr__(self) -> str:
        '''Representation of a settings instance.'''
        return '''{}(lattice={},\n
states={},\n
excitations={},\n
decay={},\n
energy_transfer={})'''.format(self.__class__.__name__, pprint.pformat(self.lattice),
                              pprint.pformat(self.states), pprint.pformat(self.excitations),
                              pprint.pformat(self.decay), pprint.pformat(self.energy_transfer))

    @staticmethod
    def parse_all_values(settings_list: List, config_dict: Dict) -> Dict:
        '''Parses the settings in the config_dict
            using the settings list.'''
        logger = logging.getLogger(__name__)
#        pprint.pprint(config_dict)

        present_values = set(config_dict.keys())

        needed_values = set(val.name for val in settings_list if val.kind is Value.mandatory)
        optional_values = set(val.name for val in settings_list if val.kind is Value.optional)
        exclusive_values = set(val.name for val in settings_list if val.kind is Value.exclusive)
        optional_values = optional_values | exclusive_values

        # if present values don't include all needed values
        if not present_values.issuperset(needed_values):
            set_needed_not_present = needed_values - present_values
            logger.error('Sections that are needed but not present in the file:')
            text = 'sections'
            logger.error(str(set(set_needed_not_present)))
            raise ConfigError('The {} '.format(text) +
                              '{} must be present.'.format(set_needed_not_present))

        set_extra = present_values - needed_values
        # if there are extra values and they aren't optional
        if set_extra and not set_extra.issubset(optional_values):
            set_not_optional = set_extra - optional_values
            logger.warning('WARNING! The following values are not recognized:')
            logger.warning(str(set_not_optional))
            warnings.warn('Some values or sections should not be present '
                          'in the file: ' + str(set_not_optional), ConfigWarning)

        parsed_dict = {}  # type: Dict
        for value in settings_list:
            name = value.name
            if value.kind is not Value.mandatory and (name not in config_dict or config_dict[name] is None):
                continue
            value.name = name
            parsed_dict.update({name: value.parse(config_dict[name])})

#        pprint.pprint(parsed_dict)
        return parsed_dict

    @staticmethod
    def _parse_excitations(dict_states: Dict, dict_excitations: Dict) -> Dict:
        '''Parses the excitation section
            Returns the parsed excitations dict'''
        logger = logging.getLogger(__name__)

        sensitizer_labels = dict_states['sensitizer_states_labels']
        activator_labels = dict_states['activator_states_labels']

        # for each excitation
        for exc_label, exc_dict in dict_excitations.items() or dict().items():

#            # make a list if they aren't already
            list_deg = exc_dict['degeneracy']
            list_pump = exc_dict['pump_rate']
            list_proc = exc_dict['process']
            if not isinstance(list_deg, (list, tuple)):
                list_deg = [exc_dict['degeneracy']]
            if not isinstance(list_pump, (list, tuple)):
                list_pump = [exc_dict['pump_rate']]
            if not isinstance(list_proc, (list, tuple)):
                list_proc = [exc_dict['process']]
            exc_dict['degeneracy'] = list_deg
            exc_dict['pump_rate'] = list_pump
            exc_dict['process'] = list_proc

            # all three must have the same length
            if not len(list_pump) == len(list_deg) == len(list_proc):
                msg = ('pump_rate, degeneracy, and process ' +
                       'must have the same number of items in {}.').format(exc_label)
                logger.error(msg)
                raise ValueError(msg)

            exc_dict['init_state'] = []
            exc_dict['final_state'] = []
            exc_dict['ion_exc'] = []

            # for each process in the excitation
            for process in list_proc:
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

                    exc_dict['ion_exc'].append('S')
                elif init_ion == dict_states['activator_ion_label']:  # ACTIVATOR
                    init_ion_num = _get_state_index(activator_labels, init_state,
                                                    section='excitation process')
                    final_ion_num = _get_state_index(activator_labels, final_state,
                                                     section='excitation process')

                    exc_dict['ion_exc'].append('A')
                else:
                    msg = 'Incorrect ion label in excitation: {}'.format(process)
                    logger.error(msg)
                    raise ValueError(msg)
                # add to list
                exc_dict['init_state'].append(init_ion_num)
                exc_dict['final_state'].append(final_ion_num)

            # all ions must be the same in the list!
            ion_list = exc_dict['ion_exc']
            if not all(ion == ion_list[0] for ion in ion_list):
                msg = 'All processes must involve the same ion in {}.'.format(exc_label)
                logger.error(msg)
                raise ValueError(msg)

            dict_excitations[exc_label] = exc_dict

        # at least one excitation must exist and be active
        if not any(dict_excitations[label]['active'] for label in dict_excitations or []):
            msg = 'At least one excitation must be present and active'
            logger.error(msg)
            raise ConfigError(msg)

        return dict_excitations

    @staticmethod
    def _parse_decay_rates(parsed_settings: Dict) -> Tuple[List[Tuple[int, float]],
                                                      List[Tuple[int, float]]]:
        '''Parse the decay rates and return two lists with the state index and decay rate'''
        logger = logging.getLogger(__name__)

        # DECAY RATES in inverse seconds

        sensitizer_labels = parsed_settings['states']['sensitizer_states_labels']
        activator_labels = parsed_settings['states']['activator_states_labels']

        # the number of user-supplied lifetimes must be the same as
        # the number of energy states (minus the GS)
        if parsed_settings['sensitizer_decay'] is None or\
                len(parsed_settings['sensitizer_decay']) != len(sensitizer_labels)-1:
            msg = 'All sensitizer states must have a decay rate.'
            logger.error(msg)
            raise ConfigError(msg)
        if parsed_settings['activator_decay'] is None or\
                len(parsed_settings['activator_decay']) != len(activator_labels)-1:
            msg = 'All activator states must have a decay rate.'
            logger.error(msg)
            raise ConfigError(msg)

        parsed_S_decay = parsed_settings['sensitizer_decay']
        parsed_A_decay = parsed_settings['activator_decay']

        try:
            # list of tuples of state and decay rate
            pos_value_S = [(_get_state_index(sensitizer_labels, key, section='decay rate',
                                             process=key, num=num), 1/tau)
                           for num, (key, tau) in enumerate(parsed_S_decay.items())]
            pos_value_A = [(_get_state_index(activator_labels, key, section='decay rate',
                                             process=key, num=num), 1/tau)
                           for num, (key, tau) in enumerate(parsed_A_decay.items())]
        except ValueError as err:
            logger.error('Invalid value for parameter in decay rates.')
            logger.error(str(err.args))
            raise

        return (pos_value_S, pos_value_A)

    @staticmethod
    def _parse_branching_ratios(parsed_settings: Dict) -> Tuple[List[Tuple[int, int, float]],
                                                           List[Tuple[int, int, float]]]:
        '''Parse the branching ratios'''
        sensitizer_labels = parsed_settings['states']['sensitizer_states_labels']
        activator_labels = parsed_settings['states']['activator_states_labels']

        branch_ratios_S = parsed_settings.get('sensitizer_branching_ratios', None)
        branch_ratios_A = parsed_settings.get('activator_branching_ratios', None)
        if branch_ratios_S:
            # list of tuples of states and decay rate
            B_pos_value_S = [(*_get_branching_ratio_indices(process, sensitizer_labels), value)
                             for process, value in branch_ratios_S.items()]
        else:
            B_pos_value_S = []
        if branch_ratios_A:
            B_pos_value_A = [(*_get_branching_ratio_indices(process, activator_labels), value)
                             for process, value in branch_ratios_A.items()]
        else:
            B_pos_value_A = []

        return (B_pos_value_S, B_pos_value_A)

    @staticmethod
    def _parse_ET(parsed_dict: Dict) -> Dict:
        '''Parse the energy transfer processes'''
        logger = logging.getLogger(__name__)

        sensitizer_ion_label = parsed_dict['states']['sensitizer_ion_label']
        activator_ion_label = parsed_dict['states']['activator_ion_label']
        list_ion_label = [sensitizer_ion_label, activator_ion_label]

        sensitizer_labels = parsed_dict['states']['sensitizer_states_labels']
        activator_labels = parsed_dict['states']['activator_states_labels']
        tuple_state_labels = (sensitizer_labels, activator_labels)

        # ET PROCESSES.
        ET_dict = OrderedDict()  # type: Dict
        for num, (name, et_subdict) in enumerate(parsed_dict['energy_transfer'].items()):

            process = et_subdict['process']
            mult = et_subdict['multipolarity']
            strength = et_subdict['strength']
            strength_avg = et_subdict.get('strength_avg', None)

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

    @staticmethod
    def _parse_optimization(parsed_dict: Dict, dict_ET: Dict,
                            dict_decay: Dict, dict_states: Dict, dict_exc: Dict) -> Dict[str, Any]:
        '''Parse the optional optimization settings.'''
        logger = logging.getLogger(__name__)

        if 'processes' in parsed_dict:
            parsed_dict['processes'] = Settings._parse_optim_params(parsed_dict['processes'],
                                                                    dict_ET, dict_decay,
                                                                    dict_states)

        parsed_dict['method'] = parsed_dict.get('method', None)

        for label in parsed_dict.get('excitations', []):
            if label not in dict_exc:
                msg = ('Label "{}" in optimization: excitations '.format(label)
                       + 'not found in excitations section above!')
                logger.error(msg)
                raise LabelError(msg)


        return parsed_dict

    @staticmethod
    def _parse_optim_params(dict_optim: Dict, dict_ET: Dict,
                            dict_decay: Dict, dict_states: Dict) -> List:
        '''Parse the optional list of parameters to optimize.
           Some of the params are ET, other are branching ratios'''
           # TODO: this only works for activator branching ratios

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

    #    sensitizer_labels = dict_states['sensitizer_states_labels']
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
                    raise ValueError('Unrecognized branching ratio process.')
            else:
                B_pos_value_A = []
        except ValueError as err:
            msg = 'Wrong labels in optimization: processes! ' + str(err.args[0])
            logger.error(msg)
            raise LabelError(msg) from err

        return list(set_ET_params) + B_pos_value_A

    @staticmethod
    def _parse_simulation_params(user_settings: Dict = None) -> Dict:
        '''Parse the optional simulation parameters
            If some are not given, the default values are used
        '''
        # use the file located where the package is installed
        _log_config_file = 'settings.cfg'
        # resource_string opens the file and gets it as a string. Works inside .egg too
        _log_config_location = resource_string(__name__, os.path.join('config', _log_config_file))
        default_settings = Loader().load_settings_file((_log_config_location), direct_file=True)
        default_settings = default_settings['simulation_params']

        if user_settings is None:
            return default_settings

        new_settings = dict(default_settings)
        new_settings.update(user_settings)

        return new_settings

    @staticmethod
    def _parse_power_dependence(user_list: List = None) -> List[float]:
        '''Parses the power dependence list with the minimum, maximum and number of points.'''

        min_power = user_list[0]
        max_power = user_list[1]
        num_points = int(user_list[2])

        power_list = np.logspace(np.log10(min_power), np.log10(max_power), num_points)

        return list(power_list)

    @staticmethod
    def _parse_conc_dependence(user_list: Tuple[List[float], List[float]] = None
                              ) -> List[Tuple[float, float]]:
        '''Parses the concentration dependence list with
            the minimum, maximum and number of points.'''

        # get the lists of concentrations from the user
        # if empty, set to 0.0
        S_conc_l = user_list[0]
        A_conc_l = user_list[1]

        # make a regular grid of values
        conc_grid = np.meshgrid(S_conc_l, A_conc_l)
        conc_grid[0].shape = (conc_grid[0].size, 1)
        conc_grid[1].shape = (conc_grid[0].size, 1)
        conc_list = [((float(a), float(b))) for a, b in zip(conc_grid[0], conc_grid[1])]

        return conc_list

    def modify_param_value(self, process: str, new_value: float) -> None:
        '''Change the value of the process.'''
        if isinstance(process, str):
            self._modify_ET_param_value(process, new_value)
        elif isinstance(process, tuple):
            self._modify_branching_ratio_value(process, new_value)

    def _modify_ET_param_value(self, process: str, new_strength: float) -> None:
        '''Modify a ET parameter'''
        self.energy_transfer[process]['value'] = new_strength

    def _modify_branching_ratio_value(self, process: Tuple[int, int], new_value: float) -> None:
        '''Modify a branching ratio param.'''
        list_tups = self['decay']['B_pos_value_A']
        for num, tup in enumerate(list_tups):
            if tup[:2] == process:
                old_tup = self['decay']['B_pos_value_A'][num]
                self['decay']['B_pos_value_A'][num] = (*old_tup[:2], new_value)

    def get_ET_param_value(self, process: str, average: bool = False) -> float:
        '''Get a ET parameter value.
            Return the average value if it exists.
        '''
        if average:
            return self.energy_transfer[process].get('value_avg',
                                                     self.energy_transfer[process]['value'])
        else:
            return self.energy_transfer[process]['value']

    def get_branching_ratio_value(self, process: Tuple[int, int]) -> float:
        '''Gets a branching ratio value.'''
        list_tups = self['decay']['B_pos_value_A']
        for num, tup in enumerate(list_tups):
            if tup[:2] == process:
                return self['decay']['B_pos_value_A'][num][2]
        raise ValueError('Branching ratio ({}) not found'.format(process))

    def load(self, filename: str) -> None:
        ''' Load filename and extract the settings for the simulations
            If mandatory values are missing, errors are logged
            and exceptions are raised
            Warnings are logged if extra settings are found
        '''
        logger = logging.getLogger(__name__)
        logger.info('Reading settings file (%s)...', filename)

        # load file into config_cte dictionary.
        # the function checks that the file exists and that there are no errors
        config_cte = Loader().load_settings_file(filename)

        # check version
        if 'version' not in config_cte or config_cte['version'] != 1:
            logger.error('Error in configuration file (%s)!', filename)
            logger.error('Version number must be 1!')
            raise ConfigError('Version number must be 1!')
        del config_cte['version']

        # store original configuration file
        with open(filename, 'rt') as file:
            self.config_file = file.read()

        parsed_settings = self.parse_all_values(configs.settings, config_cte)

        # LATTICE
        # parse lattice params
        self.lattice = parsed_settings['lattice']

        # NUMBER OF STATES
        self.states = parsed_settings['states']
        self.states['sensitizer_states'] = len(self.states['sensitizer_states_labels'])
        self.states['activator_states'] = len(self.states['activator_states_labels'])

        # EXCITATIONS
        self.excitations = self._parse_excitations(parsed_settings['states'],
                                                   parsed_settings['excitations'])

        # DECAY RATES
        pos_value_S, pos_value_A = self._parse_decay_rates(parsed_settings)
        self.decay['pos_value_S'] = pos_value_S
        self.decay['pos_value_A'] = pos_value_A

        # BRANCHING RATIOS (from 0 to 1)
        B_pos_value_S, B_pos_value_A = self._parse_branching_ratios(parsed_settings)
        self.decay['B_pos_value_S'] = B_pos_value_S
        self.decay['B_pos_value_A'] = B_pos_value_A

        # ET PROCESSES.
        # not mandatory -> check
        if 'energy_transfer' in parsed_settings:
            self.energy_transfer = self._parse_ET(parsed_settings)
        else:
            self.energy_transfer = OrderedDict()

        # OPTIMIZATION
        # not mandatory -> check
        if 'optimization' in parsed_settings:
            self.optimization = self._parse_optimization(parsed_settings['optimization'],
                                                         self.energy_transfer, self.decay,
                                                         self.states, self.excitations)


        # SIMULATION PARAMETERS
        # not mandatory -> check
        self.simulation_params = self._parse_simulation_params(parsed_settings.get('simulation_params',
                                                                              None))

        # POWER DEPENDENCE LIST
        # not mandatory -> check
        if 'power_dependence' in parsed_settings:
            self.power_dependence = self._parse_power_dependence(parsed_settings['power_dependence'])

        # CONCENTRATION DEPENDENCE LIST
        # not mandatory -> check
        if 'concentration_dependence' in parsed_settings:
            self.conc_dependence =\
                self._parse_conc_dependence(parsed_settings['concentration_dependence'])

        # log read and parsed settings
        # use pretty print
        logger.debug('Settings dump:')
        logger.debug('File dict (config_cte):')
        logger.debug(pprint.pformat(config_cte))
        logger.debug('Parsed dict (cte):')
        logger.debug(repr(self))

        logger.info('Settings loaded!')


class Loader():
    '''Load a settings file'''

    def __init__(self) -> None:
        '''Init variables'''
        self.file_dict = {}  # type: Dict

    def load_settings_file(self, filename: Union[str, bytes], file_format: str = 'yaml',
                           direct_file: bool = False) -> Dict:
        '''Loads a settings file with the given format (only YAML supported at this time).
            If direct_file=True, filename is actually a file and not a path to a file.
            If the file doesn't exist ir it's emtpy, raise ConfigError.'''
        logger = logging.getLogger(__name__)

        if file_format.lower() == 'yaml':
            self.file_dict = self._load_yaml_file(filename, direct_file)
        else:
            return NotImplemented

        if self.file_dict is None or not isinstance(self.file_dict, dict):
            msg = 'The settings file is empty or otherwise invalid ({})!'.format(filename)
            logger.error(msg)
            raise ConfigError(msg)

        return self.file_dict

    def _load_yaml_file(self, filename: Union[str, bytes], direct_file: bool = False) -> Dict:
        '''Open a yaml filename and loads it into a dictionary
            ConfigError exceptions are raised if the file doesn't exist or is invalid.
            If direct_file=True, filename is actually a file and not a path to one
        '''
        logger = logging.getLogger(__name__)

        file_dict = {}  # type: Dict
        try:
            if not direct_file:
                with open(filename) as file:
                    # load data as ordered dictionaries so the ET processes are in the right order
                    file_dict = self._ordered_load(file, yaml.SafeLoader)
            else:
                file_dict = self._ordered_load(filename, yaml.SafeLoader)
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

        return file_dict

    @staticmethod
    def _ordered_load(stream, YAML_Loader=yaml.Loader, object_pairs_hook=OrderedDict):  # type: ignore
        '''Load data as ordered dictionaries so the ET processes are in the right order
        # not necessary any more, but still used
            http://stackoverflow.com/a/21912744
        '''

        class OrderedLoader(YAML_Loader):
            '''Load the yaml file use an OderedDict'''
            pass

        def no_duplicates_constructor(loader, node, deep=False):  # type: ignore
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
    settings = Settings()
    settings.load(filename)
    return settings


#if __name__ == "__main__":
#    import simetuc.settings as settings
#    cte = settings.load('config_file.cfg')
