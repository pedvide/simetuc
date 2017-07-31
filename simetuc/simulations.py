# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:22:41 2015

@author: Villanueva
"""

import time
import csv
import logging
import warnings
import os
from typing import List, Tuple, Iterator, Sequence, cast
import copy

import h5py
import ruamel.yaml as yaml

import numpy as np

import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy import stats

# nice progress bar
from tqdm import tqdm

import simetuc.precalculate as precalculate
import simetuc.odesolver as odesolver
#import simetuc.odesolver_assimulo as odesolver  # warning: it's slower!
import simetuc.plotter as plotter
from simetuc.util import Conc
from simetuc.util import cached_property, log_exceptions_warnings, disable_loggers
import simetuc.settings as settings
from simetuc.settings import Settings


def exp_to_10(float_number: float) -> str:
    '''Convert a float to str using power of ten notation instead of exp.
        ie: 3e5 -> 3×10^5'''
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number/10**exponent
    mantissa_format = str(mantissa)[0:3]
    return "{}×10^{}".format(mantissa_format, str(int(exponent)))


class Solution():
    '''Base class for solutions of rate equation problems'''

    def __init__(self, t_sol: np.array, y_sol: np.array,
                 index_S_i: List[int], index_A_j: List[int],
                 cte: Settings, average: bool = False) -> None:
        # simulation time
        self.t_sol = t_sol
        # population of each state of each ion
        self.y_sol = y_sol
        # list of average population for each state
        self._list_avg_data = np.array([])
        # settings
        self.cte = copy.deepcopy(cte)
        # sensitizer and activator indices of their ground states
        self.index_S_i = index_S_i
        self.index_A_j = index_A_j
        # state labels
        self._state_labels = []  # type: List[str]

        # average or microscopic rate equations?
        self.average = average

        # The first is the sim color, the second the exp data color.
        self.cte['colors'] = 'bk' if average else 'rk'

        # prefix for the name of the saved files
        self._prefix = 'solution'

    def __bool__(self) -> bool:
        '''Instance is True if all its data structures have been filled out'''
        return (self.t_sol.size != 0 and self.y_sol.size != 0 and bool(self.cte) and
                len(self.index_S_i) != 0 and len(self.index_A_j) != 0)

    def __eq__(self, other: object) -> bool:
        '''Two solutions are equal if all its vars are equal or numerically close'''
        if not isinstance(other, Solution):
            return NotImplemented
        return (self.y_sol.shape == other.y_sol.shape and np.allclose(self.t_sol, other.t_sol) and
                np.allclose(self.y_sol, other.y_sol) and
                self.cte == other.cte and self.index_S_i == other.index_S_i and
                self.index_A_j == other.index_A_j)

    def __ne__(self, other: object) -> bool:
        '''Define a non-equality test'''
        if not isinstance(other, Solution):
            return NotImplemented
        return not self == other

    def __repr__(self) -> str:
        '''Representation of a solution.'''
        return '{}(num_states={}, {}, power_dens={:.1e})'.format(self.__class__.__name__,
                                                                 self.y_sol.shape[1],
                                                                 self.concentration,
                                                                 self.power_dens)

    def _calculate_avg_populations(self) -> List[np.array]:
        '''Returs the average populations of each state. First S then A states.'''
        cte = self.cte
        index_S_i = self.index_S_i
        index_A_j = self.index_A_j
        y_sol = self.y_sol

        # average population of the ground and excited states of S
        if cte.ions['sensitizers'] is not 0:
            sim_data_Sensitizer = []
            for state in range(cte.states['sensitizer_states']):
                population = np.sum([y_sol[:, index_S_i[i]+state]
                                     for i in range(cte.ions['total'])
                                     if index_S_i[i] != -1], 0)/cte.ions['sensitizers']
                sim_data_Sensitizer.append(population.clip(0).reshape((y_sol.shape[0],)))
        else:
            sim_data_Sensitizer = cte.states['sensitizer_states']*[np.zeros((y_sol.shape[0],))]
        # average population of the ground and excited states of A
        if cte.ions['activators'] is not 0:
            sim_data_Activator = []
            for state in range(cte.states['activator_states']):
                population = np.sum([y_sol[:, index_A_j[i]+state]
                                     for i in range(cte.ions['total'])
                                     if index_A_j[i] != -1], 0)/cte.ions['activators']
                sim_data_Activator.append(population.clip(0).reshape((y_sol.shape[0],)))
        else:
            sim_data_Activator = cte.states['activator_states']*[np.zeros((y_sol.shape[0],))]

        return sim_data_Sensitizer + sim_data_Activator

    def _get_ion_state_labels(self) -> List[str]:
        '''Returns a list of ion_state labels'''
        cte = self.cte
        sensitizer_labels = [cte.states['sensitizer_ion_label'] + '_' + s
                             for s in cte.states['sensitizer_states_labels']]
        activator_labels = [cte.states['activator_ion_label'] + '_' + s
                            for s in cte.states['activator_states_labels']]
        state_labels = sensitizer_labels + activator_labels
        return state_labels

    def save_file_full_name(self, prefix: str = None) -> str:  # pragma: no cover
        '''Return the full name to save a file (without extention or prefix).'''
        lattice_name = self.cte.lattice['name']
        path = os.path.join('results', lattice_name)
        os.makedirs(path, exist_ok=True)
        filename = prefix + '_' + '{}uc_{}S_{}A'.format(int(self.cte.lattice['N_uc']),
                                                        float(self.concentration.S_conc),
                                                        float(self.concentration.A_conc))
        return os.path.join(path, filename)

    @cached_property
    def state_labels(self) -> List[str]:
        '''List of ion_state labels'''
        return self._get_ion_state_labels()

    @cached_property
    def list_avg_data(self) -> List[np.array]:
        '''List of average populations for each state in the solution'''
        return self._calculate_avg_populations()

    @cached_property
    @log_exceptions_warnings
    def power_dens(self) -> float:
        '''Return the power density used to obtain this solution.'''
        for excitation in self.cte.excitations.keys():  # pragma: no branch
            if self.cte.excitations[excitation][0].active:
                return self.cte.excitations[excitation][0].power_dens
        raise AttributeError('This Solution has no power_dens attribute!')

    @cached_property
    def concentration(self) -> Conc:
        '''Return the tuple (sensitizer, activator) concentration used to obtain this solution.'''
        return Conc(self.cte.lattice['S_conc'], self.cte.lattice['A_conc'])

    def _plot_avg(self) -> None:
        '''Plot the average simulated data (list_avg_data).
            Override to plot other lists of averaged data or experimental data.
        '''
        title = '{}: {}% {}, {}% {}. P={} W/cm²'.format(self.cte.lattice['name'],
                                        self.concentration.S_conc, self.cte.states['sensitizer_ion_label'],
                                        self.concentration.A_conc, self.cte.states['activator_ion_label'],
                                        exp_to_10(self.power_dens))
        index_GS_S = 0
        index_GS_A = self.cte.states['sensitizer_states']
        list_data = self.list_avg_data[index_GS_S+1:index_GS_A-1] + self.list_avg_data[index_GS_A+1:]
        list_labels = self.state_labels[index_GS_S+1:index_GS_A-1] + self.state_labels[index_GS_A+1:]
        plotter.plot_avg_decay_data(self.t_sol, list_data,
                                    state_labels=list_labels, colors=self.cte['colors'],
                                    title=title)

    def _plot_state(self, state: int) -> None:
        '''Plot all decays of a state as a function of time.'''
        if state < self.cte.states['sensitizer_states']:
            indices = self.index_S_i
            label = self.state_labels[state]
        else:
            indices = self.index_A_j
            label = self.state_labels[state]
            state -= self.cte.states['sensitizer_states']
        populations = np.array([self.y_sol[:, index+state]
                                for index in indices if index != -1])
        plotter.plot_state_decay_data(self.t_sol, populations.T,
                                      state_label=label, atol=1e-18)

    @log_exceptions_warnings
    def plot(self, state: int = None) -> None:
        '''Plot the soltion of a problem.
            If state is given, the population of only that state for all ions
            is shown along with the average.
        '''
        if self.cte['no_plot']:
            msg = 'A plot was requested, but no_plot setting is set'
            warnings.warn(msg, plotter.PlotWarning)
            return

        if state is None:
            self._plot_avg()
        elif 0 <= state < len(self.state_labels):
            self._plot_state(state)
        else:
            msg = 'The selected state does not exist!'
            raise ValueError(msg)

    def save(self, full_path: str = None) -> None:
        '''Save data to disk as a HDF5 file'''
        logger = logging.getLogger(__name__)
        if full_path is None:  # pragma: no cover
            full_path = self.save_file_full_name(self._prefix) + '.hdf5'
        logger.info('Saving solution to {}.'.format(full_path))
        with h5py.File(full_path, 'w') as file:
            file.create_dataset("t_sol", data=self.t_sol, compression='gzip')
            file.create_dataset("y_sol", data=self.y_sol, compression='gzip')
            file.create_dataset("y_sol_avg", data=self.list_avg_data, compression='gzip')
            file.create_dataset("index_S_i", data=self.index_S_i, compression='gzip')
            file.create_dataset("index_A_j", data=self.index_A_j, compression='gzip')
            file.attrs['config_file'] = self.cte['config_file']
            # serialize cte
            file.attrs['cte'] = yaml.dump(self.cte.settings)

    def save_txt(self, full_path: str = None, mode: str = 'wt') -> None:
        '''Save the settings, the time and the average populations to disk as a textfile'''
        logger = logging.getLogger(__name__)
        if full_path is None:  # pragma: no cover
            full_path = self.save_file_full_name(self._prefix) + '.txt'
        logger.info('Saving solution as text to {}.'.format(full_path))
        # print cte
        with open(full_path, mode) as csvfile:
            csvfile.write('Settings:\n')
            csvfile.write(self.cte['config_file'])
            csvfile.write('\n\n\nData:\n')
        # print t_sol and avg sim data
        header = ('time (s)      ' +
                  '         '.join(self.cte.states['sensitizer_states_labels']) +
                  '         ' +
                  '         '.join(self.cte.states['activator_states_labels']))
        with open(full_path, 'ab') as csvfile:
            np.savetxt(csvfile, np.transpose([self.t_sol, *self.list_avg_data]),
                       fmt='%1.4e', delimiter=', ', newline='\r\n', header=header)

    @classmethod
    @log_exceptions_warnings
    def load(cls, full_path: str) -> 'Solution':
        '''Load data from a HDF5 file'''
        logger = logging.getLogger(__name__)
        logger.info('Loading solution from {}.'.format(full_path))
        try:
            with h5py.File(full_path, 'r') as file:
                t_sol = np.array(file['t_sol'])
                y_sol = np.array(file['y_sol'])
                index_S_i = list(file['index_S_i'])
                index_A_j = list(file['index_A_j'])
                # deserialze cte
                cte_dict = yaml.load(file.attrs['cte'], Loader=yaml.Loader)
                cte = settings.load_from_text(cte_dict['config_file'])
                for key, value in cte_dict.items():
                    cte[key] = value

            return cls(t_sol, y_sol, index_S_i, index_A_j, cte)
        except OSError as err:
            msg = 'File not found! ({})'.format(full_path)
            raise OSError(msg) from err


class SteadyStateSolution(Solution):
    '''Class representing the solution to a steady state problem'''

    def __init__(self, t_sol: np.array, y_sol: np.array,
                 index_S_i: List[int], index_A_j: List[int],
                 cte: Settings, average: bool = False) -> None:
        super(SteadyStateSolution, self).__init__(t_sol, y_sol, index_S_i, index_A_j,
                                                  cte, average=average)
        self._final_populations = np.array([])

        # prefix for the name of the saved files
        self._prefix = 'steady_state'

    def _calculate_final_populations(self) -> List[float]:
        '''Calculate the final population for all states after a steady state simulation'''
        return [curve[-1] for curve in self.list_avg_data]

    @cached_property
    def steady_state_populations(self) -> List[float]:
        '''List of final steady-state populations for each state in the solution'''
        # if empty, calculate
        return self._calculate_final_populations()

    def log_populations(self) -> None:
        '''Log the steady state populations'''
        logger = logging.getLogger(__name__)
        # get ion_state labels
        state_labels = self.state_labels

        logger.info('Steady state populations: ')
        for (label, population) in zip(state_labels, self.steady_state_populations):
            logger.info('%s: %.4e', label, population)


class DynamicsSolution(Solution):
    '''Class representing the solution to a dynamics problem.
        It handles the loading of experimetal decay data and calculates errors.'''

    def __init__(self, t_sol: np.array, y_sol: np.array,
                 index_S_i: List[int], index_A_j: List[int],
                 cte: Settings, average: bool = False) -> None:
        super(DynamicsSolution, self).__init__(t_sol, y_sol, index_S_i, index_A_j,
                                               cte, average=average)

        self._list_exp_data = []  # type: List[np.array]
        self._list_avg_data_ofs = []  # type: List[np.array]
        self._list_binned_data = []  # type: List[np.array]
        self._list_interp_data = []  # type: List[np.array]

        self._total_error = None  # type: float
        self._errors = np.array([])

        # prefix for the name of the saved files
        self._prefix = 'dynamics'

    #@profile
    @staticmethod
    @log_exceptions_warnings
    def _load_exp_data(filename: str, lattice_name: str, filter_window: int = 35) -> np.array:
        '''Load the experimental data from the expData/lattice_name folder.
           Two columns of numbers: first is time (seconds), second intensity
        '''
        logger = logging.getLogger(__name__)

        # use absolute path from here
        if os.path.isdir("expData"):  # pragma: no cover
            path = os.path.join('expData', lattice_name, filename)
        elif os.path.isdir(os.path.join('simetuc', 'expData')):
            path = os.path.join(os.path.join('simetuc', 'expData'), lattice_name, filename)
        else:
            return None

        logger.debug('Trying to read experimental data file: %s', filename)

        delimiters = [' ', '\t', ',', ';']
        for delim in delimiters:
            try:
                with open(path, 'rt') as file:
                    try:  # TODO: get a better way to read data, PANDAS?
                        #  data = np.loadtxt(path, usecols=(0, 1)) # 10x slower
                        csv_data = csv.reader(file, delimiter=delim)
                        # ignore emtpy lines and comments
                        data = np.array([row for row in csv_data
                                         if len(row) == 2 and not row[0].startswith('#')], dtype=np.float64)

                        # if data isn't right, retry with a different delimiter
                        if (not isinstance(data, np.ndarray) or
                                len(data.shape) != 2 or data.shape[1] != 2):
                            raise ValueError
                    except ValueError:  # pragma: no cover
                        continue
                    else:  # success
                        break
            except FileNotFoundError:
                # exp data doesn't exist. not a problem.
                logger.debug('File not found.')
                return None

        # make sure data is valid
        if not isinstance(data, np.ndarray) or len(data.shape) != 2 or data.shape[1] != 2:
            warnings.warn('Invalid experimental data in' +
                          ' file "{}", it will be ignored.'.format(filename))
            return None

        logger.debug('Experimental data succesfully read.')

        # smooth the data to get an "average" of the maximum
        smooth_data = signal.savgol_filter(data[:, 1], filter_window, 5, mode='nearest')
        smooth_data = smooth_data.clip(min=0)

        # average maximum
        max_point = np.where(smooth_data == np.max(smooth_data))[0][0]
        # average over 0.5% of points around max_point
        delta = int(0.005*len(smooth_data))
        delta = delta if delta > 1 else 1  # at least 1 point
        if max_point == 0:
            avg_max = np.max(smooth_data)
        elif max_point > delta//2:
            avg_max = np.mean(smooth_data[max_point-delta//2:max_point+delta//2])
        else:
            avg_max = np.mean(smooth_data[:max_point+delta//2])
        # normalize data
        data[:, 1] = (data[:, 1]-np.min(smooth_data))/(avg_max-np.min(smooth_data))
        # set negative values to zero
        data = data.clip(min=0)
        return data

    #@profile
    @staticmethod
    def _correct_background(exp_data: np.array, sim_data: np.array,
                            offset_points: int = 50) -> np.array:
        '''Add the experimental background to the simulated data.
           Returns the same simulated data if there's no exp_data
            expData is already normalized when loaded.
        '''
        if not np.any(exp_data):  # if there's no experimental data, don't do anything
            return sim_data
        if not np.any(sim_data):  # pragma: no cover
            return None

        # eliminate zeros before calculating background
        count_data = exp_data[:,1]
        non_zero_exp_data = count_data[count_data>0]
        last_points = non_zero_exp_data[-offset_points:]  # get last 50 points
        if np.any(last_points > 0):
            offset = np.mean(last_points[last_points > 0])*np.max(sim_data)
        else:
            offset = 0

        if np.isnan(offset):  # pragma: no cover
            sim_data_ofs = sim_data
        else:  # offset-correct simulated data
            sim_data_ofs = sim_data+offset

        return sim_data_ofs

    #@profile
    @staticmethod
    def _interpolate_sim_data(exp_data: np.array, sim_data: np.array,
                              t_sol: np.array, t_interp: np.array = None) -> List[np.array]:
        '''Interpolate simulated corrected data to exp data points.'''

        if not np.any(exp_data):  # if there's no experimental data, don't do anything
            return sim_data
        if not np.any(sim_data):  # pragma: no cover
            return None

        if t_interp is None:
            t_interp = exp_data[:, 0]

        # create function to interpolate
        iterp_sim_func = interpolate.interp1d(t_sol, sim_data, fill_value='extrapolate')
        # interpolate them to the experimental data times (or the requested times)
        iterp_sim_data = iterp_sim_func(t_interp)

        return iterp_sim_data

    @staticmethod
    def _bin_sim_data(exp_data: np.array, sim_data: np.array) -> List[np.array]:
        '''Bin simulated data to the same bin centers and width that the experimental data.
           Add all the counts in a bin.'''

        if not np.any(exp_data):  # if there's no experimental data, don't do anything
            return sim_data
        if not np.any(sim_data):  # pragma: no cover
            return None

        exp_time = exp_data[:, 0]
        bin_time = np.linspace(exp_time[0], exp_time[-1], 10*len(exp_time))
        bin_sums, _, _ = stats.binned_statistic(bin_time, sim_data,
                                                statistic='sum', bins=len(exp_time))
#        print(bin_sums, bin_edges, binnumber)

        return bin_sums

    #@profile
    def _calc_errors(self) -> np.array:
        '''Calculate root-square-deviation between experiment and simulation.'''
        # get interpolated simulated data
        list_sim_data = self.list_interp_data

        # calculate the relative mean square deviation
        # error = 1/mean(y)*sqrt(sum( (y-yexp)^2 )/N )
        rmdevs = [(sim-exp[:, 1]*np.max(sim))**2
                  if (exp is not None) and (sim is not None) else 0
                  for (sim, exp) in zip(list_sim_data, self.list_exp_data)]
        errors = [1/np.mean(sim)*np.sqrt(1/len(sim)*np.sum(rmdev))
                  if rmdev is not 0 else 0
                  for (rmdev, sim) in zip(rmdevs, list_sim_data)]
        errors = np.array(errors)

        return errors

    def _load_decay_data(self) -> List[np.array]:
        '''Load and return the decay experimental data.'''
        # get filenames from the ion_state labels, excitation and concentrations
        state_labels = self.state_labels
        active_exc_labels = [label for label, excitation in self.cte.excitations.items()
                             if excitation[0].active]
        exc_label = '_'.join(active_exc_labels)
        S_conc = str(float(self.cte.lattice['S_conc']))
        S_label = self.cte.states['sensitizer_ion_label']
        A_conc = str(float(self.cte.lattice['A_conc']))
        A_label = self.cte.states['activator_ion_label']
        conc_str = '_' + S_conc + S_label + '_' + A_conc + A_label
        exp_data_filenames = ['decay_' + label + '_exc_' + exc_label + conc_str + '.txt'
                              for label in state_labels]

        # if exp data doesn't exist, it's set to zero inside the function
        _list_exp_data = [self._load_exp_data(filename, self.cte.lattice['name'])
                          for filename in exp_data_filenames]

        return _list_exp_data

    def _plot_avg(self) -> None:
        '''Overrides the Solution method to plot
            the average offset-corrected simulated data (list_avg_data) and experimental data.
        '''
        title = '{}: {}% {}, {}% {}. P={} W/cm²'.format(self.cte.lattice['name'],
                                        self.concentration.S_conc, self.cte.states['sensitizer_ion_label'],
                                        self.concentration.A_conc, self.cte.states['activator_ion_label'],
                                        exp_to_10(self.power_dens))

        index_GS_S = 0
        index_GS_A = self.cte.states['sensitizer_states']

        list_t_sim = [data[:, 0] if data is not None else self.t_sol for data in self.list_exp_data]
        list_t_sim = list_t_sim[index_GS_S+1:index_GS_A-1] + list_t_sim[index_GS_A+1:]

        list_data = self.list_binned_data[index_GS_S+1:index_GS_A-1] + self.list_binned_data[index_GS_A+1:]
        list_exp_data = self.list_exp_data[index_GS_S+1:index_GS_A-1] + self.list_exp_data[index_GS_A+1:]
        list_labels = self.state_labels[index_GS_S+1:index_GS_A-1] + self.state_labels[index_GS_A+1:]

        plotter.plot_avg_decay_data(list_t_sim, list_data,
                                    state_labels=list_labels,
                                    list_exp_data=list_exp_data,
                                    colors=self.cte['colors'], title=title)

#    def save(self, full_path: str = None) -> None:
#        '''Save data to disk as a HDF5 file'''
#        # save common data
#        super(DynamicsSolution, self).save(full_path)
#
#        if full_path is None:  # pragma: no cover
#            full_path = self.save_file_full_name(self._prefix) + '.hdf5'
#
#        # save exp data
#        with h5py.File(full_path, 'a') as file:
#            file.create_dataset("list_exp_data", data=self.list_exp_data, compression='gzip')


#    def save(self, full_path: str = None) -> None:
#        '''Save data to disk as a HDF5 file'''
#        # save common data
#        super(DynamicsSolution, self).save(full_path)
#
#        if full_path is None:  # pragma: no cover
#            full_path = self.save_file_full_name(self._prefix) + '.hdf5'
#
#        # save exp data
#        with h5py.File(full_path, 'a') as file:
#            file.create_dataset("list_exp_data", data=self.list_exp_data, compression='gzip')

    def log_errors(self) -> None:
        '''Log errors'''
        logger = logging.getLogger(__name__)

        # log errors
        logger.info('State errors: ')
        for (label, error) in zip(self.state_labels, self.errors):
            logger.info('%s: %.4e', label, error)
        logger.info('Total error: %.4e', self.total_error)

    @cached_property
    def errors(self) -> np.array:
        '''List of root-square-deviation between experiment and simulation
            for each state in the solution
        '''
        return self._calc_errors()

    @cached_property
    def total_error(self) -> float:
        '''Total root-square-deviation between experiment and simulation'''
        if np.any(self.errors):
            total_error = np.sqrt(np.sum(self.errors**2))
        else:
            total_error = 0
        return total_error

    @cached_property
    @log_exceptions_warnings
    def list_interp_data(self) -> List[np.array]:
        '''List of offset-corrected (due to experimental background) average populations
            for each state in the solution
        '''
        return [DynamicsSolution._interpolate_sim_data(expData, simData, self.t_sol)
                for expData, simData in zip(self.list_exp_data, self.list_avg_data_ofs)]

    @cached_property
    def list_avg_data_ofs(self) -> List[np.array]:
        '''List of offset-corrected (due to experimental background) average populations
            for each state in the solution
        '''
        return [DynamicsSolution._correct_background(expData, simData)
                for expData, simData in zip(self.list_exp_data, self.list_avg_data)]

    @cached_property
    def list_binned_data(self) -> List[np.array]:
        '''List of offset-corrected (due to experimental background) average populations
            for each state in the solution
        '''
        # if empty, calculate
        def bin_time(expdata: np.array) -> np.array:
            '''Return a vector with the same expdata time but with 10x the number of points.'''
            if expdata is not None:
                return np.linspace(expdata[0, 0], expdata[-1, 0], 10*len(expdata[:, 0]))

        list_interp_data = [DynamicsSolution._interpolate_sim_data(expData, simData, self.t_sol,
                                                                   bin_time(expData))
                            for expData, simData in zip(self.list_exp_data,
                                                        self.list_avg_data_ofs)]

        list_binned_data = [DynamicsSolution._bin_sim_data(exp_data, sim_data)
                            for exp_data, sim_data in zip(self.list_exp_data, list_interp_data)]
        return list_binned_data

    @cached_property
    def list_exp_data(self) -> List[np.array]:
        '''List of ofset-corrected average populations for each state in the solution'''
        # if empty, calculate
        if not self._list_exp_data:
            self._list_exp_data = self._load_decay_data()
        return self._list_exp_data


class SolutionList(Sequence[Solution]):
    '''Base class for a list of solutions for problems like power or concentration dependence.'''
    def __init__(self) -> None:
        self.solution_list = []  # type: List[Solution]
        # constructor of the underliying class that the list stores.
        # the load method will create instances of this type
        self._items_class = Solution
        self._prefix = 'solutionlist'
        self.dynamics = False

    def __bool__(self) -> bool:
        '''Instance is True if its list is not emtpy.'''
        return len(self.solution_list) != 0

    def __eq__(self, other: object) -> bool:
        '''Two solutions are equal if all their solutions are equal.'''
        if not isinstance(other, SolutionList):
            return NotImplemented
        return self.solution_list == other.solution_list

    # __iter__, __len__ and __getitem__ implement all requirements for a Sequence,
    # which is also Sized and Iterable
    def __iter__(self) -> Iterator:
        '''Make the class iterable by returning a iterator over the solution_list.'''
        return iter(self.solution_list)

    def __len__(self) -> int:
        '''Return the length of the solution_list.'''
        return len(self.solution_list)

    def __getitem__(self, index: int) -> Solution:  # type: ignore
        '''Implements solution[number].'''
        if 0 > index > len(self.solution_list):
            raise IndexError
        return self.solution_list[index]

    def __repr__(self) -> str:
        '''Representation of a solution list.'''
        concs = [sol.concentration for sol in self]
        powers = [sol.power_dens for sol in self]
        return '{}(num_solutions={}, concs={}, power_dens={})'.format(self.__class__.__name__,
                                                                      len(self),
                                                                      concs,
                                                                      powers)

    def add_solutions(self, sol_list: List[Solution]) -> None:
        '''Add a list of solutions.'''
        self.solution_list = list(sol_list)

    def save(self, full_path: str = None) -> None:
        '''Save all data from all solutions in a HDF5 file'''
        if full_path is None:  # pragma: no cover
            full_path = self[0].save_file_full_name(self._prefix) + '.hdf5'

        with h5py.File(full_path, 'w') as file:
            for num, sol in enumerate(self):
                group = file.create_group(str(num))
                group.create_dataset("t_sol", data=sol.t_sol, compression='gzip')
                group.create_dataset("y_sol", data=sol.y_sol, compression='gzip')
                group.create_dataset("y_sol_avg", data=sol.list_avg_data, compression='gzip')
                group.create_dataset("index_S_i", data=sol.index_S_i, compression='gzip')
                group.create_dataset("index_A_j", data=sol.index_A_j, compression='gzip')
                # serialze cte as text and store it as an attribute
                group.attrs['cte'] = yaml.dump(sol.cte.settings)
                file.attrs['config_file'] = sol.cte['config_file']
            if hasattr(self, 'dynamics'):
                file.attrs['dynamics'] = self.dynamics

    @classmethod
    @log_exceptions_warnings
    def load(cls, full_path: str) -> 'SolutionList':
        '''Load data from a HDF5 file'''
        solutions = []
        try:
            with h5py.File(full_path, 'r') as file:
                if file.attrs.get('dynamics', None) is not None:
                    dynamics = file.attrs['dynamics']
                else:
                    dynamics = False
                sol_list = cls()
                sol_list.dynamics = dynamics
                for group_num in file:
                    group = file[group_num]
                    # deserialze cte
                    cte_dict = yaml.load(group.attrs['cte'], Loader=yaml.Loader)
                    cte = settings.load_from_text(cte_dict['config_file'])
                    for key, value in cte_dict.items():
                        cte[key] = value
                    # create appropiate object
                    sol = sol_list._items_class(np.array(group['t_sol']), np.array(group['y_sol']),
                                                list(group['index_S_i']), list(group['index_A_j']),
                                                cte)
                    solutions.append(sol)
        except OSError as err:
            msg = 'File not found! ({})'.format(full_path)
            raise OSError(msg) from err
        sol_list.add_solutions(solutions)
        return sol_list

    def save_txt(self, full_path: str = None, mode: str = 'w') -> None:
        '''Save the settings, the time and the average populations to disk as a textfile'''
        if full_path is None:  # pragma: no cover
            full_path = self[0].save_file_full_name('solutionlist') + '.txt'
        with open(full_path, mode+'t') as csvfile:
            csvfile.write('Solution list:\n')
        for sol in self:
            sol.save_txt(full_path, 'at')

    def plot(self) -> None:
        '''Interface of plot.
        '''
        raise NotImplementedError


class PowerDependenceSolution(SolutionList):
    '''Solution to a power dependence simulation'''
    def __init__(self) -> None:
        '''All solutions are SteadStateSolution'''
        super(PowerDependenceSolution, self).__init__()
        # constructor of the underliying class that the list stores
        # the load method will create instances of this type
        self._items_class = SteadyStateSolution
        self._prefix = 'pow_dep'

    def __repr__(self) -> str:
        '''Representation of a power dependence list.'''
        conc = self[0].concentration
        powers = [sol.power_dens for sol in self]
        return '{}(num_solutions={}, conc={}, power_dens={})'.format(self.__class__.__name__,
                                                                     len(self),
                                                                     conc,
                                                                     powers)
    @log_exceptions_warnings
    def plot(self) -> None:
        '''Plot the power dependence of the emission for all states.
        '''
        if len(self) == 0:  # nothing to plot
            logger = logging.getLogger(__name__)
            msg = 'Nothing to plot! The power_dependence list is emtpy!'
            warnings.warn(msg, plotter.PlotWarning)
            return

        if self[0].cte['no_plot']:
            logger = logging.getLogger(__name__)
            msg = 'A plot was requested, but no_plot setting is set'
            logger.warning(msg)
            warnings.warn(msg, plotter.PlotWarning)
            return

        sim_data_arr = np.array([np.array(sol.steady_state_populations)
                                 for sol in self])
        power_dens_arr = np.array([sol.power_dens for sol in self])
        state_labels = self[0].state_labels

        plotter.plot_power_dependence(sim_data_arr, power_dens_arr, state_labels)


class ConcentrationDependenceSolution(SolutionList):
    '''Solution to a concentration dependence simulation'''
    def __init__(self, dynamics: bool = False) -> None:
        '''If dynamics is true the solution list stores DynamicsSolution,
           otherwise it stores SteadyStateSolution
        '''
        super(ConcentrationDependenceSolution, self).__init__()
        self.dynamics = dynamics
        # constructor of the underliying class that the list stores
        # the load method will create instances of this type
        if dynamics:
            self._items_class = DynamicsSolution
        else:
            self._items_class = SteadyStateSolution
        self._prefix = 'conc_dep'

    def __repr__(self) -> str:
        '''Representation of a concentration dependence list.'''
        concs = [sol.concentration for sol in self]
        power = self[0].power_dens
        return ('{}(num_solutions={}, concs={}, '.format(self.__class__.__name__,
                                                         len(self), concs) +
                'power_dens={}, dynamics={})'.format(power, self.dynamics))

    @log_exceptions_warnings
    def plot(self) -> None:
        '''Plot the concentration dependence of the emission for all states.
        '''
        if len(self) == 0:  # nothing to plot
            logger = logging.getLogger(__name__)
            msg = 'Nothing to plot! The concentration_dependence list is emtpy!'
            logger.warning(msg)
            warnings.warn(msg, plotter.PlotWarning)
            return

        if self[0].cte['no_plot']:
            logger = logging.getLogger(__name__)
            msg = 'A plot was requested, but no_plot setting is set'
            logger.warning(msg)
            warnings.warn(msg, plotter.PlotWarning)
            return

        title = '{}: {}, {}. P={} W/cm²'.format(self[0].cte.lattice['name'],
                                        self[0].cte.states['sensitizer_ion_label'],
                                        self[0].cte.states['activator_ion_label'],
                                        exp_to_10(self[0].power_dens))

        if self.dynamics is True:
            # plot all decay curves together
            import matplotlib.pyplot as plt
            color_map = plt.get_cmap('Paired', 2*len(self))
            color_list = [(color_map(num), color_map(num+1)) for num in range(0, 2*len(self), 2)]
            # plot all concentrations in the same figure
            single_figure = plotter.new_figure()
            for color, sol in zip(color_list, self):
                sol = cast(DynamicsSolution, sol)  # no runtime effect, only for mypy

                plotter.plot_avg_decay_data(sol.t_sol, sol.list_avg_data_ofs,
                                            list_exp_data=sol.list_exp_data,
                                            state_labels=sol.state_labels,
                                            concentration=sol.concentration,
                                            colors=color,
                                            fig=single_figure, title=title)
        else:
            sim_data_arr = np.array([np.array(sol.steady_state_populations)
                                     for sol in self])

            S_states = self[0].cte.states['sensitizer_states']
            A_states = self[0].cte.states['activator_states']
            conc_factor_arr = np.array([([float(sol.concentration.S_conc)]*S_states +
                                         [float(sol.concentration.A_conc)]*A_states)
                                        for sol in self])

            # multiply the average state populations by the concentration
            # TODO: is this correct?
            sim_data_arr *= conc_factor_arr

            # if all elements of S_conc_l are equal use A_conc to plot and viceversa
            S_conc_l = [float(sol.concentration.S_conc) for sol in self]
            A_conc_l = [float(sol.concentration.A_conc) for sol in self]
            if S_conc_l.count(S_conc_l[0]) == len(S_conc_l):
                conc_arr = np.array(A_conc_l)
            elif A_conc_l.count(A_conc_l[0]) == len(A_conc_l):
                conc_arr = np.array(S_conc_l)
            else:
                # do a 2D heatmap otherwise
                conc_arr = np.array(list(zip(S_conc_l, A_conc_l)))

            # plot
            state_labels = self[0].state_labels
            plotter.plot_concentration_dependence(sim_data_arr, conc_arr, state_labels)


class Simulations():
    '''Setup and solve a dynamics or a steady state problem'''

    def __init__(self, cte: Settings, full_path: str = None) -> None:
        # settings
        self.cte = cte
        self.full_path = full_path

    def __bool__(self) -> bool:
        '''Instance is True if the cte dict has been filled'''
        return bool(self.cte)

    def __eq__(self, other: object) -> bool:
        '''Two solutions are equal if all its vars are equal.'''
        if not isinstance(other, Simulations):
            return NotImplemented
        return self.cte == other.cte and self.full_path == other.full_path

    def __ne__(self, other: object) -> bool:
        '''Define a non-equality test'''
        if not isinstance(other, Simulations):
            return NotImplemented
        return not self == other

    def __repr__(self) -> str:
        '''Representation of a simulation.'''
        return '{}(lattice={}, '.format(self.__class__.__name__,
                                        self.cte.lattice['name']) +\
               'n_uc={}, num_states={})'.format(self.cte.lattice['N_uc'],
                                                self.cte.states['energy_states'])


    @log_exceptions_warnings
    def _get_t_pulse(self) -> float:
        '''Return the pulse width of the simulation'''
        try:
            for excitation in self.cte.excitations.values():  # pragma: no branch
                if excitation[0].active:
                    tf_p = excitation[0].t_pulse  # pulse width.
                    break
            type(tf_p)
        except (KeyError, NameError):  # pragma: no cover
            msg = ('t_pulse value not found! ' +
                   'Please add t_pulse to your excitation settings.')
            raise ValueError(msg)
        return tf_p

    def save_file_full_name(self, prefix: str = None) -> str:  # pragma: no cover
        '''Return the full name to save a file (without extention or prefix).'''
        lattice_name = self.cte.lattice['name']
        path = os.path.join('results', lattice_name)
        os.makedirs(path, exist_ok=True)
        filename = prefix + '_' + '{}uc_{}S_{}A'.format(int(self.cte.lattice['N_uc']),
                                                        float(self.cte.lattice['S_conc']),
                                                        float(self.cte.lattice['A_conc']))
        return os.path.join(path, filename)

#    @profile
    def simulate_dynamics(self, average: bool = False) -> DynamicsSolution:
        ''' Simulates the absorption, decay and energy transfer processes contained in cte
            Returns a DynamicsSolution instance
            average=True solves an average rate equation problem instead of the microscopic one.
        '''
        logger = logging.getLogger(__name__ + '.dynamics')

        start_time = time.time()
        logger.info('Starting simulation...')

        setup_func = precalculate.setup_microscopic_eqs
        if average:
            setup_func = precalculate.setup_average_eqs

        # get matrices of interaction, initial conditions, abs, decay, etc
        (cte_updated, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix,
         ET_matrix, N_indices, jac_indices,
         coop_ET_matrix, coop_N_indices,
         coop_jac_indices) = setup_func(self.cte, full_path=self.full_path)

        # update cte
        self.cte = cte_updated

        # initial and final times for excitation and relaxation
        t0 = 0
        tf = (10*np.max(precalculate.get_lifetimes(self.cte))).round(8)  # total simulation time
        t0_p = t0
        tf_p = self._get_t_pulse()
        N_steps_pulse = 2
        t0_sol = tf_p
        tf_sol = tf
        N_steps = self.cte.simulation_params['N_steps']

        rtol = self.cte.simulation_params['rtol']
        atol = self.cte.simulation_params['atol']

        start_time_ODE = time.time()
        logger.info('Solving equations...')

        # excitation pulse
        logger.info('Solving excitation pulse...')
        logger.info('Active excitation(s): ')
        for exc_label, excitation in self.cte.excitations.items():
            # if the current excitation is not active jump to the next one
            if excitation[0].active is True:
                logger.info(exc_label)
        t_pulse = np.linspace(t0_p, tf_p, N_steps_pulse, dtype=np.float64)
        y_pulse = odesolver.solve_pulse(t_pulse, initial_population.transpose(),
                                        total_abs_matrix, decay_matrix,
                                        ET_matrix, N_indices, jac_indices,
                                        coop_ET_matrix, coop_N_indices, coop_jac_indices,
                                        rtol=rtol, atol=atol, quiet=self.cte['no_console'])

        # relaxation
        logger.info('Solving relaxation...')
        t_sol = np.logspace(np.log10(t0_sol), np.log10(tf_sol), N_steps, dtype=np.float64)
        y_sol = odesolver.solve_relax(t_sol, y_pulse[-1, :], decay_matrix,
                                      ET_matrix, N_indices, jac_indices,
                                      coop_ET_matrix, coop_N_indices, coop_jac_indices,
                                      rtol=rtol, atol=atol, quiet=self.cte['no_console'])

        formatted_time = time.strftime("%Mm %Ss", time.localtime(time.time()-start_time_ODE))
        logger.info('Equations solved! Total time: %s.', formatted_time)
        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Simulation finished! Total time: %s.', formatted_time)

        # substract the pulse width from t_sol so that it starts with 0
        # like it happens in a measurement
        t_sol = t_sol - t_sol[0]

        # store solution and settings
        dynamics_sol = DynamicsSolution(t_sol, y_sol, index_S_i, index_A_j,
                                        self.cte, average=average)
        return dynamics_sol

    def simulate_avg_dynamics(self) -> DynamicsSolution:
        '''Simulates the dynamics of a average rate equations system,
            it calls simulate_dynamics
        '''
        return self.simulate_dynamics(average=True)

    def simulate_steady_state(self, average: bool = False) -> SteadyStateSolution:
        ''' Simulates the steady state of the problem
            Returns a SteadyStateSolution instance
            average=True solves an average rate equation problem instead of the microscopic one.
        '''
        logger = logging.getLogger(__name__ + '.steady_state')

        start_time = time.time()
        logger.info('Starting simulation...')

        setup_func = precalculate.setup_microscopic_eqs
        if average:
            setup_func = precalculate.setup_average_eqs

        # get matrices of interaction, initial conditions, abs, decay, etc
        (cte_updated, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix,
         ET_matrix, N_indices, jac_indices,
         coop_ET_matrix, coop_N_indices,
         coop_jac_indices) = setup_func(self.cte, full_path=self.full_path)

        # update cte
        self.cte = cte_updated

        # initial and final times for excitation and relaxation
        t0 = 0
        tf = (10*np.max(precalculate.get_lifetimes(self.cte))).round(8)  # total simulation time
        t0_p = t0
        tf_p = tf
        N_steps_pulse = self.cte.simulation_params['N_steps']

        rtol = self.cte.simulation_params['rtol']
        atol = self.cte.simulation_params['atol']

        start_time_ODE = time.time()
        logger.info('Solving equations...')

        # steady state
        logger.info('Solving steady state...')
        logger.info('Active excitation(s): ')
        for exc_label, excitation in self.cte.excitations.items():
            # if the current excitation is not active jump to the next one
            if excitation[0].active is True:
                logger.info('{}: P = {} W/cm2.'.format(exc_label, excitation[0].power_dens))
        t_pulse = np.linspace(t0_p, tf_p, N_steps_pulse)
        y_pulse = odesolver.solve_pulse(t_pulse, initial_population.transpose(),
                                        total_abs_matrix, decay_matrix,
                                        ET_matrix, N_indices, jac_indices,
                                        coop_ET_matrix, coop_N_indices, coop_jac_indices,
                                        nsteps=1000,
                                        rtol=rtol, atol=atol, quiet=self.cte['no_console'])

        logger.info('Equations solved! Total time: %.2fs.', time.time()-start_time_ODE)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Simulation finished! Total time: %s.', formatted_time)

        # store solution and settings
        steady_sol = SteadyStateSolution(t_pulse, y_pulse, index_S_i, index_A_j,
                                         self.cte, average=average)
        return steady_sol

    def simulate_avg_steady_state(self) -> SteadyStateSolution:
        '''Simulates the steady state of an average rate equations system,
            it calls simulate_steady_state
        '''
        return self.simulate_steady_state(average=True)

    def simulate_power_dependence(self, power_dens_list: List[float],
                                  average: bool = False) -> PowerDependenceSolution:
        ''' Simulates the power dependence.
            power_dens_list can be a list, tuple or a numpy array
            Returns a PowerDependenceSolution instance
            average=True solves an average rate equation problem instead of the microscopic one.
        '''
        logger = logging.getLogger(__name__)
        logger.info('Simulating power dependence curves...')
        start_time = time.time()

        # make sure it's a list of floats so the serialization of cte is correct
        power_dens_list = [float(elem) for elem in list(power_dens_list)]

        num_power_steps = len(power_dens_list)
        solutions = []  # type: List[Solution]

        for power_dens in tqdm(power_dens_list, unit='points',
                               total=num_power_steps, disable=self.cte['no_console'],
                               desc='Total progress'):
            # update power density
            for excitation in self.cte.excitations.keys():
                for exc in self.cte.excitations[excitation]:
                    exc.power_dens = power_dens
            # calculate steady state populations
            with disable_loggers([__name__+'.steady_state', 'simetuc.precalculate',
                                   'simetuc.lattice']):
                steady_sol = self.simulate_steady_state(average=average)
            solutions.append(steady_sol)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Power dependence curves finished! Total time: %s.', formatted_time)

        power_dep_solution = PowerDependenceSolution()
        power_dep_solution.add_solutions(solutions)

        return power_dep_solution

    def simulate_concentration_dependence(self, concentration_list: List[Tuple[float, float]],
                                          dynamics: bool = False, average: bool = False
                                         ) -> ConcentrationDependenceSolution:
        ''' Simulates the concentration dependence of the emission
            concentration_list must be a list of tuples
            If dynamics is True, the dynamics is simulated instead of the steady state
            Returns a ConcentrationDependenceSolution instance
            average=True solves an average rate equation problem instead of the microscopic one.
        '''
        logger = logging.getLogger(__name__)
        logger.info('Simulating concentration dependence curves of ' +
                    '{}.'.format('dynamics' if dynamics is True else 'steady state'))

        start_time = time.time()

        # make sure it's a list of tuple of two floats
        concentration_list = [(float(a), float(b)) for a, b in list(concentration_list)]

        num_conc_steps = len(concentration_list)
        solutions = []  # type: List[Solution]

        for concs in tqdm(concentration_list, unit='points',
                          total=num_conc_steps, disable=self.cte['no_console'],
                          desc='Total progress'):
            # update concentrations
            self.cte.lattice['S_conc'] = concs[0]
            self.cte.lattice['A_conc'] = concs[1]

            # simulate
            if dynamics:
                with disable_loggers([__name__+'.dynamics', 'simetuc.precalculate',
                                       'simetuc.lattice']):
                    sol = self.simulate_dynamics(average=average)  # type: Solution
            else:
                with disable_loggers([__name__+'.steady_state', 'simetuc.precalculate',
                                       'simetuc.lattice']):
                    sol = self.simulate_steady_state(average=average)  # pylint: disable=R0204
            solutions.append(sol)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Power dependence curves finished! Total time: %s.', formatted_time)

        conc_dep_solution = ConcentrationDependenceSolution(dynamics=dynamics)
        conc_dep_solution.add_solutions(solutions)

        return conc_dep_solution


#if __name__ == "__main__":
#    from simetuc.util import disable_console_handler
#
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#    logger.info('Called from cmd.')
#
#    cte = settings.load('config_file.cfg')
#
#    cte['no_console'] = False
#    cte['no_plot'] = False
#
#    sim = Simulations(cte)
#
#    with disable_console_handler('simetuc.precalculate'):
#
#        solution = sim.simulate_dynamics()
#        solution.log_errors()
#        solution.plot()
#
#    solution.save()
#    sol = DynamicsSolution.load('results/bNaYF4/dynamics_20uc_0.0S_0.3A.hdf5')
#    assert solution == sol
#    sol.log_errors()
#    sol.plot()
#
#    solution.plot(state=7)
#
#    solution_avg = sim.simulate_avg_dynamics()
#    solution_avg.log_errors()
#    solution_avg.plot()
#
#
#    solution = sim.simulate_steady_state()
#    solution.log_populations()
#    solution.plot()
#
#
#    solution_avg = sim.simulate_avg_steady_state()
#    solution_avg.log_populations()
#    solution_avg.plot()
#
#
#        solution = sim.simulate_power_dependence(cte.power_dependence)
#        solution.plot()
#
#        conc_list = [(0.0, 0.1), (0.0, 0.175), (0.0, 0.3), (0, 0.5)]#, (0, 1.0)]
#        solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
#        solution.plot()
