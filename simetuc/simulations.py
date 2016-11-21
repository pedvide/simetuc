# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:22:41 2015

@author: Villanueva
"""
# unused arguments, needed for ODE solver
# pylint: disable=W0613
# pylint: disable=C0326

import time
import csv
import logging
import warnings
import copy
import os
import pprint
import typing
import ctypes

import h5py
import yaml

import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
import scipy.signal as signal
from scipy.integrate import ode
import scipy.interpolate as interpolate

# nice progress bar
from tqdm import tqdm

import simetuc.setup as setup
import simetuc.lattice as lattice


def _rate_eq_pulse(t, y, abs_matrix, decay_matrix, UC_matrix, N_indices):
    ''' Calculates the rhs of the ODE for the excitation pulse
    '''
    N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
    UC_matrix = UC_matrix.dot(N_prod_sel)

    return abs_matrix.dot(y) + decay_matrix.dot(y) + UC_matrix


def _jac_rate_eq_pulse(t, y, abs_matrix, decay_matrix, UC_matrix, jac_indices):
    ''' Calculates the jacobian of the ODE for the excitation pulse
    '''
    y_values = y[jac_indices[:, 2]]
    nJ_matrix = csc_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                           shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
    UC_J_matrix = UC_matrix.dot(nJ_matrix)

    return abs_matrix.toarray() + decay_matrix.toarray() + UC_J_matrix.toarray()


def _rate_eq(t, y, decay_matrix, UC_matrix, N_indices):
    '''Calculates the rhs of the ODE for the relaxation'''
    N_prod_sel = y[N_indices[:, 0]]*y[N_indices[:, 1]]
    UC_matrix = UC_matrix.dot(N_prod_sel)

    return decay_matrix.dot(y) + UC_matrix


def _rate_eq_dll(decay_matrix, UC_matrix, N_indices):  # pragma: no cover
    ''' Calculates the rhs of the ODE for the relaxation using odesolver.dll'''
    odesolver = ctypes.windll.odesolver

    matrix_ctype = ndpointer(dtype=np.float64, ndim=2, flags='F_CONTIGUOUS')
    # uint64 not supported by c++?, use int32
    vector_int_ctype = ndpointer(dtype=np.int32, ndim=1, flags='F_CONTIGUOUS')
    vector_ctype = ndpointer(dtype=np.float64, ndim=1, flags='F_CONTIGUOUS')

    odesolver.rateEq.argtypes = [ctypes.c_double, vector_ctype,
                                 matrix_ctype,
                                 vector_ctype, vector_int_ctype, vector_int_ctype,
                                 vector_int_ctype, vector_int_ctype,
                                 ctypes.c_uint, ctypes.c_uint, vector_ctype]
    odesolver.rateEq.restype = ctypes.c_int

    n_states = decay_matrix.shape[0]
    n_inter = UC_matrix.shape[1]

    # eigen uses Fortran ordering
#    abs_matrix = np.asfortranarray(abs_matrix.toarray(), dtype=np.float64)
    decay_matrix = np.asfortranarray(decay_matrix.toarray(), dtype=np.float64)

    # setup gives csr matrix, which isn't fortran style
    UC_matrix = csc_matrix(UC_matrix)  # setup gives csr matrix, which isn't fortran style
    UC_matrix_data = np.asfortranarray(UC_matrix.data, dtype=np.float64)
    UC_matrix_indices = np.asfortranarray(UC_matrix.indices, dtype=np.uint32)
    UC_matrix_indptr = np.asfortranarray(UC_matrix.indptr, dtype=np.uint32)

    N_indices_i = np.asfortranarray(N_indices[:, 0], dtype=np.uint32)
    N_indices_j = np.asfortranarray(N_indices[:, 1], dtype=np.uint32)

    out_vector = np.asfortranarray(np.zeros((n_states,)), dtype=np.float64)

    def rate_eq(t, y):
        '''Calculates the rhs of the ODE for the relaxation'''
        odesolver.rateEq(y,
                         decay_matrix,
                         UC_matrix_data, UC_matrix_indices, UC_matrix_indptr,
                         N_indices_i, N_indices_j,
                         n_states, n_inter, out_vector)
        return out_vector

    return rate_eq


def _jac_rate_eq(t, y, decay_matrix, UC_matrix, jac_indices):
    ''' Calculates the jacobian of the ODE for the relaxation
    '''
    y_values = y[jac_indices[:, 2]]
    nJ_matrix = csc_matrix((y_values, (jac_indices[:, 0], jac_indices[:, 1])),
                           shape=(UC_matrix.shape[1], UC_matrix.shape[0]), dtype=np.float64)
    UC_J_matrix = UC_matrix.dot(nJ_matrix)
    jacobian = UC_J_matrix.toarray() + decay_matrix.toarray()

    return jacobian


def _solve_ode(t_arr, fun, fargs, jfun, jargs, initial_population,
               rtol=1e-3, atol=1e-5, nsteps=500, method='bdf', quiet=True):
    ''' Solve the ode for the times t_arr using rhs fun and jac jfun
        with their arguments as tuples.
    '''
    logger = logging.getLogger(__name__)

    N_steps = len(t_arr)
    y_arr = np.zeros((N_steps, len(initial_population)), dtype=np.float64)

    # setup the ode solver with the method
    ode_obj = ode(fun, jfun)
    ode_obj.set_integrator('vode', rtol=rtol, atol=atol, method=method, nsteps=nsteps)
    ode_obj.set_initial_value(initial_population, t_arr[0])
    ode_obj.set_f_params(*fargs)
    ode_obj.set_jac_params(*jargs)

    # initial conditions
    y_arr[0, :] = initial_population
    step = 1

    # console bar enabled for INFO
    # this doesn't work, as there are two handlers with different levels
    cmd_bar_disable = quiet
    pbar_cmd = tqdm(total=N_steps, unit='step', smoothing=0.1,
                    disable=cmd_bar_disable, desc='ODE progress')

    # catch numpy warnings and log them
    # DVODE (the internal routine used by the integrator 'vode') will throw a warning
    # if it needs too many steps to solve the ode.
    np.seterr(all='raise')
    with warnings.catch_warnings():
        # transform warnings into exceptions that we can catch
        warnings.filterwarnings('error')
        try:
            while ode_obj.successful() and step < N_steps:
                # advance ode to the next time step
                y_arr[step, :] = ode_obj.integrate(t_arr[step])
                step += 1
                pbar_cmd.update(1)
        except UserWarning as err:  # pragma: no cover
            logger.warning(err)
            logger.warning('Most likely the ode solver is taking too many steps.')
            logger.warning('Either change your settings or increase "nsteps".')
            logger.warning('The program will continue, but the accuracy of the ' +
                           'results cannot be guaranteed.')
    np.seterr(all='ignore')  # restore settings

    pbar_cmd.update(1)
    pbar_cmd.close()

    return y_arr


class Solution():
    '''Base class for solutions of rate equation problems'''

    def __init__(self):
        # simulation time
        self.t_sol = np.array([])
        # population of each state of each ion
        self.y_sol = np.array([])
        # list of average population for each state
        self._list_avg_data = np.array([])
        # settings
        self.cte_copy = {}
        # sensitizer and activator indices of their ground states
        self.index_S_i = []
        self.index_A_j = []
        # state labels
        self._state_labels = []

    def __bool__(self):
        '''Instance is True if all its data structures have been filled out'''
        return (self.t_sol.size != 0 and self.y_sol.size != 0 and self.cte_copy != {} and
                len(self.index_S_i) != 0 and len(self.index_A_j) != 0)

    def __eq__(self, other):
        '''Two solutions are equal if all its vars are equal or numerically close'''
        return (np.allclose(self.t_sol, other.t_sol) and np.allclose(self.y_sol, other.y_sol) and
                self.cte_copy == other.cte_copy and self.index_S_i == other.index_S_i and
                self.index_A_j == other.index_A_j)

    def __ne__(self, other):
        '''Define a non-equality test'''
        return not self == other

    def _calculate_avg_populations(self):
        '''Returs the average populations of each state. First S then A states.'''

        cte = self.cte_copy
        index_S_i = self.index_S_i
        index_A_j = self.index_A_j
        y_sol = self.y_sol

        # average population of the ground and excited states of S
        if cte['ions']['sensitizers'] is not 0:
            sim_data_Sensitizer = []
            for state in range(cte['states']['sensitizer_states']):
                population = np.sum([y_sol[:, index_S_i[i]+state]
                                     for i in range(cte['ions']['total'])
                                     if index_S_i[i] != -1], 0)/cte['ions']['sensitizers']
                sim_data_Sensitizer.append(population.clip(0))
        else:
            sim_data_Sensitizer = cte['states']['sensitizer_states']*[np.zeros((y_sol.shape[0],))]
        # average population of the ground and excited states of A
        if cte['ions']['activators'] is not 0:
            sim_data_Activator = []
            for state in range(cte['states']['activator_states']):
                population = np.sum([y_sol[:, index_A_j[i]+state]
                                     for i in range(cte['ions']['total'])
                                     if index_A_j[i] != -1], 0)/cte['ions']['activators']
                sim_data_Activator.append(population.clip(0))
        else:
            sim_data_Activator = cte['states']['activator_states']*[np.zeros((y_sol.shape[0],))]

        return sim_data_Sensitizer + sim_data_Activator

    def _get_ion_state_labels(self) -> list:
        '''Returns a list of ion_state labels'''
        cte = self.cte_copy
        sensitizer_labels = [cte['states']['sensitizer_ion_label'] + '_' + s
                             for s in cte['states']['sensitizer_states_labels']]
        activator_labels = [cte['states']['activator_ion_label'] + '_' + s
                            for s in cte['states']['activator_states_labels']]
        state_labels = sensitizer_labels + activator_labels
        return state_labels

    def save_full_path(self):  # pragma: no cover
        '''Return the full path to save a file (without extention).'''
        lattice_name = self.cte_copy['lattice']['name']
        path = os.path.join('results', lattice_name)
        os.makedirs(path, exist_ok=True)
        full_path = lattice.make_full_path(path, self.cte_copy['lattice']['N_uc'],
                                           self.cte_copy['lattice']['S_conc'],
                                           self.cte_copy['lattice']['A_conc'])
        return full_path

    @property
    def state_labels(self):
        '''List of ion_state labels'''
        # if empty, calculate
        if not len(self._state_labels):
            self._state_labels = self._get_ion_state_labels()
        return self._state_labels

    @property
    def list_avg_data(self):
        '''List of average populations for each state in the solution'''
        # if empty, calculate
        if not len(self._list_avg_data):
            self._list_avg_data = self._calculate_avg_populations()
        return self._list_avg_data

    def add_sim_data(self, t_sol: np.ndarray, y_sol: np.ndarray):
        '''Add the simulated solution data'''
        self.t_sol = t_sol
        self.y_sol = y_sol

    def add_ion_lists(self, index_S_i: typing.List[int], index_A_j: typing.List[int]):
        '''Add the sensitizer and activator ion lists'''
        self.index_S_i = index_S_i
        self.index_A_j = index_A_j

    def copy_settings(self, cte: dict):
        '''Copy the settings related to this solution'''
        self.cte_copy = copy.deepcopy(cte)

    def plot(self, state: int=None):
        '''Plot the soltion of a problem.
            If state is given, the population of only that state for all ions
            is shown along with the average.
        '''
        if self.cte_copy['no_plot']:
            logger = logging.getLogger(__name__)
            msg = 'A plot was requested, but no_plot setting is set'
            logger.warning(msg)
            warnings.warn(msg, PlotWarning)
            return

        # get ion_state labels
        state_labels = self.state_labels

        if state is None:
            Plotter.plot_avg_decay_data(self)
        elif state < len(state_labels):
            if state < self.cte_copy['states']['sensitizer_states']:
                indices = self.index_S_i
            else:
                indices = self.index_A_j
            label = state_labels[state]
            population = np.array([self.y_sol[:, index+state]
                                   for index in indices if index != -1])
            Plotter.plot_state_decay_data(self.t_sol, population.T,
                                          state_label=label, atol=1e-18)

    def save(self, full_path: str = None):
        '''Save data to disk as a HDF5 file'''
        if full_path is None:  # pragma: no cover
            full_path = self._save_full_path() + '.hdf5'
        with h5py.File(full_path, 'w') as file:
            file.create_dataset("t_sol", data=self.t_sol, compression='gzip')
            file.create_dataset("y_sol", data=self.y_sol, compression='gzip')
            file.create_dataset("index_S_i", data=self.index_S_i, compression='gzip')
            file.create_dataset("index_A_j", data=self.index_A_j, compression='gzip')
            # serialze cte_copy as text and store it as an attribute
            file.attrs['cte_copy'] = yaml.dump(self.cte_copy)

    def save_npz(self, full_path: str = None):
        '''Save data to disk as numpy .npz file'''
        if full_path is None:  # pragma: no cover
            full_path = self._save_full_path() + '.npz'
        np.savez_compressed(full_path, t_sol=self.t_sol, y_sol=self.y_sol,
                            cte_copy=[self.cte_copy],  # store as a list of dicts
                            index_S_i=self.index_S_i, index_A_j=self.index_A_j)

    def save_txt(self, full_path: str = None):
        '''Save the settings, the time and the average populations to disk as a textfile'''
        if full_path is None:  # pragma: no cover
            full_path = self._save_full_path() + '.txt'
        # print cte
        with open(full_path, 'wt') as csvfile:
            csvfile.write('Settings:\n')
            pprint.pprint(self.cte_copy, stream=csvfile)
            csvfile.write('\nData:\n')
        # print t_sol and avg sim data
        header = ('time (s)      ' +
                  '         '.join(self.cte_copy['states']['sensitizer_states_labels']) +
                  '         ' +
                  '         '.join(self.cte_copy['states']['activator_states_labels']))
        with open(full_path, 'ab') as csvfile:
            np.savetxt(csvfile, np.transpose([self.t_sol, *self.list_avg_data]),
                       fmt='%1.4e', delimiter=', ', newline='\r\n', header=header)

    def load_npz(self, full_path: str):
        '''Load data from a numpy .npz file'''
        try:
            npz_file = np.load(full_path)
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error('File not found! (%s)', full_path)
            raise

        self.t_sol = npz_file['t_sol']
        self.y_sol = npz_file['y_sol']
        self.cte_copy = npz_file['cte_copy'][0]  # stored as a list of dicts
        self.index_S_i = list(npz_file['index_S_i'])
        self.index_A_j = list(npz_file['index_A_j'])

    def load(self, full_path: str):
        '''Load data from a HDF5 file'''
        try:
            with h5py.File(full_path, 'r') as file:
                self.t_sol = np.array(file['t_sol'])
                self.y_sol = np.array(file['y_sol'])
                self.index_S_i = list(file['index_S_i'])
                self.index_A_j = list(file['index_A_j'])
                # deserialze cte_copy
                self.cte_copy = yaml.load(file.attrs['cte_copy'])
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error('File not found! (%s)', full_path)
            raise


class SteadyStateSolution(Solution):
    '''Class representing the solution to a steady state problem'''

    def __init__(self):
        super(SteadyStateSolution, self).__init__()
        self._final_populations = np.array([])

    @property
    def power_dens(self):
        '''Return the power density used to obtain this solution.'''
        for excitation in self.cte_copy['excitations'].keys():
            return self.cte_copy['excitations'][excitation]['power_dens']

    @property
    def concentration(self):
        '''Return the tuple (sensitizer, activator) concentration used to obtain this solution.'''
        return (self.cte_copy['lattice']['S_conc'], self.cte_copy['lattice']['A_conc'])

    def _calculate_final_populations(self):
        '''Calculate the final population for all states after a steady state simulation'''
        return [curve[-1] for curve in self.list_avg_data]

    @property
    def steady_state_populations(self):
        '''List of final steady-state populations for each state in the solution'''
        # if empty, calculate
        if not len(self._final_populations):
            self._final_populations = self._calculate_final_populations()
        return self._final_populations

    def log_populations(self):
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

    def __init__(self):
        super(DynamicsSolution, self).__init__()

        self._list_exp_data = []
        self._list_avg_data_ofs = []

        self._total_error = None
        self._errors = np.array([])

    #@profile
    @staticmethod
    def _load_exp_data(filename, lattice_name, filter_window=11):
        '''Load the experimental data from the expData/lattice_name folder.
           Two columns of numbers: first is time (seconds), second intensity
        '''

        path = os.path.join('expData', lattice_name, filename)
        try:
            with open(path, 'rt') as file:
                try:  # TODO: get a better way to read data
                    data = csv.reader(file, delimiter='\t')
                    data = [row for row in data if row[0][0] is not '#']
                    data = np.array(data, dtype=np.float64)
                except ValueError:  # pragma: no cover
                    data = csv.reader(file, delimiter=',')
                    data = [row for row in data if row[0][0] is not '#']
                    data = np.array(data, dtype=np.float64)
    #        data = np.loadtxt(path, usecols=(0, 1)) # 10x slower
        except FileNotFoundError:
            # exp data dones't exist. not a problem.
            return 0

        if len(data) == 0:
            return 0

        # smooth the data to get an "average" of the maximum
        smooth_data = signal.savgol_filter(data[:, 1], filter_window, 2, mode='nearest')
        # normalize data
        data[:, 1] = (data[:, 1]-min(smooth_data))/(max(smooth_data)-min(smooth_data))
        # set negative values to zero
        data = data.clip(min=0)
        return data

    #@profile
    @staticmethod
    def _correct_background(exp_data, sim_data, offset_points=50):
        '''Add the experimental background to the simulated data.
           Returns the same simulated data if there's no exp_data
            expData is already normalized when loaded.
        '''
        if not np.any(exp_data):  # if there's no experimental data, don't do anything
            return sim_data
        if not np.any(sim_data):  # pragma: no cover
            return 0

        last_points = exp_data[-offset_points:, 1]  # get last 50 points
        offset = np.mean(last_points[last_points > 0])*max(sim_data)

        if np.isnan(offset):  # pragma: no cover
            sim_data_ofs = sim_data
        else:  # offset-correct simulated data
            sim_data_ofs = sim_data+offset

        return sim_data_ofs

    #@profile
    def _interpolate_sim_data(self):
        '''Interpolated simulated corrected data to exp data points
        '''
        # create function to interpolate
        list_iterp_sim_funcs = [interpolate.interp1d(self.t_sol, simData_corr,
                                                     fill_value='extrapolate')
                                if simData_corr is not 0 else 0
                                for simData_corr in self.list_avg_data_ofs]
        # interpolate them to the experimental data times
        list_iterp_sim_data = [iterpFun(expData[:, 0])
                               if (expData is not 0) and (iterpFun is not 0) else 0
                               for iterpFun, expData in zip(list_iterp_sim_funcs,
                                                            self.list_exp_data)]

        return list_iterp_sim_data

    #@profile
    def _calc_errors(self):
        '''Calculate root-square-deviation between experiment and simulation
        '''
        # get interpolated simulated data
        list_iterp_sim_data = self._interpolate_sim_data()

        # calculate the relative mean square deviation
        # error = 1/mean(y)*sqrt(sum( (y-yexp)^2 )/N )
        rmdevs = [(sim-exp[:, 1]*np.max(sim))**2
                  if (exp is not 0) and (sim is not 0) else 0
                  for (sim, exp) in zip(list_iterp_sim_data, self.list_exp_data)]
        errors = [1/np.mean(sim)*np.sqrt(1/len(sim)*np.sum(rmdev))
                  if rmdev is not 0 else 0
                  for (rmdev, sim) in zip(rmdevs, list_iterp_sim_data)]
        errors = np.array(errors)

        return errors

    def _load_decay_data(self):
        '''Load the decay experimental data.
        '''
        # get filenames from the ion_state labels, excitation and concentrations
        state_labels = self.state_labels
        active_exc_labels = [label for label, exc_dict in self.cte_copy['excitations'].items()
                             if exc_dict['active']]
        exc_label = '_'.join(active_exc_labels)
        S_conc = str(float(self.cte_copy['lattice']['S_conc']))
        S_label = self.cte_copy['states']['sensitizer_ion_label']
        A_conc = str(float(self.cte_copy['lattice']['A_conc']))
        A_label = self.cte_copy['states']['activator_ion_label']
        conc_str = '_' + S_conc + S_label + '_' + A_conc + A_label
        exp_data_filenames = ['decay_' + label + '_exc_' + exc_label + conc_str + '.txt'
                              for label in state_labels]

        # if exp data doesn't exist, it's set to zero inside the function
        list_exp_data = [self._load_exp_data(filename, self.cte_copy['lattice']['name'])
                         for filename in exp_data_filenames]

        return list_exp_data

    def log_errors(self):
        '''Log errors'''
        logger = logging.getLogger(__name__)

        # get state labels
        state_labels = self.state_labels

        # log errors them
        if state_labels is not None:
            logger.info('State errors: ')
            for (label, error) in zip(state_labels, self.errors):
                logger.info('%s: %.4e', label, error)
        logger.info('Total error: %.4e', self.total_error)

    @property
    def errors(self):
        '''List of root-square-deviation between experiment and simulation
            for each state in the solution
        '''
        # if empty, calculate
        if not len(self._errors):
            self._errors = self._calc_errors()
        return self._errors

    @property
    def total_error(self):
        '''Total root-square-deviation between experiment and simulation'''
        # if none, calculate
        if not self._total_error:
            if np.any(self.errors):
                total_error = np.sqrt(np.mean(np.square(self.errors[self.errors > 0])))
            else:
                total_error = 0
            self._total_error = total_error
        return self._total_error

    @property
    def list_avg_data_ofs(self):
        '''List of offset-corrected (due to experimental background) average populations
            for each state in the solution
        '''
        # if empty, calculate
        if not self._list_avg_data_ofs:
            self._list_avg_data_ofs = [DynamicsSolution._correct_background(expData, simData)
                                       for expData, simData
                                       in zip(self.list_exp_data, self.list_avg_data)]
        return self._list_avg_data_ofs

    @property
    def list_exp_data(self):
        '''List of ofset-corrected average populations for each state in the solution'''
        # if empty, calculate
        if not self._list_exp_data:
            self._list_exp_data = self._load_decay_data()
        return self._list_exp_data


class SolutionList():
    '''Base class for a list of solutions for problems like power or concentration dependence.'''
    def __init__(self):
        self.solution_list = ()
        # constructor of the underliying class that the list stores.
        # the load method will create instances of this type
        self._items_class = Solution
        self._suffix = ''

    def __bool__(self):
        '''Instance is True if its list is not emtpy.'''
        return len(self.solution_list) != 0

    def __eq__(self, other):
        '''Two solutions are equal if all their solutions are equal.'''
        return self.solution_list == other.solution_list

    def add_solutions(self, sol_list: typing.List[Solution]) -> None:
        '''Add a list of solutions.'''
        self.solution_list = tuple(sol_list)

    def save(self, full_path: str = None) -> None:
        '''Save all data from all solutions in a HDF5 file'''
        if full_path is None:  # pragma: no cover
            full_path = self.solution_list[0].save_full_path() + '_' + self._suffix + '.hdf5'

        with h5py.File(full_path, 'w') as file:
            for num, sol in enumerate(self.solution_list):
                group = file.create_group(str(num))
                group.create_dataset("t_sol", data=sol.t_sol, compression='gzip')
                group.create_dataset("y_sol", data=sol.y_sol, compression='gzip')
                group.create_dataset("index_S_i", data=sol.index_S_i, compression='gzip')
                group.create_dataset("index_A_j", data=sol.index_A_j, compression='gzip')
                # serialze cte_copy as text and store it as an attribute
                group.attrs['cte_copy'] = yaml.dump(sol.cte_copy)

    def load(self, full_path: str) -> None:
        '''Load data from a HDF5 file'''
        solutions = []
        try:
            with h5py.File(full_path, 'r') as file:
                for group_num in file:
                    sol = self._items_class()  # create appropiate object
                    group = file[group_num]
                    sol.add_sim_data(np.array(group['t_sol']), np.array(group['y_sol']))
                    sol.add_ion_lists(list(group['index_S_i']), list(group['index_A_j']))
                    # deserialze cte_copy
                    sol.cte_copy = yaml.load(group.attrs['cte_copy'])
                    solutions.append(sol)
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error('File not found! (%s)', full_path)
            raise
        self.add_solutions(solutions)


class PowerDependenceSolution(SolutionList):
    '''Solution to a power dependence simulation'''
    def __init__(self):
        super(PowerDependenceSolution, self).__init__()
        # constructor of the underliying class that the list stores
        # the load method will create instances of this type
        self._items_class = SteadyStateSolution
        self._suffix = 'pow_dep'

    def plot(self) -> None:
        '''Plot the power dependence of the emission for all states'''
        if len(self.solution_list) == 0:  # nothing to plot
            logger = logging.getLogger(__name__)
            msg = 'Nothing to plot! The power_dependence list is emtpy!'
            logger.warning(msg)
            warnings.warn(msg, PlotWarning)
            return

        if self.solution_list[0].cte_copy['no_plot']:
            logger = logging.getLogger(__name__)
            msg = 'A plot was requested, but no_plot setting is set'
            logger.warning(msg)
            warnings.warn(msg, PlotWarning)
            return

        sim_data_arr = np.array([np.array(sol.steady_state_populations)
                                 for sol in self.solution_list])
        power_dens_arr = np.array([sol.power_dens for sol in self.solution_list])
        state_labels = self.solution_list[0].state_labels

        Plotter.plot_power_dependence(sim_data_arr, power_dens_arr, state_labels)


class ConcentrationDependenceSolution(SolutionList):
    '''Solution to a concentration dependence simulation'''
    def __init__(self, dynamics=False):
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
        self._suffix = 'conc_dep'

    def save(self, full_path: str = None) -> None:
        '''Save all data from all solutions in a HDF5 file'''
        if full_path is None:  # pragma: no cover
            conc_path = self.solution_list[0].save_full_path()
            # get only the data_Xuc part
            conc_path = '_'.join(conc_path.split('_')[:2])
            full_path = conc_path + '_' + self._suffix + '.hdf5'
        super(ConcentrationDependenceSolution, self).save(full_path)

    def plot(self, no_exp=False) -> None:
        '''Plot the concentration dependence of the emission for all states.
           If no_exp is True, no experimental data is plotted.
        '''
        if len(self.solution_list) == 0:  # nothing to plot
            logger = logging.getLogger(__name__)
            msg = 'Nothing to plot! The concentration_dependence list is emtpy!'
            logger.warning(msg)
            warnings.warn(msg, PlotWarning)
            return

        if self.solution_list[0].cte_copy['no_plot']:
            logger = logging.getLogger(__name__)
            msg = 'A plot was requested, but no_plot setting is set'
            logger.warning(msg)
            warnings.warn(msg, PlotWarning)
            return

        if self.dynamics:
            # plot all decay curves together
            color_list = [c+c for c in 'rgbmyc'*3]
            for color, sol in zip(color_list, self.solution_list):
                Plotter.plot_avg_decay_data(sol, no_exp=no_exp, show_conc=True, colors=color)
        else:
            sim_data_arr = np.array([np.array(sol.steady_state_populations)
                                     for sol in self.solution_list])

            S_states = self.solution_list[0].cte_copy['states']['sensitizer_states']
            A_states = self.solution_list[0].cte_copy['states']['activator_states']
            conc_factor_arr = np.array([([float(sol.concentration[0])]*S_states +
                                         [float(sol.concentration[1])]*A_states)
                                        for sol in self.solution_list])

            # multiply the average state populations by the concentration
            # TODO: is this correct?
            sim_data_arr *= conc_factor_arr

            # if all elements of S_conc_l are equal use A_conc to plot and viceversa
            S_conc_l = [float(sol.concentration[0]) for sol in self.solution_list]
            A_conc_l = [float(sol.concentration[1]) for sol in self.solution_list]
            if S_conc_l.count(S_conc_l[0]) == len(S_conc_l):
                conc_arr = np.array(A_conc_l)
            elif A_conc_l.count(A_conc_l[0]) == len(A_conc_l):
                conc_arr = np.array(S_conc_l)
            else:
                # do a 2D heatmap otherwise
                conc_arr = np.array(list(zip(S_conc_l, A_conc_l)))

            # plot
            state_labels = self.solution_list[0].state_labels
            Plotter.plot_concentration_dependence(sim_data_arr, conc_arr, state_labels)


class PlotWarning(UserWarning):
    '''Warning for empty plots'''
    pass


class Plotter():
    '''Plot different solutions to rate equations problems'''

    @staticmethod
    def plot_avg_decay_data(solution: Solution, atol: float = 1e-15,
                            no_exp: bool = False, show_conc: bool = False, colors: str = 'rk'):
        ''' Plot the list of experimental and average simulated data against time in solution.
            If no_exp is True no experimental data will be plotted.
            If show_conc is True, the legend will show the concentrations.
            colors is a string with two chars. The first is the sim color,
            the second the exp data color.
        '''

        # if we have simulated data that has been offset-corrected, use it
        if (hasattr(solution, 'list_avg_data_ofs') and
                solution.list_avg_data_ofs is not solution.list_avg_data):
            list_sim_data = solution.list_avg_data_ofs
        else:
            list_sim_data = [data for data in solution.list_avg_data]

        num_plots = len(list_sim_data)
        num_rows = 3
        num_cols = int(np.ceil(num_plots/3))
        t_sol = solution.t_sol[:]*1000  # convert to ms

        if (not hasattr(solution, 'list_exp_data') or
                solution.list_exp_data is None or
                no_exp is True):
            list_exp_data = len(list_sim_data)*[0]
        else:
            list_exp_data = solution.list_exp_data

        state_labels = solution.state_labels
        if show_conc is True:
            S_conc = str(float(solution.cte_copy['lattice']['S_conc']))
            S_label = solution.cte_copy['states']['sensitizer_ion_label']
            A_conc = str(float(solution.cte_copy['lattice']['A_conc']))
            A_label = solution.cte_copy['states']['activator_ion_label']
            conc_str = '_' + S_conc + S_label + '_' + A_conc + A_label
            state_labels = [label+conc_str for label in state_labels]

        for num, (sim_data_corr, exp_data, state_label)\
            in enumerate(zip(list_sim_data, list_exp_data, state_labels)):
            if sim_data_corr is 0:
                continue
            if (np.isnan(sim_data_corr)).any() or not np.any(sim_data_corr > 0):
                continue

            plt.subplot(num_rows, num_cols, num+1)

            sim_color = colors[0]
            exp_color = colors[1]

            if exp_data is 0:  # no exp data: either a GS or simply no exp data available
                # nonposy='clip': clip non positive values to a very small positive number
                plt.semilogy(t_sol, sim_data_corr, sim_color, label=state_label, nonposy='clip')
                plt.yscale('log', nonposy='clip')
                plt.axis('tight')
                # add some white space above and below
                margin_factor = np.array([0.7, 1.3])
                plt.ylim(*np.array(plt.ylim())*margin_factor)
                if plt.ylim()[0] < atol:
                    plt.ylim(ymin=atol)  # don't show noise below atol
                    # detect when the simulation goes above and below atol
                    above = sim_data_corr > atol
                    change_indices = np.where(np.roll(above, 1) != above)[0]
                    if change_indices.size > 0:
                        # last time it changes
                        max_index = change_indices[-1]
                        # show simData until it falls below atol
                        plt.xlim(xmax=t_sol[max_index])
                min_y = min(*plt.ylim())
                max_y = max(*plt.ylim())
                plt.ylim(ymin=min_y, ymax=max_y)
            else:  # exp data available
                # convert exp_data time to ms
                plt.semilogy(exp_data[:, 0]*1000, exp_data[:, 1]*np.max(sim_data_corr),
                             exp_color, t_sol, sim_data_corr, sim_color, label=state_label)
                plt.axis('tight')
                plt.ylim(ymax=plt.ylim()[1]*1.2)  # add some white space on top
                plt.xlim(xmax=exp_data[-1, 0]*1000)  # don't show beyond expData

            plt.legend(loc="best", fontsize='small')
            plt.xlabel('t (ms)')

    @staticmethod
    def plot_state_decay_data(t_sol: np.ndarray, sim_data_array: np.ndarray,
                              state_label: typing.List[str] = None, atol: int = 1e-15):
        ''' Plots a state's simulated data against time t_sol'''
        t_sol *= 1000  # convert to ms

        if sim_data_array is 0:
            return
        if (np.isnan(sim_data_array)).any() or not np.any(sim_data_array):
            return

        avg_sim = np.mean(sim_data_array, axis=1)

        # nonposy='clip': clip non positive values to a very small positive number
        plt.semilogy(t_sol, sim_data_array, 'k', nonposy='clip')
        plt.semilogy(t_sol, avg_sim, 'r', nonposy='clip', linewidth=5)
        plt.yscale('log', nonposy='clip')
        plt.axis('tight')
        # add some white space above and below
        margin_factor = np.array([0.7, 1.3])
        plt.ylim(*np.array(plt.ylim())*margin_factor)
        if plt.ylim()[0] < atol:
            plt.ylim(ymin=atol)  # don't show noise below atol
            # detect when the simulation goes above and below atol
            above = sim_data_array > atol
            change_indices = np.where(np.roll(above, 1) != above)[0]
            if change_indices.size > 0:
                # last time it changes
                max_index = change_indices[-1]
                # show simData until it falls below atol
                plt.xlim(xmax=t_sol[max_index])

        plt.legend([state_label], loc="best")
        plt.xlabel('t (ms)')

    @staticmethod
    def plot_power_dependence(sim_data_arr: np.ndarray, power_dens_arr: np.ndarray,
                              state_labels: typing.List[str]):
        ''' Plots the intensity as a function of power density for each state'''
        num_plots = len(state_labels)
        num_rows = 3
        num_cols = int(np.ceil(num_plots/3))

        # calculate the slopes for each consecutive pair of points in the curves
        Y = np.log10(sim_data_arr)[:-1, :]
        X = np.log10(power_dens_arr)
        dX = (np.roll(X, -1, axis=0) - X)[:-1]
        # list of slopes
        slopes = [np.gradient(Y_arr, dX) for Y_arr in Y.T]
        slopes = np.around(slopes, 1)

        for num, state_label in enumerate(state_labels):  # for each state
            sim_data = sim_data_arr[:, num]
            if not np.any(sim_data):
                continue

            axis = plt.subplot(num_rows, num_cols, num+1)

            plt.loglog(power_dens_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
            plt.axis('tight')
            margin_factor = np.array([0.7, 1.3])
            plt.ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
            plt.xlim(*np.array(plt.xlim())*margin_factor)

            plt.legend(loc="best")
            plt.xlabel('Power density / W/cm^2')

            if slopes is not None:
                for i, txt in enumerate(slopes[num]):
                    axis.annotate(txt, (power_dens_arr[i], sim_data[i]), xytext=(5, -7),
                                  xycoords='data', textcoords='offset points')

    @staticmethod
    def plot_concentration_dependence(sim_data_arr: np.ndarray, conc_arr: np.ndarray,
                                      state_labels: typing.List[str]):
        '''Plots the concentration dependence of the steady state emission'''
        num_plots = len(state_labels)
        num_rows = 3
        num_cols = int(np.ceil(num_plots/3))

        heatmap = False
        if len(conc_arr.shape) == 2:
            heatmap = True

        for num, state_label in enumerate(state_labels):  # for each state
            sim_data = sim_data_arr[:, num]
            if not np.any(sim_data):
                continue

            ax = plt.subplot(num_rows, num_cols, num+1)

            if not heatmap:
                plt.plot(conc_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
                plt.axis('tight')
                margin_factor = np.array([0.9, 1.1])
                plt.ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
                plt.xlim(*np.array(plt.xlim())*margin_factor)

                plt.legend(loc="best")
                plt.xlabel('Concentration (%)')
                # change axis format to scientifc notation
                xfmt = plt.ScalarFormatter(useMathText=True)
                xfmt.set_powerlimits((-1, 1))
                ax.yaxis.set_major_formatter(xfmt)
            else:
                x, y = conc_arr[:, 0], conc_arr[:, 1]
                z = sim_data

                # Set up a regular grid of interpolation points
                xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)

                # Interpolate
                # random grid
                interp_f = interpolate.Rbf(x, y, z, function='gaussian', epsilon=2)
                zi = interp_f(xi, yi)
#                zi = interpolate.griddata((x, y), z, (xi, yi), method='cubic')
#                interp_f = interpolate.interp2d(x, y, z, kind='linear')
#                zi = interp_f(xi, yi)

                plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                           extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
                plt.scatter(x, y, c=z)
                plt.xlabel('S concentration (%)')
                plt.ylabel('A concentration (%)')
                cb = plt.colorbar()
                cb.formatter.set_powerlimits((0, 0))
                cb.update_ticks()
                cb.set_label(state_label)


class Simulations():
    '''Setup and solve a dynamics or a steady state problem'''

    def __init__(self, cte: dict, full_path: str = None):
        # settings
        self.cte = cte
        self.full_path = full_path

    def modify_ET_param_value(self, process: str, value: float):
        '''Modify a ET parameter'''
        self.cte['ET'][process]['value'] = value

#    @profile
    def simulate_dynamics(self) -> DynamicsSolution:
        ''' Simulates the absorption, decay and energy transfer processes contained in cte
            Returns a DynamicsSolution instance
        '''
        logger = logging.getLogger(__name__)

        start_time = time.time()
        logger.info('Starting simulation...')

        # get matrices of interaction, initial conditions, abs, decay, etc
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix,
         UC_matrix,
         N_indices, jac_indices) = setup.precalculate(self.cte, full_path=self.full_path)
        initial_population = np.asfortranarray(initial_population, dtype=np.float64)

        # update cte
        self.cte = cte

        # initial and final times for excitation and relaxation
        t0 = 0
        tf = (10*np.max(setup.get_lifetimes(cte))).round(8)  # total simulation time
        t0_p = t0
        # make sure t_pulse exists and get the active one
        try:
            for exc_dict in self.cte['excitations'].values():
                if exc_dict['active']:
                    tf_p = exc_dict['t_pulse']  # pulse width.
                    break
            type(tf_p)
        except (KeyError, NameError):
            logger.error('t_pulse value not found!')
            logger.error('Please add t_pulse to your excitation settings.')
            raise
        N_steps_pulse = 2
        t0_sol = tf_p
        tf_sol = tf
        N_steps = self.cte['simulation_params']['N_steps']

        rtol = self.cte['simulation_params']['rtol']
        atol = self.cte['simulation_params']['atol']

        start_time_ODE = time.time()
        logger.info('Solving equations...')

        # excitation pulse
        logger.info('Solving excitation pulse...')
        t_pulse = np.linspace(t0_p, tf_p, N_steps_pulse, dtype=np.float64)
        y_pulse = _solve_ode(t_pulse, _rate_eq_pulse,
                             (total_abs_matrix, decay_matrix, UC_matrix, N_indices),
                             _jac_rate_eq_pulse,
                             (total_abs_matrix, decay_matrix, UC_matrix, jac_indices),
                             initial_population.transpose(), method='adams',
                             rtol=rtol, atol=atol, quiet=self.cte['no_console'])

        # relaxation

        logger.info('Solving relaxation...')
        t_sol = np.logspace(np.log10(t0_sol), np.log10(tf_sol), N_steps, dtype=np.float64)
        y_sol = _solve_ode(t_sol, _rate_eq, (decay_matrix, UC_matrix, N_indices),
                           _jac_rate_eq, (decay_matrix, UC_matrix, jac_indices),
                           y_pulse[-1, :], rtol=rtol, atol=atol,
                           nsteps=1000, quiet=self.cte['no_console'])
#        function = _rate_eq_dll(decay_matrix, UC_matrix, N_indices)
#        y_sol = _solve_ode(t_sol, function, (),
#                           _jac_rate_eq, (decay_matrix, UC_matrix, jac_indices),
#                           y_pulse[-1, :], rtol=rtol, atol=atol,
#                           nsteps=1000, quiet=self.cte['no_console'])

        formatted_time = time.strftime("%Mm %Ss", time.localtime(time.time()-start_time_ODE))
        logger.info('Equations solved! Total time: %s.', formatted_time)
        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Simulation finished! Total time: %s.', formatted_time)

        # store solution and settings
        dynamics_sol = DynamicsSolution()
        dynamics_sol.add_sim_data(t_sol, y_sol)
        dynamics_sol.add_ion_lists(index_S_i, index_A_j)
        dynamics_sol.copy_settings(cte)

        return dynamics_sol

    def simulate_steady_state(self) -> SteadyStateSolution:
        ''' Simulates the steady state of the problem
            Returns a SteadyStateSolution instance
        '''
        logger = logging.getLogger(__name__)

        cte = self.cte

        start_time = time.time()
        logger.info('Starting simulation...')

        # get matrices of interaction, initial conditions, abs, decay, etc
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix,
         UC_matrix, N_indices, jac_indices) = setup.precalculate(self.cte, full_path=self.full_path)

        # initial and final times for excitation and relaxation
        t0 = 0
        tf = (10*np.max(setup.get_lifetimes(cte))).round(8)  # total simulation time
        t0_p = t0
        tf_p = tf
        N_steps_pulse = cte['simulation_params']['N_steps']

        rtol = cte['simulation_params']['rtol']
        atol = cte['simulation_params']['atol']

        start_time_ODE = time.time()
        logger.info('Solving equations...')

        # steady state
        logger.info('Solving steady state...')
        t_pulse = np.linspace(t0_p, tf_p, N_steps_pulse)
        y_pulse = _solve_ode(t_pulse, _rate_eq_pulse,
                             (total_abs_matrix, decay_matrix, UC_matrix, N_indices),
                             _jac_rate_eq_pulse,
                             (total_abs_matrix, decay_matrix, UC_matrix, jac_indices),
                             initial_population.transpose(), nsteps=1000,
                             rtol=rtol, atol=atol, quiet=cte['no_console'])

        logger.info('Equations solved! Total time: %.2fs.', time.time()-start_time_ODE)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Simulation finished! Total time: %s.', formatted_time)

        # store solution and settings
        steady_sol = SteadyStateSolution()
        steady_sol.add_sim_data(t_pulse, y_pulse)
        steady_sol.add_ion_lists(index_S_i, index_A_j)
        steady_sol.copy_settings(cte)

        return steady_sol

    def simulate_power_dependence(self, power_dens_list) -> PowerDependenceSolution:
        ''' Simulates the power dependence.
            power_dens_list can be a list, tuple or a numpy array
            Returns a PowerDependenceSolution instance
        '''
        logger = logging.getLogger(__name__)
        logger.info('Simulating power dependence curves...')
        start_time = time.time()

        # make sure it's a list of floats so the serialization of cte is correct
        power_dens_list = [float(elem) for elem in list(power_dens_list)]

        num_power_steps = len(power_dens_list)
        solutions = []

        for power_dens in tqdm(power_dens_list, unit='points',
                               total=num_power_steps, disable=self.cte['no_console'],
                               desc='Total progress'):
            # update power density
            for excitation in self.cte['excitations'].keys():
                self.cte['excitations'][excitation]['power_dens'] = power_dens
            # calculate steady state populations
            steady_sol = self.simulate_steady_state()
            solutions.append(steady_sol)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Power dependence curves finished! Total time: %s.', formatted_time)

        power_dep_solution = PowerDependenceSolution()
        power_dep_solution.add_solutions(solutions)

        return power_dep_solution

    def simulate_concentration_dependence(self, concentration_list: list, dynamics: bool = False
                                         ) -> ConcentrationDependenceSolution:
        ''' Simulates the concentration dependence of the emission
            concentration_list must be a list of tuples
            If dynamics is True, the dynamics is simulated instead of the steady state
            Returns a ConcentrationDependenceSolution instance
        '''
        logger = logging.getLogger(__name__)
        logger.info('Simulating power dependence curves...')

        cte = self.cte

        start_time = time.time()

        # make sure it's a list of floats
        concentration_list = [(float(a), float(b)) for a, b in list(concentration_list)]

        num_conc_steps = len(concentration_list)
        solutions = []

        for concs in tqdm(concentration_list, unit='points',
                          total=num_conc_steps, disable=cte['no_console'],
                          desc='Total progress'):
            # update concentrations
            cte['lattice']['S_conc'] = concs[0]
            cte['lattice']['A_conc'] = concs[1]
            # simulate
            if dynamics:
                sol = self.simulate_dynamics()
            else:
                sol = self.simulate_steady_state()  # pylint: disable=R0204
            solutions.append(sol)

        total_time = time.time()-start_time
        formatted_time = time.strftime("%Mm %Ss", time.localtime(total_time))
        logger.info('Power dependence curves finished! Total time: %s.', formatted_time)

        conc_dep_solution = ConcentrationDependenceSolution(dynamics=dynamics)
        conc_dep_solution.add_solutions(solutions)

        return conc_dep_solution

#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#
#    logger.info('Called from cmd.')
#
#    import simetuc.settings as settings
#    cte = settings.load('config_file.cfg')
#
#    cte['no_console'] = False
#    cte['no_plot'] = False
#
#    (cte, initial_population, index_S_i, index_A_j,
#     total_abs_matrix, decay_matrix,
#     UC_matrix,
#     N_indices, jac_indices) = setup.precalculate(cte)
#
#    sim = Simulations(cte)
#
##    solution = sim.simulate_dynamics()
##    solution.log_errors()
#
##    solution = sim.simulate_steady_state()
##    solution.log_populations()
#
##    solution.plot()
#
##    solution.save()
##    new_sol = DynamicsSolution()
##    new_sol.load('results/bNaYF4/DynamicsSolution.hdf5')
#
##    power_dens_list = np.logspace(1, 8, 8-1+1)
##    solution = sim.simulate_power_dependence(cte['power_dependence'])
##    solution.plot()
##    solution.save()
##    new_sol = PowerDependenceSolution()
##    new_sol.load('results/bNaYF4/data_30uc_0.0S_0.3A_pow_dep.hdf5')
##    new_sol.plot()
#
#    conc_list = [(0, 0.1), (0, 0.2), (0, 0.3)]
##    N_points = 3
##    S_conc_l = np.linspace(0, 10, N_points)
##    A_conc_l = np.linspace(0.01, 1, N_points)
##    conc_list = list(zip(S_conc_l, A_conc_l))
##    conc_list = np.meshgrid(S_conc_l, A_conc_l)
##    conc_list[0].shape = (conc_list[0].size,1)
##    conc_list[1].shape = (conc_list[0].size,1)
##    conc_list = list(zip(conc_list[0], conc_list[1]))
#    solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
##    solution.plot()
##    solution.save()
#
##    new_sol = ConcentrationDependenceSolution()
##    new_sol.load('results/bNaYF4/data_30uc_0.0S_0.3A_conc_dep.hdf5')
##    new_sol.plot()
#
