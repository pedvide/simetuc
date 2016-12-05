# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:07:21 2015

@author: Villanueva
"""
# pylint: disable=E1101
# TODO: build csr matrices directly using the native: data, indices, indptr.
# now (internally) we build a coo and then it's transformed into csr,
# this goes over the elements and tries to sum duplicates,
# which we don't have (note: not for the ET matrix, but maybe we have for the abs or decay?).
# This wastes time.

import time
import itertools
import os
import logging
from typing import Dict, List, Tuple

import h5py
import yaml

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

import simetuc.lattice as lattice


def _load_lattice(filename: str) -> Dict:
    '''Loads the filename and returns its associanted lattice_info
        Exceptions aren't handled by this function
    '''
    with h5py.File(filename, mode='r') as file:
        # deserialze lattice_info
        lattice_info = yaml.load(file.attrs['lattice_info'])

    return lattice_info


#@profile
def _create_absorption_matrix(abs_sensitizer: np.array, abs_activator: np.array,
                              index_S_i: List[int], index_A_j: List[int]
                             ) -> scipy.sparse.csr_matrix:
    '''Creates the total_states x total_states absorption matrix.
       The block diagonal is either abs_sensitizer or abs_activator depending on
       the position of the GS of S and A given in index_S_i and index_A_j
    '''

    num_total_ions = len(index_A_j)

    abs_sensitizer_sp = scipy.sparse.csr_matrix(abs_sensitizer, dtype=np.float64)
    abs_activator_sp = scipy.sparse.csr_matrix(abs_activator, dtype=np.float64)

    def get_block(num):
        '''Returns a S or A block matrix depending on what kind of ion num is'''
        if index_S_i[num] != -1:
            return abs_sensitizer_sp
        elif index_A_j[num] != -1:
            return abs_activator_sp

    diag_list = [get_block(i) for i in range(num_total_ions)]
    # eliminate None blocks. It happens if the number of states is 0 for either ion
    diag_list = [elem for elem in diag_list if elem is not None]

    absorption_matrix = scipy.sparse.block_diag(diag_list, format='csr', dtype=np.float64)
    return absorption_matrix


def _create_total_absorption_matrix(sensitizer_states: int, activator_states: int,
                                    num_energy_states: int,
                                    excitations_dict: Dict, index_S_i: List[int],
                                    index_A_j: List[int]) -> scipy.sparse.csr_matrix:
    '''Returns the total absorption matrix'''

    # list absorption matrices: a list of abs_matrix that are active
    # (multiplied by the power density)
    total_abs_matrix = scipy.sparse.csr_matrix((num_energy_states,
                                                num_energy_states), dtype=np.float64)

    # for each excitation
    for current_exc_dict in excitations_dict.values():

        # if the current excitation is not active jump to the next one
        if current_exc_dict['active'] is False:
            continue

        power_dens = current_exc_dict['power_dens']
        for num in range(len(current_exc_dict['process'])):
            abs_sensitizer = np.zeros((sensitizer_states, sensitizer_states), dtype=np.float64)
            abs_activator = np.zeros((activator_states, activator_states), dtype=np.float64)

            pump_rate = current_exc_dict['pump_rate'][num]
            degeneracy = current_exc_dict['degeneracy'][num]
            init_state = current_exc_dict['init_state'][num]
            final_state = current_exc_dict['final_state'][num]
            ion_exc = current_exc_dict['ion_exc'][num]

            if ion_exc == 'S' and sensitizer_states:
                if init_state < sensitizer_states and final_state < sensitizer_states:
                    abs_sensitizer[init_state, init_state] = -pump_rate
                    abs_sensitizer[init_state, final_state] = +degeneracy*pump_rate
                    abs_sensitizer[final_state, init_state] = +pump_rate
                    abs_sensitizer[final_state, final_state] = -degeneracy*pump_rate
                    abs_sensitizer *= power_dens  # multiply by the power density

            elif ion_exc == 'A' and activator_states:
                if init_state < activator_states and final_state < activator_states:
                    abs_activator[init_state, init_state] = -pump_rate
                    abs_activator[init_state, final_state] = +degeneracy*pump_rate
                    abs_activator[final_state, init_state] = +pump_rate
                    abs_activator[final_state, final_state] = -degeneracy*pump_rate
                    abs_activator *= power_dens  # multiply by the power density
            # create matrix with this process
            absorption_matrix = _create_absorption_matrix(abs_sensitizer, abs_activator,
                                                          index_S_i, index_A_j)
            # add it to the matrix with all processes of all active excitations
            total_abs_matrix = total_abs_matrix + absorption_matrix

    return total_abs_matrix


def _create_branching_ratios(sensitizer_states: int, activator_states: int,
                             decay_dict: Dict) -> Tuple[np.array, np.array]:
    '''Create the branching ratio matrices.'''
    # branching ratios given directly by the user
    B_sensitizer = np.zeros((sensitizer_states, sensitizer_states), dtype=np.float64)
    B_activator = np.zeros((activator_states, activator_states), dtype=np.float64)
    B_pos_value_S = decay_dict['B_pos_value_S']
    B_pos_value_A = decay_dict['B_pos_value_A']

    try:
        for pos_i, pos_f, value in B_pos_value_S:
            # discard processes with indices higher than sensitizer_states
            if pos_i < sensitizer_states and pos_f < sensitizer_states:
                B_sensitizer[pos_i, pos_f] = value
        for pos_i, pos_f, value in B_pos_value_A:
            # discard processes with indices higher than activator_states
            if pos_i < activator_states and pos_f < activator_states:
                B_activator[pos_i, pos_f] = value
    # this shouldn't happen
    except IndexError as err:  # pragma: no cover
        logging.getLogger(__name__).error('Wrong number of states!')
        logging.getLogger(__name__).error(str(err))
        raise

    # discard branching ratios to the ground state
    if sensitizer_states > 0:
        B_sensitizer[:, 0] = 0
    if activator_states > 0:
        B_activator[:, 0] = 0

    return (B_sensitizer, B_activator)

def _create_decay_vectors(sensitizer_states: int, activator_states: int,
                          decay_dict: Dict) -> Tuple[np.array, np.array]:
    '''Create the decay vectors.'''

    k_sensitizer = np.zeros((sensitizer_states, ), dtype=np.float64)
    k_activator = np.zeros((activator_states, ), dtype=np.float64)
    # list of tuples of state and decay rate
    pos_value_S = decay_dict['pos_value_S']
    pos_value_A = decay_dict['pos_value_A']
    try:
        for pos, value in pos_value_S:
            if pos < sensitizer_states:
                k_sensitizer[pos] = value
        for pos, value in pos_value_A:
            if pos < activator_states:
                k_activator[pos] = value
    # this shouldn't happen
    except IndexError as err:  # pragma: no cover
        logging.getLogger(__name__).debug('Wrong number of states!')
        raise

    return (k_sensitizer, k_activator)

#@profile
def _create_decay_matrix(sensitizer_states: int, activator_states: int, decay_dict: Dict,
                         index_S_i: List[int], index_A_j: List[int]) -> scipy.sparse.csr_matrix:
    '''Returns the decay matrix'''

    # branching ratios
    B_sensitizer, B_activator = _create_branching_ratios(sensitizer_states,
                                                         activator_states,
                                                         decay_dict)

    # decay constants
    k_sensitizer, k_activator = _create_decay_vectors(sensitizer_states,
                                                      activator_states,
                                                      decay_dict)

    # final decay matrix
    if sensitizer_states > 0:
        # add a -1 on the diagonal
        decay_sensitizer = B_sensitizer.transpose() - np.diagflat([1]*k_sensitizer.size)
        # multiply each column by its decay rate
        decay_sensitizer = decay_sensitizer*k_sensitizer.transpose()
        decay_sensitizer[0, :] = -np.sum(decay_sensitizer, 0)
    else:
        decay_sensitizer = None

    if activator_states > 0:
        # add a -1 on the diagonal
        decay_activator = B_activator.transpose() - np.diagflat([1]*k_activator.size)
        # multiply each column by its decay rate
        decay_activator = decay_activator*k_activator.transpose()
        decay_activator[0, :] = -np.sum(decay_activator, 0)
    else:
        decay_activator = None

    def get_block(num):
        '''Returns a S or A block matrix depending on what kind of ion num is'''
        if index_S_i[num] != -1:
            return decay_sensitizer
        elif index_A_j[num] != -1:
            return decay_activator

    num_total_ions = len(index_A_j)
    diag_list = (get_block(i) for i in range(num_total_ions))
    diag_list_clean = [elem for elem in diag_list if elem is not None]

    decay_matrix = scipy.sparse.block_diag(diag_list_clean, format='csr', dtype=np.float64)

    return decay_matrix


#@profile
def _create_ET_matrices(index_S_i: List[int], index_A_j: List[int], dict_ET: Dict,
                        indices_S_k: List[np.array], indices_S_l: List[np.array],
                        indices_A_k: List[np.array], indices_A_l: List[np.array],
                        dists_S_k: List[np.array], dists_S_l: List[np.array],
                        dists_A_k: List[np.array], dists_A_l: List[np.array],
                        sensitizer_states: int, activator_states: int
                       ) -> Tuple[scipy.sparse.csr_matrix, np.array]:
    '''Calculates the ET_matrix and N_indices matrices of energy transfer
       The ET_matrix has size num_interactions x num_states:
       at each column there are 4 nonzero values corresponding to the ET rate.
       Their positions in the column are at the indices of the populations affected
       by that particular ET process.
       N_indices has size 2 x num_interactions: each row corresponds to the populations
       that need to be multiplied: y(N_indices[:,0]) * y(N_indices[:,1]).

       The total ET contribution to the rate equations is then:
       ET_matrix * y(N_indices[:,0]) * y(N_indices[:,1]).
    '''
#    @profile
    def add_ET_process(index_ion: int, indices_ions: np.array, dist_ions: np.array,
                       strength: float, mult: int,
                       ii_state: int, fi_state: int, if_state: int, ff_state: int) -> None:
        ''' Adds an energy transfer process
            ii_state: initial ion, initial state
            fi_state: final ion, initial state

            if_state: initial ion, final state
            ff_state: final ion, final state
            '''

        # this tells python to use the outer uc_index variable
        # the other outer variables (x_index, N_index_X, ...) are mutable and
        # are modified without problems
        nonlocal uc_index, uc_index_indep

        indices_ions = indices_ions[indices_ions != -1]
        dist_ions = dist_ions[indices_ions != -1]

        i_vec_1 = np.repeat(np.array([index_ion+ii_state], dtype=np.uint64), len(indices_ions))
        i_vec_2 = np.repeat(np.array([index_ion+if_state], dtype=np.uint64), len(indices_ions))
        i_vec_3 = indices_ions+fi_state
        i_vec_4 = indices_ions+ff_state
        # interweave i_vec_Xs
        i_index[uc_index_indep:uc_index_indep+4*len(indices_ions)] = \
            np.ravel(np.column_stack((i_vec_1, i_vec_2, i_vec_3, i_vec_4)))

        temp = np.arange(uc_index, uc_index+len(indices_ions))
        j_index[uc_index_indep:uc_index_indep+4*len(indices_ions)] = \
            np.ravel(np.column_stack((temp, temp, temp, temp)))

        temp = dist_ions**(-mult)*strength
        v_index[uc_index_indep:uc_index_indep+4*len(indices_ions)] = \
            np.ravel(np.column_stack((-temp, temp, -temp, temp)))

        N_index_I[uc_index:uc_index+len(indices_ions)] = i_vec_1
        N_index_J[uc_index:uc_index+len(indices_ions)] = i_vec_3

        uc_index_indep += 4*len(indices_ions)
        uc_index += len(indices_ions)

    num_S_atoms = np.count_nonzero(np.array(index_S_i) != -1)
    num_A_atoms = np.count_nonzero(np.array(index_A_j) != -1)
    num_total_ions = num_S_atoms + num_A_atoms
    num_energy_states = sensitizer_states*num_S_atoms + activator_states*num_A_atoms

    if num_energy_states <= 2:
        ET_matrix = csr_matrix(np.zeros((num_energy_states, 0), dtype=np.float64))
        N_indices = np.column_stack(([], []))
        return (ET_matrix, N_indices)

    num_et_processes = 10*(sum(len(arr) for arr in indices_S_k) +
                           sum(len(arr) for arr in indices_S_l) +
                           sum(len(arr) for arr in indices_A_k) +
                           sum(len(arr) for arr in indices_A_l))
    N_index_I = np.zeros((num_et_processes,), dtype=np.uint32)
    N_index_J = np.zeros_like(N_index_I)

    uc_index = uc_index_indep = 0
    i_index = np.zeros_like(N_index_I)
    j_index = np.zeros_like(N_index_I)
    v_index = np.zeros_like(N_index_I, dtype=np.float64)
    # indices legend:
    # i, i+1 = current Yb
    # j, j+1, j+2, ... = current Tm
    # k, k+1 = Yb that interacts
    # l, l+1, l+2, ... = Tm that interacts

    # make sure the number of states for A and S are greater or equal than the processes require
    try:
        for proc_name, dict_process in dict_ET.items():
             # discard processes whose states are larger than activator states
            if not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'AA':
                if np.any(np.array(dict_process['indices']) > activator_states):
                    raise lattice.LatticeError
            elif not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'AS':
                if np.any(np.array(dict_process['indices'][::2]) > activator_states) or\
                    np.any(np.array(dict_process['indices'][1::2]) > sensitizer_states):
                    raise lattice.LatticeError
            elif not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'SS':
                if np.any(np.array(dict_process['indices']) > sensitizer_states):
                    raise lattice.LatticeError
            elif not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'SA':
                if np.any(np.array(dict_process['indices'][::2]) > activator_states) or\
                    np.any(np.array(dict_process['indices'][1::2]) > sensitizer_states):
                    raise lattice.LatticeError
    except lattice.LatticeError:
        msg = ('The number of A or S states is lower ' +
               'than required by process {}.').format(proc_name)
        logging.getLogger(__name__).error(msg)
        raise lattice.LatticeError(msg)

    num_A = num_S = 0
    for num in range(num_total_ions):
        if index_A_j[num] != -1 and activator_states != 0:  # Tm ions
            index_j = index_A_j[num]  # position of ion num on the solution vector
            # add all A-A ET processes
            for proc_name, dict_process in dict_ET.items():
                if not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'AA':
                    # reshape to (n,) from (n,1)
                    indices_l = indices_A_l[num_A].reshape((len(indices_A_l[num_A]),))
                    dists_l = dists_A_l[num_A]
                    add_ET_process(index_j, indices_l, dists_l,
                                   dict_process['value'],
                                   dict_process['mult'],
                                   *dict_process['indices'])
            # add all A-S ET processes
            for proc_name, dict_process in dict_ET.items():
                if not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'AS':
                    indices_k = indices_A_k[num_A].reshape((len(indices_A_k[num_A]),))
                    dists_k = dists_A_k[num_A]
                    add_ET_process(index_j, indices_k, dists_k,
                                   dict_process['value'],
                                   dict_process['mult'],
                                   *dict_process['indices'])
            num_A += 1
        if index_S_i[num] != -1 and sensitizer_states != 0:  # Yb ions
            index_i = index_S_i[num]  # position of ion num on the solution vector
            # add all S-S ET processes
            for proc_name, dict_process in dict_ET.items():
                if not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'SS':
                    indices_k = indices_S_k[num_S].reshape((len(indices_S_k[num_S]),))
                    dists_k = dists_S_k[num_S]
                    add_ET_process(index_i, indices_k, dists_k,
                                   dict_process['value'],
                                   dict_process['mult'],
                                   *dict_process['indices'])
            # add all S-A ET processes
            for proc_name, dict_process in dict_ET.items():
                if not np.allclose(dict_process['value'], 0.0) and dict_process['type'] == 'SA':
                    indices_l = indices_S_l[num_S].reshape((len(indices_S_l[num_S]),))
                    dists_l = dists_S_l[num_S]
                    add_ET_process(index_i, indices_l, dists_l,
                                   dict_process['value'],
                                   dict_process['mult'],
                                   *dict_process['indices'])
            num_S += 1

    # clear all extra terms
    N_index_I = N_index_I[0:uc_index]
    N_index_J = N_index_J[0:uc_index]

    i_index = i_index[0:4*(uc_index)]
    j_index = j_index[0:4*(uc_index)]
    v_index = v_index[0:4*(uc_index)]

    # create ET matrix
    ET_matrix = csr_matrix((v_index, (i_index, j_index)),
                           shape=(num_energy_states, uc_index),
                           dtype=np.float64)
#    ET_matrix.eliminate_zeros()
    N_indices = np.column_stack((N_index_I, N_index_J))

    return (ET_matrix, N_indices)


#@profile
def _calculate_jac_matrices(N_indices: np.array) -> np.array:
    '''Calculates the jacobian matrix helper data structures (non-zero values):
       N_indices has two columns and num_interactions rows
       with the index i and j of each interaction.
       jac_indices's first column is the row, second is the column,
       and third is the population index.
       The jacobian is then J_i,j = y(jac_indices_i,j)
       The size of jac_indices is 3 x (2*num_interactions)
    '''

    num_interactions = len(N_indices[:, 0])

    # calculate indices for the jacobian
    temp = np.arange(0, num_interactions, dtype=np.uint64)
    row_indices = np.ravel(np.column_stack((temp, temp)))
    col_indices = np.ravel(np.column_stack((N_indices[:, 0], N_indices[:, 1])))
    y_indices = np.ravel(np.column_stack((N_indices[:, 1], N_indices[:, 0])))

    jac_indices = np.column_stack((row_indices, col_indices, y_indices))

    return jac_indices


def get_lifetimes(cte: Dict) -> List[float]:
    '''Returns a list of all lifetimes in seconds.
       First sensitizer and then activator
    '''

    pos_value_S = cte['decay']['pos_value_S']
    pos_value_A = cte['decay']['pos_value_A']

    return [1/float(k) for num, k in pos_value_S+pos_value_A]


#@profile
def setup_microscopic_eqs(cte: Dict, gen_lattice: bool = False, full_path: str = None
                         ) -> Tuple[Dict, np.array, List[int], List[int],
                                    scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
                                    scipy.sparse.csr_matrix, np.array, np.array]:
    '''Setups all data structures necessary for the microscopic rate equations
        As arguments it gets the cte dict (that can be read from a file with settings.py)
        It returns the updated cte, initial conditions vector,
        index_Yb_i, index_Tm_j arrays that check that the ion exists at that position
        Abs, Decay, ET_matrix, and N_index matrices for the ODE solver function
        and also jac_indices for the jacobian

        gen_lattice=True will generate a lattice even if it already exists
        full_path=True will load a specific lattice from that path.
    '''
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # convert to float
    S_conc = float(cte['lattice']['S_conc'])
    A_conc = float(cte['lattice']['A_conc'])

    num_uc = cte['lattice']['N_uc']
    lattice_name = cte['lattice']['name']

    logger.info('Starting microscopic rate equations setup.')
    logger.info('Lattice: %s.', lattice_name)
    logger.info('Size: %sx%sx%s unit cells.', num_uc, num_uc, num_uc)
    logger.info('Concentrations: %.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)

    # check if data exists, otherwise create it
    logger.info('Checking data...')

    if full_path is not None:  # if the user requests a specific lattice
        filename = full_path
    else:  # pragma: no cover
        folder_path = os.path.join('latticeData', lattice_name)
        full_path = lattice.make_full_path(folder_path, num_uc, S_conc, A_conc)
        filename = full_path

    try:
        # generate the lattice in any case
        if gen_lattice:  # pragma: no cover
            logger.debug('User request to recreate lattice.')
            raise FileNotFoundError('Recalculate lattice')

        # try load the lattice data from disk
        lattice_info = _load_lattice(filename)

        # check that the number of states is correct
        if (lattice_info['sensitizer_states'] is not cte['states']['sensitizer_states'] or
                lattice_info['activator_states'] is not cte['states']['activator_states']):
            logger.info('Wrong number of states, recalculate lattice...')
            raise FileNotFoundError('Wrong number of states, recalculate lattice...')

    except OSError:
        logger.info('Creating lattice...')

        # don't show the plot
        old_no_plot = cte['no_plot']
        cte['no_plot'] = True
        # generate lattice, data will be saved to disk
        lattice.generate(cte, full_path=filename)
        cte['no_plot'] = old_no_plot

        # load data from disk
        lattice_info = _load_lattice(filename)
        logger.info('Lattice data created.')
    else:
        logger.info('Lattice data found.')

    cte['ions'] = {}
    cte['ions']['total'] = lattice_info['num_total']
    cte['ions']['sensitizers'] = lattice_info['num_sensitizers']
    cte['ions']['activators'] = lattice_info['num_activators']

    num_energy_states = cte['states']['energy_states'] = lattice_info['energy_states']
    sensitizer_states = cte['states']['sensitizer_states'] = lattice_info['sensitizer_states']
    activator_states = cte['states']['activator_states'] = lattice_info['activator_states']

    num_total_ions = cte['ions']['total']
    num_sensitizers = cte['ions']['sensitizers']
    num_activators = cte['ions']['activators']

    logger.info('Number of ions: %d, sensitizers: %d, activators: %d.',
                num_total_ions, num_sensitizers, num_activators)
    logger.info('Number of states: %d.', num_energy_states)

    logger.info('Calculating parameters...')

    # get data structures from the file
    # i: current S ion
    # j: current A ion
    # k: other S ion that interacts
    # l: other A ion that interacts

    with h5py.File(filename, mode='r') as file:
        index_S_i = list(itertools.chain.from_iterable(np.array(file['indices_S_i']).tolist()))
        index_A_j = list(itertools.chain.from_iterable(np.array(file['indices_A_j']).tolist()))

        # S interact with S
        indices_S_k = [np.array(x, dtype=np.int64) for x in file['index_S_k']]
        dists_S_k = np.array(file['dist_S_k'])

        # S interact with A
        indices_S_l = [np.array(x, dtype=np.int64) for x in file['index_S_l']]
        dists_S_l = np.array(file['dist_S_l'])

        # A interact with S
        indices_A_k = [np.array(x, dtype=np.int64) for x in file['index_A_k']]
        dists_A_k = np.array(file['dist_A_k'])

        # A interact with A
        indices_A_l = [np.array(x, dtype=np.int64) for x in file['index_A_l']]
        dists_A_l = np.array(file['dist_A_l'])

        initial_population = np.array(file['initial_population'])


    logger.info('Building matrices...')

    logger.info('Absorption and decay matrices...')
    total_abs_matrix = _create_total_absorption_matrix(sensitizer_states, activator_states,
                                                       num_energy_states, cte['excitations'],
                                                       index_S_i, index_A_j)
    decay_matrix = _create_decay_matrix(sensitizer_states, activator_states,
                                        cte['decay'], index_S_i, index_A_j)

    # ET matrices
    logger.info('Energy transfer matrices...')
    ET_matrix, N_indices = _create_ET_matrices(index_S_i, index_A_j, cte['ET'],
                                               indices_S_k, indices_S_l,
                                               indices_A_k, indices_A_l,
                                               dists_S_k, dists_S_l,
                                               dists_A_k, dists_A_l,
                                               sensitizer_states, activator_states)

    jac_indices = _calculate_jac_matrices(N_indices)

    logger.info('Number of interactions: %d.', N_indices.shape[0])

    logger.info('Setup finished. Total time: %.2fs.', time.time()-start_time)
    return (cte, initial_population, index_S_i, index_A_j,
            total_abs_matrix, decay_matrix, ET_matrix, N_indices, jac_indices)


def setup_average_eqs(cte: Dict, gen_lattice: bool = False, full_path: str = None
                     ) -> Tuple[Dict, np.array, List[int], List[int],
                                scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
                                scipy.sparse.csr_matrix, np.array, np.array]:
    '''Setups all data structures necessary for the average rate equations
        As arguments it gets the cte dict (that can be read from a file with settings.py)
        It returns the updated cte, initial conditions vector,
        index_Yb_i, index_Tm_j arrays that check that the ion exists at that position
        Abs, Decay, ET_matrix, and N_index matrices for the ODE solver function
        and also jac_indices for the jacobian

        gen_lattice=True will generate a lattice even if it already exists
        full_path=True will load a specific lattice from that path.
    '''
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # convert to float
    S_conc = float(cte['lattice']['S_conc'])
    A_conc = float(cte['lattice']['A_conc'])

    lattice_name = cte['lattice']['name']

    logger.info('Starting setup.')
    logger.info('Lattice: %s.', lattice_name)
    logger.info('Concentrations: %.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)

    cte['ions'] = {}
    num_sensitizers = cte['ions']['sensitizers'] = 1 if S_conc != 0 else 0
    num_activators = cte['ions']['activators'] = 1 if A_conc != 0 else 0
    num_total_ions = cte['ions']['total'] = num_sensitizers + num_activators

    sensitizer_states = cte['states']['sensitizer_states']
    activator_states = cte['states']['activator_states']
    num_energy_states = cte['states']['energy_states'] = (num_sensitizers*sensitizer_states +
                                                          num_activators*activator_states)
    lattice_info = {}
    lattice_info['num_total'] = num_total_ions
    lattice_info['num_activators'] = num_activators
    lattice_info['num_sensitizers'] = num_sensitizers
    # save number of states so this lattice is only used with the right settings
    lattice_info['energy_states'] = num_energy_states
    lattice_info['sensitizer_states'] = sensitizer_states
    lattice_info['activator_states'] = activator_states

    if num_total_ions == 0:
        msg = 'No ions generated, the concentrations are too small!'
        logger.error(msg)
        raise lattice.LatticeError(msg)

    # discard the results, but it does a lot of error checking
    # don't show the plot
    old_no_plot = cte['no_plot']
    cte['no_plot'] = True
    # generate lattice, data will be saved to disk
    lattice.generate(cte, no_save=True)
    cte['no_plot'] = old_no_plot

    logger.info('Number of ions: %d, sensitizers: %d, activators: %d.',
                num_total_ions, num_sensitizers, num_activators)
    logger.info('Number of states: %d.', num_energy_states)

    logger.info('Calculating parameters...')
    # list of ion types. 0=S, 1=A
    if num_sensitizers:
        if num_activators:
            lst = [0, 1]
        else:
            lst = [0]
    else:
        lst = [1]
    ion_type = np.array(lst)

    # distance array, 1 A distance
    dist_array = np.ones((num_total_ions, num_total_ions))

    (indices_S_i, indices_A_j,
     initial_population) = lattice.create_ground_states(ion_type, lattice_info)

    (indices_S_k, indices_S_l,
     indices_A_k, indices_A_l,
     dists_S_k, dists_S_l,
     dists_A_k, dists_A_l) = lattice.create_interaction_matrices(ion_type, dist_array,
                                                                 indices_S_i, indices_A_j,
                                                                 lattice_info)
    indices_S_k = [np.array(x, dtype=np.int64) for x in indices_S_k]
    dists_S_k = np.array(dists_S_k)
    indices_S_l = [np.array(x, dtype=np.int64) for x in indices_S_l]
    dists_S_l = np.array(dists_S_l)
    indices_A_k = [np.array(x, dtype=np.int64) for x in indices_A_k]
    dists_A_k = np.array(dists_A_k)
    indices_A_l = [np.array(x, dtype=np.int64) for x in indices_A_l]
    dists_A_l = np.array(dists_A_l)

    logger.info('Building matrices...')
    logger.info('Absorption and decay matrices...')
    total_abs_matrix = _create_total_absorption_matrix(sensitizer_states, activator_states,
                                                       num_energy_states, cte['excitations'],
                                                       indices_S_i, indices_A_j)
    decay_matrix = _create_decay_matrix(sensitizer_states, activator_states, cte['decay'],
                                        indices_S_i, indices_A_j)

    # ET matrices
    logger.info('Energy transfer matrices...')
    # use the avg value if present
    ET_dict = cte['ET'].copy()
    for proc_name, dict_process in ET_dict.items():
        if 'value_avg' in dict_process:
            dict_process['value'] = dict_process['value_avg']
    ET_matrix, N_indices = _create_ET_matrices(indices_S_i, indices_A_j, ET_dict,
                                               indices_S_k, indices_S_l,
                                               indices_A_k, indices_A_l,
                                               dists_S_k, dists_S_l,
                                               dists_A_k, dists_A_l,
                                               sensitizer_states, activator_states)
    # clean emtpy columns in the matrix due to energy migration
    ET_matrix = ET_matrix.toarray()
    emtpy_indices = [ind for ind in range(N_indices.shape[0]) if np.allclose(ET_matrix[:,ind], 0)]
    ET_matrix = csr_matrix(np.delete(ET_matrix, np.array(emtpy_indices), axis=1))
    N_indices = np.delete(N_indices, np.array(emtpy_indices), axis=0)

    jac_indices = _calculate_jac_matrices(N_indices)

    logger.info('Number of interactions: %d.', N_indices.shape[0])

    logger.info('Setup finished. Total time: %.2fs.', time.time()-start_time)
    return (cte, initial_population, indices_S_i, indices_A_j,
            total_abs_matrix, decay_matrix, ET_matrix, N_indices, jac_indices)



if __name__ == "__main__":
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger.debug('Called from main.')

    import simetuc.settings as settings
    cte = settings.load('config_file.cfg')
    cte['no_console'] = False
    cte['no_plot'] = False

    cte['lattice']['S_conc'] = 0.0
    cte['lattice']['A_conc'] = 0.0

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices) = setup_microscopic_eqs(cte)

    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()
