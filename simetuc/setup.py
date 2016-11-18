# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:07:21 2015

@author: Villanueva
"""
# pylint: disable=E1101

import time
import itertools
import logging

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

import simetuc.lattice as lattice


def _load_lattice(filename):
    '''Loads the filename and returns it along with its associanted lattice_info
        Exceptions aren't handled by this function
    '''

    # try load the lattice data
    npzfile = np.load(filename)

    # dict with the total number of ions, sensitizers, activators,
    # number of S states, A states and total number of states
    lattice_info = npzfile['lattice_info'][()]

    return (npzfile, lattice_info)


#@profile
def _create_absorption_matrix(abs_sensitizer, abs_activator, index_S_i, index_A_j):
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

#    # list of data, row_ind, col_ind
#    data = np.zeros((4*num_total_ions,), dtype=np.float32)
#    col_ind = np.zeros_like(data, dtype=np.uint32)
#    row_ptr = np.zeros_like(col_ind)
#    index = 0
#    for block in diag_list:
#        block_len = block.shape[0]
#        if len(block.data) != 0:
#            print(block_len, len(block.data))
#            print(index, index+len(block.data))
#            print(col_ind[index:index+len(block.data)])
#            col_ind[index:index+len(block.data)] = block.indices+index
#            row_ptr[index:index+len(block.indptr)] = block.indptr
#            data[index:index+len(block.data)] = block.data
#        index += block_len
#
##    data = data[:index]
##    col_ind = col_ind[:index]
##    row_ptr = row_ptr[:index]
#
#    absorption_matrix_good = scipy.sparse.block_diag(diag_list, format='csr')
#    data_good = absorption_matrix_good.data
#    col_ind_good = absorption_matrix_good.indices
#    row_ptr_good = absorption_matrix_good.indptr
#
#    absorption_matrix = scipy.sparse.csr_matrix((data, col_ind, row_ptr),
#                                                shape=(total_states, total_states))

    return absorption_matrix


def _setup_absorption(cte, index_S_i, index_A_j):

    num_energy_states = cte['states']['energy_states']
    sensitizer_states = cte['states']['sensitizer_states']
    activator_states = cte['states']['activator_states']

    # list absorption matrices: a list of abs_matrix that are active
    # (multiplied by the power density)
    total_abs_matrix = scipy.sparse.csr_matrix((num_energy_states,
                                                num_energy_states), dtype=np.float64)

    # for each excitation
    for current_exc_dict in cte['excitations'].values():

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

            if ion_exc == 'S' and cte['states']['sensitizer_states']:
                if init_state < sensitizer_states and final_state < sensitizer_states:
                    abs_sensitizer[init_state, init_state] = -pump_rate
                    abs_sensitizer[init_state, final_state] = +degeneracy*pump_rate
                    abs_sensitizer[final_state, init_state] = +pump_rate
                    abs_sensitizer[final_state, final_state] = -degeneracy*pump_rate
                    abs_sensitizer *= power_dens  # multiply by the power density

            elif ion_exc == 'A' and cte['states']['activator_states']:
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


#@profile
def _create_decay_matrix(B_sensitizer, B_activator,
                         k_sensitizer, k_activator,
                         index_S_i, index_A_j):

    num_total_ions = len(index_A_j)

    activator_states = len(k_activator)
    sensitizer_states = len(k_sensitizer)

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

    diag_list = [get_block(i) for i in range(num_total_ions)]
    diag_list = [elem for elem in diag_list if elem is not None]

    decay_matrix = scipy.sparse.block_diag(diag_list, format='csr', dtype=np.float64)

    return decay_matrix


#@profile
def _create_ET_matrices(index_S_i, index_A_j, dict_ET,
                        indices_S_k, indices_S_l, indices_A_k, indices_A_l,
                        dists_S_k, dists_S_l, dists_A_k, dists_A_l,
                        sensitizer_states, activator_states):
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
    def add_ET_process(index_ion, indices_ions, dist_ions,
                       strength, mult, ii_state, fi_state, if_state, ff_state):
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

        return

    num_total_ions = len(index_A_j)

    temp1 = np.array(index_S_i)
    temp2 = np.array(index_A_j)
    num_energy_states = (sensitizer_states*len(temp1[temp1 != -1]) +
                         activator_states*len(temp2[temp2 != -1]))

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
    num_A = num_S = 0
    for num in range(num_total_ions):
        if index_A_j[num] != -1 and activator_states != 0:  # Tm ions
            index_j = index_A_j[num]  # position of ion num on the solution vector

            # add all A-A ET processes
            for process in dict_ET:
                if dict_ET[process]['value'] != 0 and dict_ET[process]['type'] == 'AA':
                    # reshape to (n,) from (n,1)
                    indices_l = indices_A_l[num_A].reshape((len(indices_A_l[num_A]),))
                    dists_l = dists_A_l[num_A]

                    # discard processes whose states are larger than activator states
                    if np.any(np.array(dict_ET[process]['indices']) > activator_states):
                        continue
                    add_ET_process(index_j, indices_l, dists_l,
                                   dict_ET[process]['value'],
                                   dict_ET[process]['mult'],
                                   *dict_ET[process]['indices'])
            # add all A-S ET processes
            if sensitizer_states != 0:
                for process in dict_ET:
                    if dict_ET[process]['value'] != 0 and dict_ET[process]['type'] == 'AS':
                        indices_k = indices_A_k[num_A].reshape((len(indices_A_k[num_A]),))
                        dists_k = dists_A_k[num_A]
                        # discard processes whose states are larger than the number of states
                        if np.any(np.array(dict_ET[process]['indices'][::2]) > activator_states) or\
                            np.any(np.array(dict_ET[process]['indices'][1::2]) > sensitizer_states):
                            continue
                        add_ET_process(index_j, indices_k, dists_k,
                                       dict_ET[process]['value'],
                                       dict_ET[process]['mult'],
                                       *dict_ET[process]['indices'])
            num_A += 1
        if index_S_i[num] != -1 and sensitizer_states != 0:  # Yb ions
            index_i = index_S_i[num]  # position of ion num on the solution vector

            # add all S-S ET processes
            for process in dict_ET:
                if dict_ET[process]['value'] != 0 and dict_ET[process]['type'] == 'SS':
                    indices_k = indices_S_k[num_S].reshape((len(indices_S_k[num_S]),))
                    dists_k = dists_S_k[num_S]
                    # discard processes whose states are larger than activator states
                    if np.any(np.array(dict_ET[process]['indices']) > sensitizer_states):
                        continue
                    add_ET_process(index_i, indices_k, dists_k,
                                   dict_ET[process]['value'],
                                   dict_ET[process]['mult'],
                                   *dict_ET[process]['indices'])
            # add all S-A ET processes
            if activator_states != 0:
                for process in dict_ET:
                    if dict_ET[process]['value'] != 0 and dict_ET[process]['type'] == 'SA':
                        indices_l = indices_S_l[num_S].reshape((len(indices_S_l[num_S]),))
                        dists_l = dists_S_l[num_S]
                        # discard processes whose states are larger than the number of states
                        if np.any(np.array(dict_ET[process]['indices'][::2]) > activator_states) or\
                            np.any(np.array(dict_ET[process]['indices'][1::2]) > sensitizer_states):
                            continue
                        add_ET_process(index_i, indices_l, dists_l,
                                       dict_ET[process]['value'],
                                       dict_ET[process]['mult'],
                                       *dict_ET[process]['indices'])
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
    N_indices = np.column_stack((N_index_I, N_index_J))

    return (ET_matrix, N_indices)


#@profile
def _calculate_jac_matrices(N_indices):
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


def get_lifetimes(cte):
    '''Returns a list of all lifetimes in seconds.
       First sensitizer and then activator
    '''

    pos_value_S = cte['decay']['pos_value_S']
    pos_value_A = cte['decay']['pos_value_A']

    return [1/float(k) for num, k in pos_value_S+pos_value_A]


#@profile
def precalculate(cte, gen_lattice=False, test_filename=None):
    '''Setups all data structures necessary for simulations.py to work
        As arguments it gets the cte dict (that can be read from a file with settings.py)
        It returns the updated cte, initial conditions vector,
        index_Yb_i, index_Tm_j arrays that check that the ion exists at that position
        Abs, Decay, ET_matrix, and N_index matrices for the ODE solver function
        and also jac_indices for the jacobian

        gen_lattice=True will generate a lattice even if it already exists
        test_filename=True will load the test lattice
    '''
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # convert to float
    S_conc = float(cte['lattice']['S_conc'])
    A_conc = float(cte['lattice']['A_conc'])

    num_uc = cte['lattice']['N_uc']
    lattice_name = cte['lattice']['name']

    logger.info('Starting setup.')
    logger.info('Lattice: %s.', lattice_name)
    logger.info('Size: %sx%sx%s unit cells.', num_uc, num_uc, num_uc)
    logger.info('Concentrations: %.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)

    # check if data exists, otherwise create it
    logger.info('Checking data...')

    if test_filename is not None:  # if the user requests a test lattice
        filename = test_filename
    else:  # pragma: no cover
        filename = ('latticeData/' + lattice_name + '/' +
                    'data_{}uc_{}S_{}A.npz'.format(num_uc, S_conc, A_conc))

    try:
        # generate the lattice in any case
        if gen_lattice:  # pragma: no cover
            logger.debug('User request to recreate lattice.')
            raise FileNotFoundError('Recalculate lattice')

        # try load the lattice data from disk
        npzfile, lattice_info = _load_lattice(filename)

        # check that the number of states is correct
        if (lattice_info['sensitizer_states'] is not cte['states']['sensitizer_states'] or
                lattice_info['activator_states'] is not cte['states']['activator_states']):
            logger.info('Wrong number of states, recalculate lattice...')
            raise FileNotFoundError('Wrong number of states, recalculate lattice...')

    except FileNotFoundError:
        logger.info('Creating lattice...')

        # don't show the plot
        old_no_plot = cte['no_plot']
        cte['no_plot'] = True
        # generate lattice, data will be saved to disk
        lattice.generate(cte)
        cte['no_plot'] = old_no_plot

        # load data from disk
        npzfile, lattice_info = _load_lattice(filename)

        logger.info('Lattice data created.')
    else:
        logger.info('Lattice data found.')

    cte['ions'] = {}
    cte['ions']['total'] = lattice_info['num_total']
    cte['ions']['sensitizers'] = lattice_info['num_sensitizers']
    cte['ions']['activators'] = lattice_info['num_activators']

    cte['states']['energy_states'] = lattice_info['energy_states']
    sensitizer_states = cte['states']['sensitizer_states'] = lattice_info['sensitizer_states']
    activator_states = cte['states']['activator_states'] = lattice_info['activator_states']

    num_total_ions = cte['ions']['total']
    num_sensitizers = cte['ions']['sensitizers']
    num_activators = cte['ions']['activators']
    num_energy_states = cte['states']['energy_states']

    logger.info('Number of ions: %d, sensitizers: %d, activators: %d.',
                num_total_ions, num_sensitizers, num_activators)
    logger.info('Number of states: %d.', num_energy_states)

    logger.info('Calculating parameters...')

    # get data structures from the file
    # i: current S ion
    # j: current A ion
    # k: other S ion that interacts
    # l: other A ion that interacts

    # index of ion i (Yb) or j (Tm). It links the ion number with the position
    # of its GS in the solution vector
    # completely flatten the lists
    index_S_i = list(itertools.chain.from_iterable(npzfile['index_S_i'].tolist()))
    index_A_j = list(itertools.chain.from_iterable(npzfile['index_A_j'].tolist()))

    # for the ith ion, we find the indices of the energy levels of the ions it interacts with
    # indices and distances from S number i to another S number k
    indices_S_k = [np.array(x, dtype=np.int64) for x in npzfile['index_S_k']]  # list of arrays
    dists_S_k = npzfile['dist_S_k']

    # indices and distances from S number i to an A number l
    indices_S_l = [np.array(x, dtype=np.int64) for x in npzfile['index_S_l']]  # list of arrays
    dists_S_l = npzfile['dist_S_l']

    # indices and distances from A number j to an S number k
    indices_A_k = [np.array(x, dtype=np.int64) for x in npzfile['index_A_k']]  # list of arrays
    dists_A_k = npzfile['dist_A_k']

    # indices and distances from A number j to an A number l
    indices_A_l = [np.array(x, dtype=np.int64) for x in npzfile['index_A_l']]  # list of arrays
    dists_A_l = npzfile['dist_A_l']

    initial_population = npzfile['initial_population']

    # build rate equations absorption and decay matrices
    logger.info('Building matrices...')

    logger.info('Absorption and decay matrices...')

    total_abs_matrix = _setup_absorption(cte, index_S_i, index_A_j)

    # decay matrix
    # branching ratios given directly by the user
    B_sensitizer = np.zeros((sensitizer_states, sensitizer_states), dtype=np.float64)
    B_activator = np.zeros((activator_states, activator_states), dtype=np.float64)
    B_pos_value_S = cte['decay']['B_pos_value_S']
    B_pos_value_A = cte['decay']['B_pos_value_A']

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
        logger.error('Wrong number of states!')
        logger.error(err)
        raise

    # discard branching ratios to the ground state
    if sensitizer_states > 0:
        B_sensitizer[:, 0] = 0
    if activator_states > 0:
        B_activator[:, 0] = 0

    # decay constants
    k_sensitizer = np.zeros((sensitizer_states, ), dtype=np.float64)
    k_activator = np.zeros((activator_states, ), dtype=np.float64)
    # list of tuples of state and decay rate
    pos_value_S = cte['decay']['pos_value_S']
    pos_value_A = cte['decay']['pos_value_A']
    try:
        for pos, value in pos_value_S:
            if pos < sensitizer_states:
                k_sensitizer[pos] = value
        for pos, value in pos_value_A:
            if pos < activator_states:
                k_activator[pos] = value
    # this shouldn't happen
    except IndexError as err:  # pragma: no cover
        logger.debug('Wrong number of states!')
        raise

    decay_matrix = _create_decay_matrix(B_sensitizer, B_activator,
                                        k_sensitizer, k_activator,
                                        index_S_i, index_A_j)

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


#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#    logger.debug('Called from main.')
#
#    import simetuc.settings as settings
#    cte = settings.load('test/test_settings/test_standard_config.txt')
#    cte['no_console'] = False
#    cte['no_plot'] = False
#
#
#    (cte, initial_population, index_S_i, index_A_j,
#     total_abs_matrix, decay_matrix, UC_matrix,
#     N_indices, jac_indices) = precalculate(cte, test_filename='test/test_setup/data_2S_2A.npz')
#
#    UC_matrix = UC_matrix.toarray()
#    total_abs_matrix = total_abs_matrix.toarray()
#    decay_matrix = decay_matrix.toarray()
