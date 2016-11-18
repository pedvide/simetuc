# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:52:23 2015

@author: Villanueva

"""

import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from ase.spacegroup import crystal


class LatticeError(Exception):
    '''The generated lattice is not valid'''
    pass


def _create_lattice(spacegroup, cell_par, num_uc, sites_pos, sites_occ):
    '''Creates the lattice with the specified parameters.
        Returns an ase.Atoms object with all atomic positions
    '''
    atoms = crystal(symbols=['Y']*len(sites_pos), basis=sites_pos,
                    spacegroup=spacegroup,
                    cellpar=cell_par, size=(num_uc, num_uc, num_uc))

    # eliminate atoms in sites with occupations less than one
    num_sites = len(sites_occ)
    rand_num = np.random.random_sample((len(atoms),))
    del atoms[[at.index for at in atoms if rand_num[at.index] > sites_occ[at.index % num_sites]]]

    return atoms


def _impurify_lattice(atoms, S_conc, A_conc):
    '''Impurifies the lattice atoms with the specified concentration of
       sensitizers and activators.
       Returns an array with the dopant positions and another with the ion type
    '''
    # convert from percentage
    S_conc = S_conc/100.0
    A_conc = A_conc/100.0

    num_atoms = len(atoms)
    # populate the lattice with the requested concentrations of sensitizers and activators
    ion_type = np.zeros((3*int(np.ceil(num_atoms*(S_conc+A_conc))),),
                        dtype=np.uint32)  # preallocate 3x more
    num_doped_atoms = 0  # number of doped ions

    # for each atom in atoms (random numbers denoted by rand_num_x):
    # if rand_num_2 <= S_conc+A_conc then we'll populate it, otherwise eliminate
    # if rand_num_3 <= S_conc then that ion is a sensitizer, otherwise is an activator
    del_list = []
    for atom in atoms:
        rand_num_2 = np.random.rand()  # doped or un doped site
        rand_num_3 = np.random.uniform(0, A_conc+S_conc)  # sensitizer or activator

        # doped site
        if rand_num_2 < (A_conc+S_conc):
            if rand_num_3 < S_conc:  # sensitizer
                ion_type[num_doped_atoms] = 0
            else:  # activator
                ion_type[num_doped_atoms] = 1
            num_doped_atoms += 1
        else:
            del_list.append(atom.index)
    # delete un-doped positions
    del atoms[del_list]

    # eliminate extra zeros from preallocation
    ion_type = ion_type[:num_doped_atoms]

    return ion_type


def _calculate_distances(atoms, min_im_conv=True):
    '''Calculates the distances between each pair of ions
       By defaul it uses the minimum image convention
       It returns a square array 'dist_array' with the distances
       between ions i and j in dist_array[i][j]
    '''
    # calculate distances between the points in A
    # D = atoms.get_all_distances(mic=True, simple=True) # eats up ALL the ram N=30 MAX
    num_atoms = len(atoms)
    dist_array = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    for i in range(num_atoms):
        dist_array[i, i:num_atoms] = atoms.get_distances(i, range(i, num_atoms), mic=min_im_conv)

    # get the distance along the c axis divided by two
    # dmaxReal = atoms.cell[2][2]/2;
    # dist_array[dist_array > dmaxReal] = 0 # discard distances longer than dmax
    dist_array = dist_array + dist_array.T  # make symmetrical
#    Ds_mic = get_all_distances(atoms)
    # Ds = csr_matrix(dist_array)

    return dist_array


def _create_ground_states(ion_type, lattice_info):
    '''Returns two arrays with the position of the sensitizers' and activators'
       ground states in the total simulation population index.
       It also returns an array with the initial populations, that is
       with all populations set to zero except those of the ground states
    '''
    num_atoms = lattice_info['num_total']
    num_states = lattice_info['energy_states']
    num_A_states = lattice_info['activator_states']
    num_S_states = lattice_info['sensitizer_states']

    # index of ion i (Yb) or j (Tm). It links the ion number with the position
    # of its GS in the solution vector
    index_S_i = -1*np.ones((num_atoms, 1), dtype=np.int64)  # position of the GS
    index_A_j = -1*np.ones((num_atoms, 1), dtype=np.int64)  # position of the GS

    initial_population = np.zeros((num_states), dtype=np.uint64)

    # for each S or A: fill the index of its GS
    gs_state_counter = 0
    for ion_index in range(num_atoms):
        # activator
        if ion_type[ion_index] != 0:
            # if there are energy states, else it continues to the next ion
            if num_A_states != 0:
                index_A_j[ion_index] = gs_state_counter
                gs_state_counter += num_A_states
        # sensitizer
        else:
            # if there are energy states, else it continues to the next ion
            if num_S_states != 0:
                index_S_i[ion_index] = gs_state_counter
                gs_state_counter += num_S_states

    # populate GS with 1
    initial_population[index_S_i[index_S_i != -1]] = 1
    initial_population[index_A_j[index_A_j != -1]] = 1

    return (index_S_i, index_A_j, initial_population)


def _create_interaction_matrices(ion_type, dist_array, index_S_i, index_A_j,
                                 lattice_info):
    '''It returns the interaction lists distances:
        index_S_k = position of the GS of S ions that interact with S ions
        index_S_l = position of the GS of A ions that interact with S ions
        index_A_k = position of the GS of S ions that interact with A ions
        index_A_l = position of the GS of A ions that interact with A ions
        The same for the dist_X_y arrays
    '''
    num_atoms = lattice_info['num_total']
    num_A_states = lattice_info['activator_states']
    num_S_states = lattice_info['sensitizer_states']

    # for the ith ion, we find the indices of the energy levels of the ions it interacts with
    index_S_k = []  # position of the GS of S ions that interact with S ions
    dist_S_k = []

    index_S_l = []  # position of the GS of A ions that interact with S ions
    dist_S_l = []

    index_A_k = []  # position of the GS of S ions that interact with A ions
    dist_A_k = []

    index_A_l = []  # position of the GS of A ions that interact with A ions
    dist_A_l = []

    # fill the matrices of interactions
    for i in range(num_atoms):
        # find the ions that this ion interacts with
        dist_ions = dist_array[i, :]  # distances to ions
        ions_inter = np.nonzero(dist_ions)[0]  # indices of ions that interact (non-zero distance)

        if ion_type[i] != 0:  # activator
            # if there are energy states, else it continues to the next ion
            if num_A_states != 0:
                # we need to find out which ions we interact with are A or S
                A_inter = ion_type[ions_inter] != 0  # indices of A ions that interact
                if A_inter.any():
                    index_A_l.append(index_A_j[list(ions_inter[A_inter])])
                    dist_A_l.append(dist_ions[ions_inter[A_inter]])
                else:  # this only happens if there's only one activator ion!
                    index_A_l.append([])
                    dist_A_l.append([])
                S_inter = np.logical_not(A_inter)  # indices of S ions that interact
                if S_inter.any():
                    index_A_k.append(index_S_i[list(ions_inter[S_inter])])
                    dist_A_k.append(dist_ions[ions_inter[S_inter]])
                else:
                    index_A_k.append([])
                    dist_A_k.append([])
        else:  # Sensitizer
            # if there are energy states, else it continues to the next ion
            if num_S_states != 0:
                A_inter = ion_type[ions_inter] != 0  # indices of A ions that interact
                if A_inter.any():
                    index_S_l.append(index_A_j[list(ions_inter[A_inter])])
                    dist_S_l.append(dist_ions[ions_inter[A_inter]])
                else:
                    index_S_l.append([])
                    dist_S_l.append([])
                S_inter = np.logical_not(A_inter)  # indices of S ions that interact
                if S_inter.any():
                    index_S_k.append(index_S_i[list(ions_inter[S_inter])])
                    dist_S_k.append(dist_ions[ions_inter[S_inter]])
                else:  # this only happens if there's only one sensitizer ion!
                    index_S_k.append([])
                    dist_S_k.append([])

    return (index_S_k, index_S_l, index_A_k, index_A_l,
            dist_S_k, dist_S_l, dist_A_k, dist_A_l)


def _plot_lattice(doped_lattice, ion_type):
    from mpl_toolkits.mplot3d import proj3d

    def orthogonal_proj(zfront, zback):  # pragma: no cover
        '''
        This code sets the 3d projection to orthogonal so the plots are easier to see
        http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d
        '''
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, a, b],
                         [0, 0, -0.0001, zback]])
    proj3d.persp_transformation = orthogonal_proj
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    # axis=Axes3D(fig)
    colors = ['b' if ion else 'r' for ion in ion_type]

    axis.scatter(doped_lattice[:, 0], doped_lattice[:, 1],
                 doped_lattice[:, 2], c=colors, marker='o')
    axis.set_xlabel('X (Å)')
    axis.set_ylabel('Y (Å)')
    axis.set_zlabel('Z (Å)')
    plt.axis('square')


# @profile
def generate(cte, min_im_conv=True):
    '''
    Generates a list of (x,y,z) ion positions and a list with the type of ion (S or A)
    for a lattice with N unit cells and the given concentrations (in percentage) of S and A.
    The results are saved in a file called lattice_name/data_Xuc_YS_ZA, where
    lattice_name = is the user-defined name of the lattice passed in cte
    X = number of unit cells
    Y = concentration of S
    Z = concentration of A
    min_im_conv=True uses the miminum image convention when calculating distances
    '''
    logger = logging.getLogger(__name__)

    # show plots
    plot_toggle = not cte['no_plot']

    num_uc = cte['lattice']['N_uc']
    S_conc = float(cte['lattice']['S_conc'])
    A_conc = float(cte['lattice']['A_conc'])
    lattice_name = cte['lattice']['name']

    if num_uc <= 0:
        logger.error('Wrong number of unit cells: %d.', num_uc)
        logger.error('It must be a positive integer.')
        raise LatticeError('Wrong number of unit cells.')

    # if the concentrations are not in the correct range
    if not ((0.0 <= S_conc <= 100.0) and (0.0 <= A_conc <= 100.0) and
            (0 <= S_conc+A_conc <= 100.0)):
        logger.error('Wrong ion concentrations:' +
                     '%.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)
        logger.error('They must be between 0% and 100%, and their sum too.')
        raise LatticeError('Wrong ion concentrations.')

    start_time = time.time()

    logger.info('Generating lattice %s...', lattice_name)
    logger.info('Size: %dx%dx%d unit cells.', num_uc, num_uc, num_uc)
    logger.info('Concentrations: %.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)

    # create a lattice of the given type
    # it would be more efficient to directly create a doped lattice,
    # i.e.: without creating un-doped atoms first
    # however, this is very fast anyways
    atoms = _create_lattice(cte['lattice']['spacegroup'], cte['lattice']['cell_par'],
                            num_uc, cte['lattice']['sites_pos'], cte['lattice']['sites_occ'])
    num_atoms = len(atoms)
    logger.info('Total number of atoms: %d', num_atoms)

    # modify a lattice to get the concentration of S and A ions
    # it deletes all non-doped ions from atoms and returns an array with ions types
    ion_type = _impurify_lattice(atoms, S_conc, A_conc)
    doped_lattice = atoms.positions
    # number of sensitizers and activators
    num_doped_atoms = len(atoms)
    num_A = np.count_nonzero(ion_type)
    num_S = num_doped_atoms-num_A

    if num_doped_atoms == 0:
        logger.error('No doped ions generated, the lattice or' +
                     ' the concentrations are too small!')
        raise LatticeError('No doped ions generated, the lattice or' +
                           ' the concentrations are too small!')

    logger.info('Total number of S+A: %d, (%.2f%%).',
                num_doped_atoms, num_doped_atoms/num_atoms*100)
    logger.info('Number of sensitizers (percentage): %d (%.2f%%).',
                num_S, num_S/num_atoms*100)
    logger.info('Number of activators (percentage): %d (%.2f%%).',
                num_A, num_A/num_atoms*100)

    elapsed_time = time.time()-start_time
    formatted_time = time.strftime("%Mm %Ss", time.localtime(elapsed_time))
    logger.info('Time to generate and populate the lattice: %s.', formatted_time)

    logger.info('Calculating distances...')
    dist_time = time.time()

    dist_array = _calculate_distances(atoms, min_im_conv=min_im_conv)

    elapsed_time = time.time()-dist_time
    formatted_time = time.strftime("%Mm %Ss", time.localtime(elapsed_time))
    logger.info('Time to calculate distances: %s.', formatted_time)

    logger.info('Calculating parameters...')

    num_S_states = cte['states']['sensitizer_states']
    num_A_states = cte['states']['activator_states']
    num_states = num_S_states*num_S + num_A_states*num_A

    # number of ions
    lattice_info = {}
    lattice_info['num_total'] = num_doped_atoms
    lattice_info['num_activators'] = num_A
    lattice_info['num_sensitizers'] = num_S
    # save number of states so this lattice is only used with the right settings
    lattice_info['energy_states'] = num_states
    lattice_info['sensitizer_states'] = num_S_states
    lattice_info['activator_states'] = num_A_states

    if num_states == 0:
        logger.error('The number of energy states is zero!')
        raise LatticeError('The number of energy states is zero!')

    index_S_i, index_A_j, initial_population = _create_ground_states(ion_type, lattice_info)

    (index_S_k, index_S_l,
     index_A_k, index_A_l,
     dist_S_k, dist_S_l,
     dist_A_k, dist_A_l) = _create_interaction_matrices(ion_type, dist_array,
                                                        index_S_i, index_A_j,
                                                        lattice_info)

    logger.info('Saving data...')

    if 'test' in lattice_name:  # save in test folder
        folder = 'test/'
    else:  # pragma: no cover
        folder = 'latticeData/'

    # check if folder exists
    os.makedirs(folder+lattice_name, exist_ok=True)
    filename = folder + '{}/data_{}uc_{}S_{}A'.format(lattice_name, num_uc,
                                                      S_conc, A_conc)
    np.savez(filename, dist_array=dist_array, ion_type=ion_type,
             doped_lattice=doped_lattice,
             initial_population=initial_population, lattice_info=lattice_info,
             index_S_i=index_S_i, index_A_j=index_A_j,
             index_S_k=index_S_k, dist_S_k=dist_S_k,
             index_S_l=index_S_l, dist_S_l=dist_S_l,
             index_A_k=index_A_k, dist_A_k=dist_A_k,
             index_A_l=index_A_l, dist_A_l=dist_A_l)

    # plot lattice
    if plot_toggle:
        _plot_lattice(doped_lattice, ion_type)

    elapsed_time = time.time()-start_time
    formatted_time = time.strftime("%Mm %Ss", time.localtime(elapsed_time))
    logger.info('Generating lattice finished. Total time: %s.', formatted_time)

    return (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
            index_S_i, index_A_j,
            index_S_k, dist_S_k,
            index_S_l, dist_S_l,
            index_A_k, dist_A_k,
            index_A_l, dist_A_l)


#if __name__ == "__main__":
#    logger = logging.getLogger()
#    logging.basicConfig(level=logging.INFO,
#                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#
#    logger.debug('Called from main.')
#
#    import simetuc.settings as settings
#    cte = settings.load('config_file.txt')
#    cte['no_console'] = False
#    cte['no_plot'] = False
#
#    cte['lattice']['S_conc'] = 25
#    cte['lattice']['A_conc'] = 0.3
#    cte['lattice']['N_uc'] = 5
##    cte['states']['sensitizer_states'] = 0
##    cte['states']['activator_states'] = 4
#
#    (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
#     index_S_i, index_A_j,
#     index_S_k, dist_S_k,
#     index_S_l, dist_S_l,
#     index_A_k, dist_A_k,
#     index_A_l, dist_A_l) = generate(cte)
