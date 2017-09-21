# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:52:23 2015

@author: Villanueva

"""

import os
import time
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import h5py
import ruamel.yaml as yaml
from tqdm import tqdm

import ase
from ase.spacegroup import crystal

from simetuc.util import ConfigError, log_exceptions_warnings
import simetuc.plotter as plotter
import simetuc.settings as settings
from simetuc.settings import SettingsValueError, SettingsExtraValueWarning
import simetuc.settings_config as configs


class LatticeError(Exception):
    '''The generated lattice is not valid'''
    pass

@log_exceptions_warnings(ignore_warns=SettingsExtraValueWarning)
def _check_lattice_settings(cte: settings.Settings) -> None:
    '''Checks that the settings for the lattice are correct.'''

    # parse the settings to catch any errors due to wrong magnitude of a setting
    try:
        # validate lattice settings
        settings_lattice = configs.settings['lattice']
        settings_lattice.validate(cte.lattice)
        cte.lattice = settings._parse_lattice(cte)
        cte_lattice = cte.lattice
    except (SettingsValueError, ConfigError) as err:
        raise LatticeError('Wrong lattice settings.') from err

    # sites pos and occs are always lists of lists and lists, respectively
    sites_pos = cte_lattice['sites_pos']
    cte_lattice['sites_pos'] = sites_pos if isinstance(sites_pos[0], (list, tuple)) else [sites_pos]
    sites_occ = cte_lattice['sites_occ']
    cte_lattice['sites_occ'] = sites_occ if isinstance(sites_occ, (list, tuple)) else [sites_occ]

    if not len(cte_lattice['sites_pos']) == len(cte_lattice['sites_occ']):
        msg = 'The number of sites must be the same in sites_pos and sites_occ.'
        raise LatticeError(msg)

    S_conc = float(cte.lattice['S_conc'])
    A_conc = float(cte.lattice['A_conc'])

    # if the concentrations are not in the correct range
    if not (0 <= S_conc+A_conc <= 100.0):
        msg = 'Wrong ion concentrations: {:.2f}% Sensitizer, {:.2f}% Activator.'.format(S_conc, A_conc)
        msg += ' Their sum must be between 0% and 100%.'
        raise LatticeError(msg)

    # at least a state must exist
    num_S_states = cte.states['sensitizer_states']
    num_A_states = cte.states['activator_states']
    if (S_conc != 0 and num_S_states == 0) or (A_conc != 0 and num_A_states == 0):
        raise LatticeError('The number of states of each ion cannot be zero.')



def _create_lattice(spacegroup: Union[int, str], cell_par: List[float], num_uc: int,
                    sites_pos: List[float], sites_occ: List[float]) -> ase.Atoms:
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


def _make_spherical(atoms: ase.Atoms, radius: float) -> ase.Atoms:
    '''Makes the lattice spherical with the given radius by deleting all atoms outside.
       It also places the center at 0,0,0'''

    # find center of the lattice
    max_pos = np.max(atoms.get_positions(), axis=0)
    min_pos = np.min(atoms.get_positions(), axis=0)
    center = (max_pos - min_pos)/2 + min_pos

    # delete ions outside
    del atoms[[at.index for at in atoms if np.linalg.norm((at.position-center)) > radius]]
    # change positions so the center is at 0,0,0
    for atom in atoms:
        atom.position -= center

    return atoms

def _impurify_lattice(atoms: ase.Atoms, S_conc: float, A_conc: float) -> np.array:
    '''Impurifies the lattice atoms with the specified concentration of
       sensitizers and activators (in %).
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


def _calculate_distances(atoms: ase.Atoms, min_im_conv: bool = True) -> np.array:
    '''Calculates the distances between each pair of ions
       By defaul it uses the minimum image convention
       It returns a square array 'dist_array' with the distances
       between ions i and j in dist_array[i][j]
    '''
    # calculate distances between the points in A
    # D = atoms.get_all_distances(mic=True, simple=True) # eats up ALL the ram N=30 MAX

    # TODO: parallelize

    num_atoms = len(atoms)
    dist_array = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    for i in tqdm(range(num_atoms), unit='atoms', total=num_atoms, desc='Calculating distances'):
        dist_array[i, i:num_atoms] = atoms.get_distances(i, range(i, num_atoms), mic=min_im_conv)

    # get the distance along the c axis divided by two
    # dmaxReal = atoms.cell[2][2]/2;
    # dist_array[dist_array > dmaxReal] = 0 # discard distances longer than dmax
    dist_array = dist_array + dist_array.T  # make symmetrical
#    Ds_mic = get_all_distances(atoms)
    # Ds = csr_matrix(dist_array)

    return dist_array


def create_ground_states(ion_type: np.array,
                         lattice_info: Dict) -> Tuple[np.array, np.array, np.array]:
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


def create_interaction_matrices(ion_type: np.array, dist_array: np.array,
                                index_S_i: np.array, index_A_j: np.array,
                                lattice_info: dict) -> Tuple[List[np.array], List[np.array],
                                                             List[np.array], List[np.array],
                                                             List[np.array], List[np.array],
                                                             List[np.array], List[np.array]]:
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


def make_full_path(folder_path: str, num_uc: int,
                   S_conc: float, A_conc: float, radius: float = None) -> str: # pragma: no cover
    '''Makes the full path to a lattice in the folder.'''
    if radius is not None:
        filename = 'data_{}r_{}S_{}A.hdf5'.format(float(radius), float(S_conc), float(A_conc))
    else:
        filename = 'data_{}uc_{}S_{}A.hdf5'.format(int(num_uc), float(S_conc), float(A_conc))
    full_path = os.path.join(folder_path, filename)
    return full_path


# @profile
@log_exceptions_warnings
def generate(cte: settings.Settings, min_im_conv: bool = True,
             full_path: str = None, no_save: bool = False) -> Tuple:
    '''
    Generates a list of (x,y,z) ion positions and a list with the type of ion (S or A)
    for a lattice with N unit cells and the given concentrations (in percentage) of S and A.
    The results are saved in a file called lattice_name/data_Xuc_YS_ZA.hdf5, where
    lattice_name = is the user-defined name of the lattice passed in cte
    X = number of unit cells
    Y = concentration of S
    Z = concentration of A
    min_im_conv=True uses the miminum image convention when calculating distances
    full_path: will use that path to save the lattice
    no_save: don't save the data, just return it'
    '''
    logger = logging.getLogger(__name__)

    # show plots
    plot_toggle = not cte['no_plot']

    _check_lattice_settings(cte)

    num_uc = cte.lattice['N_uc']
    S_conc = float(cte.lattice['S_conc'])
    A_conc = float(cte.lattice['A_conc'])
    lattice_name = cte.lattice['name']
    radius = cte.lattice.get('radius', None)

    start_time = time.time()

    logger.info('Generating lattice %s...', lattice_name)
    if radius is None:
        logger.info('Size: %dx%dx%d unit cells.', num_uc, num_uc, num_uc)
    else:
        logger.info('Size: %.1f A.', radius)
    logger.info('Concentrations: %.2f%% Sensitizer, %.2f%% Activator.', S_conc, A_conc)

    # create a lattice of the given type
    # it would be more efficient to directly create a doped lattice,
    # i.e.: without creating un-doped atoms first
    # however, this is very fast anyways
    atoms = _create_lattice(cte.lattice['spacegroup'], cte.lattice['cell_par'],
                            num_uc, cte.lattice['sites_pos'], cte.lattice['sites_occ'])
    num_atoms = len(atoms)
    logger.info('Total number of atoms: %d', num_atoms)

    # make nanoparticle of a given radius
    if radius is not None:
        atoms = _make_spherical(atoms, radius)
        min_im_conv = False

    # modify a lattice to get the concentration of S and A ions
    # it deletes all non-doped ions from atoms and returns an array with ions types
    ion_type = _impurify_lattice(atoms, S_conc, A_conc)
    doped_lattice = atoms.positions
    # number of sensitizers and activators
    num_doped_atoms = len(atoms)
    num_A = np.count_nonzero(ion_type)
    num_S = num_doped_atoms-num_A
    num_S_states = cte.states['sensitizer_states']
    num_A_states = cte.states['activator_states']
    num_states = num_S_states*num_S + num_A_states*num_A

    if num_doped_atoms == 0 or num_states == 0:
        msg = ('No doped ions generated: the lattice or' +
               ' the concentrations are too small, or the number of energy states is zero!')
        raise LatticeError(msg)

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
    # number of ions
    lattice_info = {}
    lattice_info['num_total'] = num_doped_atoms
    lattice_info['num_activators'] = num_A
    lattice_info['num_sensitizers'] = num_S
    # save number of states so this lattice is only used with the right settings
    lattice_info['energy_states'] = num_states
    lattice_info['sensitizer_states'] = num_S_states
    lattice_info['activator_states'] = num_A_states
    if radius is not None:
        lattice_info['radius'] = radius

    indices_S_i, indices_A_j, initial_population = create_ground_states(ion_type, lattice_info)

    (index_S_k, index_S_l,
     index_A_k, index_A_l,
     dist_S_k, dist_S_l,
     dist_A_k, dist_A_l) = create_interaction_matrices(ion_type, dist_array,
                                                       indices_S_i, indices_A_j,
                                                       lattice_info)

    if not no_save:
        logger.info('Saving data...')
        # check if folder exists
        if full_path is None: # pragma: no cover
            folder_path = os.path.join('latticeData', lattice_name)
            radius = cte.lattice.get('radius', None)
            full_path = make_full_path(folder_path, num_uc, S_conc, A_conc, radius=radius)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with h5py.File(full_path, mode='w') as file:
            file.create_dataset('dist_array', data=dist_array, compression='gzip')
            file.create_dataset('ion_type', data=ion_type, compression='gzip')
            file.create_dataset('doped_lattice', data=doped_lattice, compression='gzip')
            file.create_dataset('initial_population', data=initial_population, compression='gzip')
            # serialze lattice_info as text and store it as an attribute
            file.attrs['lattice_info'] = yaml.dump(lattice_info)

            file.create_dataset('indices_S_i', data=np.array(indices_S_i), compression='gzip')
            file.create_dataset('indices_A_j', data=np.array(indices_A_j), compression='gzip')

            file.create_dataset('index_S_k', data=index_S_k, compression='gzip')
            file.create_dataset('dist_S_k', data=dist_S_k, compression='gzip')

            file.create_dataset('index_S_l', data=index_S_l, compression='gzip')
            file.create_dataset('dist_S_l', data=dist_S_l, compression='gzip')

            file.create_dataset('index_A_k', data=index_A_k, compression='gzip')
            file.create_dataset('dist_A_k', data=dist_A_k, compression='gzip')

            file.create_dataset('index_A_l', data=index_A_l, compression='gzip')
            file.create_dataset('dist_A_l', data=dist_A_l, compression='gzip')

    # plot lattice
    if plot_toggle:
        plotter.plot_lattice(doped_lattice, ion_type)

    elapsed_time = time.time()-start_time
    formatted_time = time.strftime("%Mm %Ss", time.localtime(elapsed_time))
    logger.info('Generating lattice finished. Total time: %s.', formatted_time)

    return (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
            indices_S_i, indices_A_j,
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
#    cte = settings.load('config_file.cfg')
#    cte['no_console'] = False
#    cte['no_plot'] = False
#
##    cte.lattice['S_conc'] = 100
##    cte.lattice['A_conc'] = 0
##    cte.lattice['N_uc'] = 1
##    cte.lattice['radius'] = 20
##    cte.states['sensitizer_states'] = 0
##    cte.states['activator_states'] = 0
#
##    cte.lattice['sites_occ'] = 1.0
##    cte.lattice['A_conc'] = 75.0
##    cte.lattice['S_conc'] = 75.0
#
#    (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
#     index_S_i, index_A_j,
#     index_S_k, dist_S_k,
#     index_S_l, dist_S_l,
#     index_A_k, dist_A_k,
#     index_A_l, dist_A_l) = generate(cte)
