# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:44:07 2016

@author: Pedro
"""
import os
import pytest
import numpy as np
# pylint: disable=E1101

import simetuc.lattice as lattice
from simetuc.util import temp_bin_filename

test_folder_path = os.path.dirname(os.path.abspath(__file__))


def idfn(params):
    '''Returns the name of the test according to the parameters'''
    return '{}S_{}A_{}Nuc_{}Ss_{}As'.format(params[0], params[1], params[2],
                                            params[3], params[4])
# S_conc, A_conc, N_uc, S_states, A_states, test should return a exception
@pytest.mark.parametrize('params', [(10.5, 5.2, 7, 2, 7, None), # normal
                                    (0.1, 0.0, 40, 2, 7, None), # no A, change S and N_uc
                                    (5.0, 0.0, 10, 2, 7, None), # no A, change S and N_uc
                                    (20.0, 0.0, 8, 2, 7, None), # no A, change S and N_uc
                                    (100.0, 0.0, 5, 2, 7, None), # no A, change S and N_uc
                                    (0.0, 0.1, 40, 2, 7, None), # no S, change A and N_uc
                                    (0.0, 0.5, 25, 2, 7, None), # no S, change A and N_uc
                                    (0.0, 20.0, 6, 2, 7, None), # no S, change A and N_uc
                                    (0.0, 100.0, 3, 2, 7, None), # no S, change A and N_uc
                                    (5.0, 0.0, 10, 5, 0, None), # no A_states, no A_conc
                                    (0.0, 5.0, 10, 0, 4, None), # no S_states, no S_conc
                                    (2, 2, 7, 2, 7, 30), # radius
                                    ], ids=idfn)
def test_cte_ok(setup_cte, params):
    '''Test a lattice with different concentrations of S and A; different number of unit cells;
        and different number of S and A energy states
    '''
    cte = setup_cte
    cte['no_plot'] = False

    cte['lattice']['S_conc'] = params[0]
    cte['lattice']['A_conc'] = params[1]
    cte['lattice']['N_uc'] = params[2]
    cte['states']['sensitizer_states'] = params[3]
    cte['states']['activator_states'] = params[4]
    if params[5] is not None:
        cte['lattice']['radius'] = params[5]
        del cte['lattice']['N_uc']

    with temp_bin_filename() as temp_filename:
        (dist_array, ion_type, doped_lattice,
         initial_population, lattice_info,
         index_S_i, index_A_j,
         index_S_k, dist_S_k,
         index_S_l, dist_S_l,
         index_A_k, dist_A_k,
         index_A_l, dist_A_l) = lattice.generate(cte, full_path=temp_filename)

    num_ions = lattice_info['num_total']
    num_activators = lattice_info['num_activators']
    num_sensitizers = lattice_info['num_sensitizers']

    num_states = lattice_info['energy_states']
    num_S_states = lattice_info['sensitizer_states']
    num_A_states = lattice_info['activator_states']

    assert dist_array.shape == (num_ions, num_ions)
    # symmetric matrix
    assert np.allclose(dist_array, dist_array.T)
    # only positive distances
    assert np.alltrue(dist_array >= 0)

    assert ion_type.shape == (num_ions, )
    assert np.max(ion_type) == 1 or np.max(ion_type) == 0
    assert np.min(ion_type) == 1 or np.min(ion_type) == 0
    assert np.count_nonzero(ion_type) == num_activators
    assert ion_type.shape[0] - np.count_nonzero(ion_type) == num_sensitizers

    assert doped_lattice.shape == (num_ions, 3)

    assert initial_population.shape == (num_states, )
    assert np.max(initial_population) <= 1
    assert np.min(initial_population) >= 0

    assert len(index_S_i) == num_ions
    assert min(index_S_i) >= -1
    assert max(index_S_i) <= num_states-1

    assert len(index_A_j) == num_ions
    assert min(index_A_j) >= -1
    assert max(index_A_j) <= num_states-1

    if num_sensitizers > 0 and num_S_states > 0:
        assert len(index_S_k) == num_sensitizers
        assert all(len(list_elem) == num_sensitizers-1 for list_elem in index_S_k)
        assert all(-1 <= max(list_elem) <= num_states for list_elem in index_S_k if len(list_elem))
        assert len(dist_S_k) == num_sensitizers
        assert all(len(list_elem) == num_sensitizers-1 for list_elem in dist_S_k)
        assert all(np.alltrue(list_elem >= 0) for list_elem in dist_S_k)

        if num_activators > 0 and num_A_states > 0:
            assert len(index_S_l) == num_sensitizers
            assert all(len(list_elem) == num_activators for list_elem in index_S_l)
            assert all(-1 <= max(list_elem) <= num_states for list_elem in index_S_l if len(list_elem))
            assert len(dist_S_l) == num_sensitizers
            assert all(len(list_elem) == num_activators for list_elem in dist_S_l)
            assert all(np.alltrue(list_elem >= 0) for list_elem in dist_S_l)

    if num_activators > 0 and num_A_states > 0:
        assert len(index_A_l) == num_activators
        assert all(len(list_elem) == num_activators-1 for list_elem in index_A_l)
        assert all(-1 <= max(list_elem) <= num_states for list_elem in index_A_l if len(list_elem))
        assert len(dist_A_l) == num_activators
        assert all(len(list_elem) == num_activators-1 for list_elem in dist_A_l)
        assert all(np.alltrue(list_elem >= 0) for list_elem in dist_A_l)

        if num_sensitizers > 0 and num_S_states > 0:
            assert len(index_A_k) == num_activators
            assert all(len(list_elem) == num_sensitizers for list_elem in index_A_k)
            assert all(-1 <= max(list_elem) <= num_states for list_elem in index_A_k if len(list_elem))
            assert len(dist_A_k) == num_activators
            assert all(len(list_elem) == num_sensitizers for list_elem in dist_A_k)
            assert all(np.alltrue(list_elem >= 0) for list_elem in dist_A_k)


@pytest.mark.parametrize('params', [# TEST NEGATIVE AND EXCESSIVE CONCS AND N_UC
                                        (0.0, 0.0, 10, 2, 7, None), # no S nor A
                                        (25.0, 100.0, 10, 2, 7, None), # too much S+A
                                        (0.0, 0.0, 0, 2, 7, None), # no S, A, N_uc
                                        (25.0, 100.0, 0, 2, 7, None), # too much S+A, no N_uc
                                        (-25.0, 0.0, 10, 2, 7, None), # negative S
                                        (0.0, -50.0, 20, 2, 7, None), # negative A
                                        (125.0, 50.0, 5, 2, 7, None), # too much S
                                        (0.0, 50.0, -20, 2, 7, None), # negative N_uc
                                        (0.0003, 0.0, 1, 2, 7, None), # most likely no doped ions created
                                        # TEST NUMBER OF ENERGY STATES
                                        (5.0, 5.0, 10, 5, 0, None), # no A_states
                                        (10.0, 1.0, 8, 0, 4, None), # no S_states
                                        (6.0, 6.0, 5, 0, 0, None), # no S_states, A_states
                                        (5.0, 5.0, 10, 15, 0, None),
                                        # RADIUS
                                        (2, 1, 7, 2, 7, -5), # negative
                                        (2, 1, 7, 2, 7, 0), # zero
                                        ], ids=idfn)
def test_cte_wrong(setup_cte, params):
    '''Test a lattice with different unit cell parameters
    '''
    cte = setup_cte

    cte['lattice']['S_conc'] = params[0]
    cte['lattice']['A_conc'] = params[1]
    cte['lattice']['N_uc'] = params[2]
    cte['states']['sensitizer_states'] = params[3]
    cte['states']['activator_states'] = params[4]
    if params[5] is not None:
        cte['lattice']['radius'] = params[5]
        del cte['lattice']['N_uc']

    with pytest.raises(lattice.LatticeError):
        with temp_bin_filename() as temp_filename:
            (dist_array, ion_type, doped_lattice,
             initial_population, lattice_info,
             index_S_i, index_A_j,
             index_S_k, dist_S_k,
             index_S_l, dist_S_l,
             index_A_k, dist_A_k,
             index_A_l, dist_A_l) = lattice.generate(cte, full_path=temp_filename)



def idfn_cell(cell_params):
    '''Returns the name of the test according to the parameters'''
    return '{}a_{}b_{}c_{}alfa_{}beta_{}gamma'.format(*cell_params)

@pytest.mark.parametrize('cell_params', [(-5.9, 5.9, 3.5, 90, 90, 120), # negative param
                                         (5.9, 5.9, 3.5, -90, 90, 0), # negative angle
                                         (5.9, 5.9, 3.5, 90, 90, 500), # large angle
                                        ], ids=idfn_cell)
def test_unit_cell(setup_cte, cell_params):
    '''Test a lattice with different unit cell parameters
    '''

    cte = setup_cte
    for key, param in zip(['a', 'b', 'c', 'alpha', 'beta', 'gamma'], cell_params):
        cte.lattice[key] = param

    with pytest.raises(lattice.LatticeError):
        with temp_bin_filename() as temp_filename:
            (dist_array, ion_type, doped_lattice,
             initial_population, lattice_info,
             index_S_i, index_A_j,
             index_S_k, dist_S_k,
             index_S_l, dist_S_l,
             index_A_k, dist_A_k,
             index_A_l, dist_A_l) = lattice.generate(cte, full_path=temp_filename)


def idfn_sites(sites):
    '''Returns the name of the test according to the parameters'''
    return '{}pos_{}occs'.format(*sites)

@pytest.mark.parametrize('sites', [([[0, 0, 0], [-2/3, 1/3, 1/2]], [1, 1/2]), # negative pos
                                   ([[0, 0, 0], [3/2, 1/3, 1/2]], [1, 1/2]), # too large pos
                                   ([[0, 0, 0], [2/3, 1/3, 1/2]], [1, -1/2]), # negative occ
                                   ([[0, 0, 0], [2/2, 1/3, 1/2]], [1.5, 1/2]), # too large occ
                                   ([[0, 0], [2/2, 1/3, 1/2]], [1, 1/2]), # not 3 values for pos
                                   ([[0, 0, 0], [2/2, 1/3]], [1, 1/2]), # not 3 values for pos
                                   ([[0, 0, 0], [2/2, 1/3, 1/2]], [1]), # different pos and occs number
                                  ], ids=idfn_sites)
def test_sites(setup_cte, sites):
    '''Test a lattice with different unit cell parameters
    '''

    cte = setup_cte
    cte['lattice']['sites_pos'] = sites[0]
    cte['lattice']['sites_occ'] = sites[1]

    with pytest.raises(lattice.LatticeError):
        with temp_bin_filename() as temp_filename:
            (dist_array, ion_type, doped_lattice,
             initial_population, lattice_info,
             index_S_i, index_A_j,
             index_S_k, dist_S_k,
             index_S_l, dist_S_l,
             index_A_k, dist_A_k,
             index_A_l, dist_A_l) = lattice.generate(cte, full_path=temp_filename)

@pytest.mark.parametrize('concs', [(0.0, 50.0), (50.0, 0.0)])
def test_single_atom(setup_cte, concs):
    '''Generate lattices with a single S or A'''

    cte = setup_cte
    cte['lattice']['N_uc'] = 1
    cte['states']['sensitizer_states'] = 2
    cte['states']['activator_states'] = 7

    # single A atom
    success = False
    cte['lattice']['S_conc'] = concs[0]
    cte['lattice']['A_conc'] = concs[1]
    while not success:
        try:
            with temp_bin_filename() as temp_filename:
                (dist_array, ion_type, doped_lattice,
                 initial_population, lattice_info,
                 index_S_i, index_A_j,
                 index_S_k, dist_S_k,
                 index_S_l, dist_S_l,
                 index_A_k, dist_A_k,
                 index_A_l, dist_A_l) = lattice.generate(cte, full_path=temp_filename)
        except lattice.LatticeError: # no ions were generated, repeat
            pass
        else:
            if len(ion_type) == 1: # only one ion was generated
                success = True

