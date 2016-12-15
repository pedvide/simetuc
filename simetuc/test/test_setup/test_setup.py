# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:44:07 2016

@author: Pedro
"""
import os
from collections import OrderedDict

import pytest
import numpy as np
# pylint: disable=E1101
import scipy.sparse as sparse

import simetuc.precalculate as precalculate
import simetuc.lattice as lattice # for the LatticeError exception
from simetuc.util import temp_bin_filename


test_folder_path = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope='function')
def setup_cte():
    '''Load the cte data structure'''

    cte = dict([('lattice',
              dict([('name', 'bNaYF4'),
                           ('N_uc', 8),
                           ('S_conc', 0.3),
                           ('A_conc', 0.3),
                           ('a', 5.9738),
                           ('b', 5.9738),
                           ('c', 3.5297),
                           ('alpha', 90),
                           ('beta', 90),
                           ('gamma', 120),
                           ('spacegroup', 'P-6'),
                           ('sites_ions', ['Y', 'Y']),
                           ('sites_pos',
                            [(0.0, 0.0, 0.0),
                             (0.6666666666666666, 0.3333333333333333, 0.5)]),
                           ('sites_occ', [1.0, 0.5]),
                           ('cell_par',
                            [5.9738, 5.9738, 3.5297, 90.0, 90.0, 120.0])])),
             ('states',
              dict([('sensitizer_ion_label', 'Yb'),
                           ('sensitizer_states_labels', ['GS', 'ES']),
                           ('activator_ion_label', 'Tm'),
                           ('activator_states_labels',
                            ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2']),
                           ('sensitizer_states', 2),
                           ('activator_states', 7)])),
             ('excitations',
              dict([('Vis_473',
                            dict([('active', True),
                                         ('power_dens', 1000000.0),
                                         ('t_pulse', 1e-08),
                                         ('process', ['Tm(3H6) -> Tm(1G4)']),
                                         ('degeneracy', [1.4444444444444444]),
                                         ('pump_rate', [0.00093]),
                                         ('init_state', [0]),
                                         ('final_state', [5]),
                                         ('ion_exc', ['A'])])),
                           ('NIR_980',
                            dict([('active', False),
                                         ('power_dens', 10000000.0),
                                         ('t_pulse', 1e-08),
                                         ('process', ['Yb(GS)->Yb(ES)']),
                                         ('degeneracy', [1.3333333333333333]),
                                         ('pump_rate', [0.0044]),
                                         ('init_state', [0]),
                                         ('final_state', [1]),
                                         ('ion_exc', ['S'])])),
                           ('NIR_1470',
                            dict([('active', False),
                                         ('power_dens', 10000000.0),
                                         ('t_pulse', 1e-08),
                                         ('process', ['Tm(1G4) -> Tm(1D2)']),
                                         ('degeneracy', [1.8]),
                                         ('pump_rate', [0.0002]),
                                         ('init_state', [5]),
                                         ('final_state', [6]),
                                         ('ion_exc', ['A'])])),
                           ('NIR_800',
                            dict([('active', False),
                                         ('power_dens', 10000000.0),
                                         ('t_pulse', 1e-08),
                                         ('process', ['Tm(3H6)->Tm(3H4)', 'Tm(3H5)->Tm(1G4)']),
                                         ('degeneracy', [1.4444444444444444, 1.2222222222222223]),
                                         ('pump_rate', [0.0044, 0.002]),
                                         ('init_state', [0, 2]),
                                         ('final_state', [3, 5]),
                                         ('ion_exc', ['A', 'A'])]))])),
             ('optimization_params', ['CR50']),
             ('decay',
              {'B_pos_value_A': [(1, 0, 1.0),
                (2, 1, 0.4),
                (3, 1, 0.3),
                (4, 3, 0.999),
                (5, 1, 0.15),
                (5, 2, 0.16),
                (5, 3, 0.04),
                (5, 4, 0.0),
                (6, 1, 0.43)],
               'B_pos_value_S': [(1, 0, 1.0)],
               'pos_value_A': [(1, 83.33333333333333),
                (2, 40000.0),
                (3, 500.0),
                (4, 500000.0),
                (5, 1315.7894736842104),
                (6, 14814.814814814814)],
               'pos_value_S': [(1, 400.0)]}),
             ('ET', # OrderedDict so the explicit examples are correct
              OrderedDict([('CR50',
                            {'indices': [5, 0, 3, 2],
                             'mult': 6,
                             'type': 'AA',
                             'value': 887920884.0,
                             'value_avg': 1.0e3}),
                           ('ETU53',
                            {'indices': [5, 3, 6, 1],
                             'mult': 6,
                             'type': 'AA',
                             'value': 450220614.0}),
                           ('ETU55',
                            {'indices': [5, 5, 6, 4],
                             'mult': 6,
                             'type': 'AA',
                             'value': 0.0}),
                           ('ETU1',
                            {'indices': [1, 0, 0, 2],
                             'mult': 6,
                             'type': 'SA',
                             'value': 10000.0}),
                           ('BackET',
                            {'indices': [3, 0, 0, 1],
                             'mult': 6,
                             'type': 'AS',
                             'value': 4502.20614}),
                           ('EM',
                            {'indices': [1, 0, 0, 1],
                             'mult': 6,
                             'type': 'SS',
                             'value': 45022061400.0}),
                            ('coop1',
                             {'indices': [1, 1, 0, 0, 0, 5],
                              'mult': 6,
                              'type': 'SSA',
                              'value': 1000.0})
                            ]))])

    cte['no_console'] = True
    cte['no_plot'] = True
    return cte

# SIMPLE LATTICES WITH 1 OR 2 ACTIVATORS AND SENSITIZERS
# THESE RESULTS HAVE BEEN CHECKED BY HAND

def test_lattice_1A(setup_cte):
    '''Test a lattice with just one activator'''

    test_filename = os.path.join(test_folder_path, 'data_0S_1A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices,
     coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()
    power_dens = cte['excitations']['Vis_473']['power_dens']

    assert  np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0]))

    assert np.all(index_S_i == np.array([-1]))
    assert np.all(index_A_j == np.array([0]))

    good_abs_matrix = np.array([[-0.00093,0.,0.,0.,0., 0.00134333,0.],
                                [0.      ,0.,0.,0.,0., 0.        ,0.],
                                [0.      ,0.,0.,0.,0., 0.        ,0.],
                                [0.      ,0.,0.,0.,0., 0.        ,0.],
                                [0.      ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.00093,0.,0.,0.,0.,-0.00134333,0.],
                                [0.      ,0.,0.,0.,0., 0.        ,0.],], dtype=np.float64)
    assert np.allclose(power_dens*good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0.00000000e+00, 8.33333333e+01, 2.40000000e+04, 3.50000000e+02, 5.00000000e+02, 8.55263158e+02, 8.44444444e+03],
                                  [0.00000000e+00,-8.33333333e+01, 1.60000000e+04, 1.50000000e+02, 0.00000000e+00, 1.97368421e+02, 6.37037037e+03],
                                  [0.00000000e+00, 0.00000000e+00,-4.00000000e+04, 0.00000000e+00, 0.00000000e+00, 2.10526316e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+02, 4.99500000e+05, 5.26315789e+01, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+05, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.31578947e+03, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.48148148e+04]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([], dtype=np.float64).reshape((7, 0))
    assert np.allclose(good_UC_matrix, UC_matrix)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 2)) == N_indices)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 3)) == jac_indices)

def test_lattice_1A_ESA(setup_cte): # use the ESA processes in NIR_800 excitation
    '''Test a lattice with just one activator'''

    test_filename = os.path.join(test_folder_path, 'data_0S_1A.hdf5')

    setup_cte['excitations']['Vis_473']['active'] = False
    setup_cte['excitations']['NIR_800']['active'] = True

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()
    power_dens = cte['excitations']['NIR_800']['power_dens']

    # reset active excitation so next tests will work
    setup_cte['excitations']['Vis_473']['active'] = True
    setup_cte['excitations']['NIR_800']['active'] = False

    assert np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0]))

    assert np.all(index_S_i == np.array([-1]))
    assert np.all(index_A_j == np.array([0]))

    good_abs_matrix_GSA = 0.0044*np.array([[-1.,0.,0., 13/9,0.,0.,0.],
                                           [ 0.,0.,0., 0.  ,0.,0.,0.],
                                           [ 0.,0.,0., 0.  ,0.,0.,0.],
                                           [ 1.,0.,0.,-13/9,0.,0.,0.],
                                           [ 0.,0.,0., 0.  ,0.,0.,0.],
                                           [ 0.,0.,0., 0.  ,0.,0.,0.],
                                           [ 0.,0.,0., 0.  ,0.,0.,0.],], dtype=np.float64)

    good_abs_matrix_ESA = 0.002*np.array([[0.,0., 0.,0.,0., 0.  ,0.],
                                          [0.,0., 0.,0.,0., 0.  ,0.],
                                          [0.,0.,-1.,0.,0., 11/9,0.],
                                          [0.,0., 0.,0.,0., 0.  ,0.],
                                          [0.,0., 0.,0.,0., 0.  ,0.],
                                          [0.,0., 1.,0.,0.,-11/9,0.],
                                          [0.,0., 0.,0.,0., 0.  ,0.],], dtype=np.float64)
    good_abs_matrix = power_dens*(good_abs_matrix_GSA + good_abs_matrix_ESA)
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0.00000000e+00, 8.33333333e+01, 2.40000000e+04, 3.50000000e+02, 5.00000000e+02, 8.55263158e+02, 8.44444444e+03],
                                  [0.00000000e+00,-8.33333333e+01, 1.60000000e+04, 1.50000000e+02, 0.00000000e+00, 1.97368421e+02, 6.37037037e+03],
                                  [0.00000000e+00, 0.00000000e+00,-4.00000000e+04, 0.00000000e+00, 0.00000000e+00, 2.10526316e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+02, 4.99500000e+05, 5.26315789e+01, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+05, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.31578947e+03, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.48148148e+04]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([], dtype=np.float64).reshape((7, 0))
    assert np.allclose(good_UC_matrix, UC_matrix)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 2)) == N_indices)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 3)) == jac_indices)


def test_lattice_1A_two_color(setup_cte): # use two color excitation
    '''Test a lattice with just one activator'''

    test_filename = os.path.join(test_folder_path, 'data_0S_1A.hdf5')

    setup_cte['excitations']['Vis_473']['active'] = True
    setup_cte['excitations']['NIR_1470']['active'] = True

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()
    power_dens_GSA = cte['excitations']['Vis_473']['power_dens']
    power_dens_ESA = cte['excitations']['NIR_1470']['power_dens']

    # reset active excitation so next tests will work
    setup_cte['excitations']['Vis_473']['active'] = True
    setup_cte['excitations']['NIR_1470']['active'] = False

    assert  np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0]))

    assert np.all(index_S_i == np.array([-1]))
    assert np.all(index_A_j == np.array([0]))

    good_abs_matrix_GSA = power_dens_GSA*np.array([[-0.00093,0.,0.,0.,0., 0.00134333,0.],
                                                   [0.      ,0.,0.,0.,0., 0.        ,0.],
                                                   [0.      ,0.,0.,0.,0., 0.        ,0.],
                                                   [0.      ,0.,0.,0.,0., 0.        ,0.],
                                                   [0.      ,0.,0.,0.,0., 0.        ,0.],
                                                   [ 0.00093,0.,0.,0.,0.,-0.00134333,0.],
                                                   [0.      ,0.,0.,0.,0., 0.        ,0.]], dtype=np.float64)

    good_abs_matrix_ESA = power_dens_ESA*0.0002*np.array([[0.,0., 0.,0.,0., 0.   ,0.],
                                                          [0.,0., 0.,0.,0., 0.   ,0.],
                                                          [0.,0., 0.,0.,0., 0.   ,0.],
                                                          [0.,0., 0.,0.,0., 0.   ,0.],
                                                          [0.,0., 0.,0.,0., 0.   ,0.],
                                                          [0.,0., 0.,0.,0.,-1., 9/5.],
                                                          [0.,0., 0.,0.,0., 1.,-9/5.]], dtype=np.float64)
    good_abs_matrix = good_abs_matrix_GSA + good_abs_matrix_ESA
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0.00000000e+00, 8.33333333e+01, 2.40000000e+04, 3.50000000e+02, 5.00000000e+02, 8.55263158e+02, 8.44444444e+03],
                                  [0.00000000e+00,-8.33333333e+01, 1.60000000e+04, 1.50000000e+02, 0.00000000e+00, 1.97368421e+02, 6.37037037e+03],
                                  [0.00000000e+00, 0.00000000e+00,-4.00000000e+04, 0.00000000e+00, 0.00000000e+00, 2.10526316e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+02, 4.99500000e+05, 5.26315789e+01, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-5.00000000e+05, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.31578947e+03, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,-1.48148148e+04]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([], dtype=np.float64).reshape((7, 0))
    assert np.allclose(good_UC_matrix, UC_matrix)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 2)) == N_indices)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 3)) == jac_indices)

def test_lattice_2A(setup_cte):
    '''Test a lattice with two activators'''

    test_filename = os.path.join(test_folder_path, 'data_0S_2A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()
    power_dens = cte['excitations']['Vis_473']['power_dens']

    assert  np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0,
                                                   1, 0, 0, 0, 0, 0, 0]))

    assert np.all(index_S_i == np.array([-1, -1]))
    assert np.all(index_A_j == np.array([0, 7]))

    good_abs_matrix = np.array([[-0.00093,0.,0.,0.,0., 0.00134333,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.00093,0.,0.,0.,0.,-0.00134333,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0.,-0.00093,0.,0.,0.,0., 0.00134333,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.00093,0.,0.,0.,0.,-0.00134333,0.],
                                [ 0.     ,0.,0.,0.,0., 0.        ,0., 0.     ,0.,0.,0.,0., 0.        ,0.]], dtype=np.float64)
    assert np.allclose(power_dens*good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0.00000000e+00,8.33333333e+01,2.40000000e+04,3.50000000e+02,5.00000000e+02,8.55263158e+02,8.44444444e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,-8.33333333e+01,1.60000000e+04,1.50000000e+02,0.00000000e+00,1.97368421e+02,6.37037037e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,-4.00000000e+04,0.00000000e+00,0.00000000e+00,2.10526316e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+02,4.99500000e+05,5.26315789e+01,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+05,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.31578947e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.48148148e+04,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,8.33333333e+01,2.40000000e+04,3.50000000e+02,5.00000000e+02,8.55263158e+02,8.44444444e+03],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-8.33333333e+01,1.60000000e+04,1.50000000e+02,0.00000000e+00,1.97368421e+02,6.37037037e+03],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-4.00000000e+04,0.00000000e+00,0.00000000e+00,2.10526316e+02,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+02,4.99500000e+05,5.26315789e+01,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+05,0.00000000e+00,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.31578947e+03,0.00000000e+00],
                                  [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.48148148e+04]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([[   0.        ,    0.        , -145.66404614,    0.        ],
                               [   0.        ,    0.        ,    0.        ,   73.8590087 ],
                               [   0.        ,    0.        ,  145.66404614,    0.        ],
                               [ 145.66404614,    0.        ,    0.        ,  -73.8590087 ],
                               [   0.        ,    0.        ,    0.        ,    0.        ],
                               [-145.66404614,  -73.8590087 ,    0.        ,    0.        ],
                               [   0.        ,   73.8590087 ,    0.        ,    0.        ],
                               [-145.66404614,    0.        ,    0.        ,    0.        ],
                               [   0.        ,   73.8590087 ,    0.        ,    0.        ],
                               [ 145.66404614,    0.        ,    0.        ,    0.        ],
                               [   0.        ,  -73.8590087 ,  145.66404614,    0.        ],
                               [   0.        ,    0.        ,    0.        ,    0.        ],
                               [   0.        ,    0.        , -145.66404614,  -73.8590087 ],
                               [   0.        ,    0.        ,    0.        ,   73.8590087 ]], dtype=np.float64)
    assert np.allclose(good_UC_matrix, UC_matrix)

    good_N_indices = np.array([[ 5,  7],
                               [ 5, 10],
                               [12,  0],
                               [12,  3]], dtype=np.uint64)
    assert np.all(good_N_indices == N_indices)

    good_jac_indices = np.array([[ 0,  5,  7],
                                 [ 0,  7,  5],
                                 [ 1,  5, 10],
                                 [ 1, 10,  5],
                                 [ 2, 12,  0],
                                 [ 2,  0, 12],
                                 [ 3, 12,  3],
                                 [ 3,  3, 12]], dtype=np.uint64)
    assert np.all(good_jac_indices == jac_indices)

def test_lattice_1S(setup_cte):
    '''Test a lattice with just one sensitizer'''

    test_filename = os.path.join(test_folder_path, 'data_1S_0A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()

    assert  np.all(initial_population == np.array([1, 0]))

    assert np.all(index_S_i == np.array([0]))
    assert np.all(index_A_j == np.array([-1]))

    good_abs_matrix = np.array([[0.,0.],
                                [0.,0.]], dtype=np.float64)
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0., 400.],
                                  [0.,-400.]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([], dtype=np.float64).reshape((2, 0))
    assert np.allclose(good_UC_matrix, UC_matrix)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 2)) == N_indices)

    assert np.all(np.array([], dtype=np.uint64).reshape((0, 3)) == jac_indices)

def test_lattice_2S(setup_cte):
    '''Test a lattice with two sensitizers'''

    test_filename = os.path.join(test_folder_path, 'data_2S_0A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()

    assert  np.all(initial_population == np.array([1, 0, 1, 0]))

    assert np.all(index_S_i == np.array([0, 2]))
    assert np.all(index_A_j == np.array([-1, -1]))

    good_abs_matrix = np.array([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]], dtype=np.float64)
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0., 400.,0., 0.  ],
                                  [0.,-400.,0., 0.  ],
                                  [0., 0.  ,0., 400.],
                                  [0., 0.  ,0.,-400.]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([[ 32654.03267632, -32654.03267632],
                               [-32654.03267632,  32654.03267632],
                               [-32654.03267632,  32654.03267632],
                               [ 32654.03267632, -32654.03267632]], dtype=np.float64)
    assert np.allclose(good_UC_matrix, UC_matrix)

    good_N_indices = np.array([[1, 2],
                               [3, 0]], dtype=np.uint64)
    assert np.all(good_N_indices == N_indices)

    good_jac_indices = np.array([[0, 1, 2],
                                 [0, 2, 1],
                                 [1, 3, 0],
                                 [1, 0, 3]], dtype=np.uint64)
    assert np.all(good_jac_indices == jac_indices)

def test_lattice_1S_1A(setup_cte):
    '''Test a lattice with one sensitizer and one activator'''

    test_filename = os.path.join(test_folder_path, 'data_1S_1A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()

    assert  np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0, 1, 0]))

    assert np.all(index_S_i == np.array([-1, 7]))
    assert np.all(index_A_j == np.array([0, -1]))

    good_abs_matrix = np.array([[-930.,0.,0.,0.,0.,1343.33333333,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [930.,0.,0.,0.,0.,-1343.33333333,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                 [0.,0.,0.,0.,0.,0.,0.,0.,0.]], dtype=np.float64)
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[  0.00000000e+00,   8.33333333e+01,   2.40000000e+04,
                                      3.50000000e+02,   5.00000000e+02,   8.55263158e+02,
                                      8.44444444e+03,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,  -8.33333333e+01,   1.60000000e+04,
                                      1.50000000e+02,   0.00000000e+00,   1.97368421e+02,
                                      6.37037037e+03,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,  -4.00000000e+04,
                                      0.00000000e+00,   0.00000000e+00,   2.10526316e+02,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                     -5.00000000e+02,   4.99500000e+05,   5.26315789e+01,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,  -5.00000000e+05,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,  -1.31578947e+03,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                     -1.48148148e+04,   0.00000000e+00,   0.00000000e+00],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,   4.00000000e+02],
                                   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                      0.00000000e+00,   0.00000000e+00,  -4.00000000e+02]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)

    good_UC_matrix = np.array([[ 0.00043243, -0.00096047],
                               [ 0.        ,  0.        ],
                               [ 0.        ,  0.00096047],
                               [-0.00043243,  0.        ],
                               [ 0.        ,  0.        ],
                               [ 0.        ,  0.        ],
                               [ 0.        ,  0.        ],
                               [-0.00043243,  0.00096047],
                               [ 0.00043243, -0.00096047]], dtype=np.float64)
    assert np.allclose(good_UC_matrix, UC_matrix)

    good_N_indices = np.array([[3, 7],
                               [8, 0]], dtype=np.uint64)
    assert np.all(good_N_indices == N_indices)

    good_jac_indices = np.array([[0, 3, 7],
                                 [0, 7, 3],
                                 [1, 8, 0],
                                 [1, 0, 8]], dtype=np.uint64)
    assert np.all(good_jac_indices == jac_indices)

def test_lattice_2S_2A(setup_cte):
    '''Test a lattice with two sensitizers and two activators'''

    test_filename = os.path.join(test_folder_path, 'data_2S_2A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix, UC_matrix,
     N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)
    UC_matrix = UC_matrix.toarray()
    total_abs_matrix = total_abs_matrix.toarray()
    decay_matrix = decay_matrix.toarray()

    assert  np.all(initial_population == np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint64))

    assert np.all(index_S_i == np.array([-1, 7, 9, -1]))
    assert np.all(index_A_j == np.array([0, -1, -1, 11]))

    good_abs_matrix = np.array([[-930.,0.,0.,0.,0.,1343.33333333,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [930.,0.,0.,0.,0.,-1343.33333333,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-930.,0.,0.,0.,0.,1343.33333333,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,930.,0.,0.,0.,0.,-1343.33333333,0.],
                                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], dtype=np.float64)
    assert np.allclose(good_abs_matrix, total_abs_matrix)

    good_decay_matrix = np.array([[0.00000000e+00,8.33333333e+01,2.40000000e+04,3.50000000e+02,5.00000000e+02,8.55263158e+02,8.44444444e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,-8.33333333e+01,1.60000000e+04,1.50000000e+02,0.00000000e+00,1.97368421e+02,6.37037037e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,-4.00000000e+04,0.00000000e+00,0.00000000e+00,2.10526316e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+02,4.99500000e+05,5.26315789e+01,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+05,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.31578947e+03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.48148148e+04,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,4.00000000e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-4.00000000e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,4.00000000e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-4.00000000e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,8.33333333e+01,2.40000000e+04,3.50000000e+02,5.00000000e+02,8.55263158e+02,8.44444444e+03],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-8.33333333e+01,1.60000000e+04,1.50000000e+02,0.00000000e+00,1.97368421e+02,6.37037037e+03],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-4.00000000e+04,0.00000000e+00,0.00000000e+00,2.10526316e+02,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+02,4.99500000e+05,5.26315789e+01,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-5.00000000e+05,0.00000000e+00,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.31578947e+03,0.00000000e+00],
                                [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.48148148e+04]], dtype=np.float64)
    assert np.allclose(good_decay_matrix, decay_matrix)


    good_UC_matrix = np.array([[0.00000000e+00,0.00000000e+00,4.03431633e-02,2.63711661e-03,0.00000000e+00,-8.96075436e-02,0.00000000e+00,0.00000000e+00,-5.85738752e-03,0.00000000e+00,-2.37501385e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.20425165e+02,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,8.96075436e-02,0.00000000e+00,0.00000000e+00,5.85738752e-03,0.00000000e+00,2.37501385e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [2.37501385e+02,0.00000000e+00,-4.03431633e-02,-2.63711661e-03,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.20425165e+02,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [-2.37501385e+02,-1.20425165e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,1.20425165e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,-4.03431633e-02,0.00000000e+00,9.90652420e+05,8.96075436e-02,2.58929240e-03,-9.90652420e+05,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-1.16575281e-03,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,4.03431633e-02,0.00000000e+00,-9.90652420e+05,-8.96075436e-02,-2.58929240e-03,9.90652420e+05,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.16575281e-03,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,-2.63711661e-03,-9.90652420e+05,0.00000000e+00,0.00000000e+00,9.90652420e+05,5.85738752e-03,1.59873090e-02,0.00000000e+00,0.00000000e+00,0.00000000e+00,-7.19781608e-03],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,2.63711661e-03,9.90652420e+05,0.00000000e+00,0.00000000e+00,-9.90652420e+05,-5.85738752e-03,-1.59873090e-02,0.00000000e+00,0.00000000e+00,0.00000000e+00,7.19781608e-03],
                                    [-2.37501385e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-2.58929240e-03,0.00000000e+00,0.00000000e+00,-1.59873090e-02,0.00000000e+00,0.00000000e+00,1.16575281e-03,7.19781608e-03],
                                    [0.00000000e+00,1.20425165e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [2.37501385e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.58929240e-03,0.00000000e+00,0.00000000e+00,1.59873090e-02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,-1.20425165e+02,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,2.37501385e+02,0.00000000e+00,-1.16575281e-03,-7.19781608e-03],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,-2.37501385e+02,-1.20425165e+02,0.00000000e+00,0.00000000e+00],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,1.20425165e+02,0.00000000e+00,0.00000000e+00]], dtype=np.float64)
    assert np.allclose(good_UC_matrix, UC_matrix)

    good_N_indices = np.array([[ 5, 11],
                               [ 5, 14],
                               [ 3,  7],
                               [ 3,  9],
                               [ 8,  9],
                               [ 8,  0],
                               [ 8, 11],
                               [10,  7],
                               [10,  0],
                               [10, 11],
                               [16,  0],
                               [16,  3],
                               [14,  7],
                               [14,  9]], dtype=np.uint64)
    assert np.all(good_N_indices == N_indices)

    good_jac_indices = np.array([[ 0,  5, 11],
                                   [ 0, 11,  5],
                                   [ 1,  5, 14],
                                   [ 1, 14,  5],
                                   [ 2,  3,  7],
                                   [ 2,  7,  3],
                                   [ 3,  3,  9],
                                   [ 3,  9,  3],
                                   [ 4,  8,  9],
                                   [ 4,  9,  8],
                                   [ 5,  8,  0],
                                   [ 5,  0,  8],
                                   [ 6,  8, 11],
                                   [ 6, 11,  8],
                                   [ 7, 10,  7],
                                   [ 7,  7, 10],
                                   [ 8, 10,  0],
                                   [ 8,  0, 10],
                                   [ 9, 10, 11],
                                   [ 9, 11, 10],
                                   [10, 16,  0],
                                   [10,  0, 16],
                                   [11, 16,  3],
                                   [11,  3, 16],
                                   [12, 14,  7],
                                   [12,  7, 14],
                                   [13, 14,  9],
                                   [13,  9, 14]], dtype=np.uint64)
    assert np.all(good_jac_indices == jac_indices)

# RANDOMIZED TESTS, WE DON'T CHECK THE ACTUAL VALUES, JUST THE SHAPES OF THE
# MATRICES RETURNED

def idfn(params):
    '''Returns the name of the test according to the parameters'''
    return '{}S_{}A_{}Nuc_{}Ss_{}As'.format(params[0], params[1], params[2],
                                            params[3], params[4])

@pytest.mark.parametrize('problem', ['setup_microscopic_eqs', 'setup_average_eqs']) # micro or average problem?
@pytest.mark.parametrize('absorption', ['NIR_980', 'Vis_473', 'NIR_800']) # absorption of S or A
@pytest.mark.parametrize('params', [(10.5, 5.2, 7, 2, 7), # normal
                                    (0.05, 0.0, 40, 2, 7), # no A, change S and N_uc
                                    (5.0, 0.0, 10, 2, 7), #
                                    (20.0, 0.0, 8, 2, 7), #
                                    (100.0, 0.0, 5, 2, 7), #
                                    (0.0, 0.05, 40, 2, 7), # no S, change A and N_uc
                                    (0.0, 0.5, 20, 2, 7), #
                                    (0.0, 20.0, 6, 2, 7), #
                                    (0.0, 100.0, 3, 2, 7), #
                                    (6.0, 6.0, 5, 10, 10) # high S_states, A_states
                                    ],
                         ids=idfn)
def test_random_lattice(setup_cte, params, absorption, problem):
    '''Test a lattice with a random number of A and S
        This test may depend on the functioning of the lattice module
        if the lattices need to be generated so it\'s not really a unit test
        the pre-computed lattices are stored in the test/test_setup folder
    '''

    cte = setup_cte

    cte['lattice']['S_conc'] = params[0]
    cte['lattice']['A_conc'] = params[1]
    cte['lattice']['N_uc'] = params[2]
    cte['states']['sensitizer_states'] = params[3]
    cte['states']['activator_states'] = params[4]

    if absorption == 'NIR_980': # sensitizer absorbs
        cte['excitations']['NIR_980']['active'] = True
        cte['excitations']['Vis_473']['active'] = False
        cte['excitations']['NIR_800']['active'] = False
    elif absorption == 'Vis_473': # activator absorbs, normal GSA
        cte['excitations']['NIR_980']['active'] = False
        cte['excitations']['Vis_473']['active'] = True
        cte['excitations']['NIR_800']['active'] = False
    elif absorption == 'NIR_800': # activator absorbs, ESA
        cte['excitations']['NIR_980']['active'] = False
        cte['excitations']['Vis_473']['active'] = False
        cte['excitations']['NIR_800']['active'] = True

    setup_func = precalculate.setup_microscopic_eqs
    if problem == 'setup_average_eqs':
        setup_func = precalculate.setup_average_eqs

    with temp_bin_filename() as temp_filename:
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix, ET_matrix,
         N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = setup_func(cte, full_path=temp_filename)

    # some matrices can grow very large. Make sure it's returned as sparse
    assert sparse.issparse(ET_matrix)
    assert sparse.issparse(total_abs_matrix)
    assert sparse.issparse(decay_matrix)

    assert cte['ions']['total'] == cte['ions']['activators'] + cte['ions']['sensitizers']
    num_ions = cte['ions']['total']

    assert cte['states']['energy_states'] == cte['ions']['activators']*cte['states']['activator_states'] +\
                                                cte['ions']['sensitizers']*cte['states']['sensitizer_states']
    num_states = cte['states']['energy_states']
    num_interactions = N_indices.shape[0]

    assert initial_population.shape == (num_states, )

    assert len(index_S_i) == num_ions
    assert min(index_S_i) >= -1
    assert max(index_S_i) <= num_states-1

    assert len(index_A_j) == num_ions
    assert min(index_A_j) >= -1
    assert max(index_A_j) <= num_states-1

    assert total_abs_matrix.shape == (num_states, num_states)
    # sum of all rows is zero for each column
    assert np.allclose(np.sum(total_abs_matrix, axis=0), 0.0)

    assert decay_matrix.shape == (num_states, num_states)
    # sum of all rows is zero for each column
    assert np.allclose(np.sum(decay_matrix, axis=0), 0.0)
    # decay matrix is upper triangular
    assert np.allclose(decay_matrix.todense(), np.triu(decay_matrix.todense()))

    assert ET_matrix.shape == (num_states, num_interactions)
    # sum of all rows is zero for each column
    assert np.allclose(np.sum(ET_matrix, axis=0), 0.0)
    # each column has 4 nonzero values
    assert ET_matrix.getnnz() == 4*num_interactions
    assert np.all([col.getnnz() == 4 for col in ET_matrix.T])

    assert N_indices.shape == (num_interactions, 2)
    if num_interactions != 0: # np.min and np.max don't work with empty arrays
        assert np.min(N_indices) >= 0
        assert np.max(N_indices) <= num_states-1

    assert jac_indices.shape == (2*num_interactions, 3)
    if num_interactions != 0: # np.min and np.max don't work with empty arrays
        assert np.min(jac_indices) >= 0
        # first column has interaction number
        assert np.max(jac_indices[:, 0]) <= num_interactions-1
        # second and third have pupulation indices
        assert np.max(jac_indices[:, 1]) <= num_states-1
        assert np.max(jac_indices[:, 2]) <= num_states-1

@pytest.mark.parametrize('problem', ['setup_microscopic_eqs', 'setup_average_eqs']) # micro or average problem?
@pytest.mark.parametrize('absorption', ['NIR_980', 'Vis_473', 'NIR_800']) # absorption of S or A
@pytest.mark.parametrize('params', [# TEST NEGATIVE AND EXCESSIVE CONCS AND N_UC
                                    (0.0, 0.0, 10, 2, 7), # no S or A
                                    (25.0, 100.0, 10, 2, 7), # too much S+A
                                    (0.0, 0.0, 0, 2, 7), # no S, A, N_uc
                                    (25.0, 100.0, 0, 2, 7), # too much S+A, no N_uc
                                    (-25.0, 0.0, 10, 2, 7), # negative S
                                    (0.0, -50.0, 20, 2, 7), # negative A
                                    (125.0, 10.0, 5, 2, 7), # too much S
                                    (0.0, 50.0, 0, 2, 7), # no N_uc
                                    (0.0, 50.0, -20, 2, 7), # negative N_uc
                                    # TEST NUMBER OF ENERGY STATES
                                    (5.0, 5.0, 10, 2, 0), # no A_states
                                    (10.0, 1.0, 8, 0, 7), # no S_states
                                    (6.0, 6.0, 5, 0, 0), # no S_states, A_states
                                    (5.0, 0.0, 10, 5, 0), # no A_states, no A_conc
                                    (0.0, 1.0, 8, 0, 4), # no S_states, no S_conc
                                    (5.0, 5.0, 10, 2, 1), # low A_states
                                    (5.0, 5.0, 8, 1, 7)], # low S_states
                         ids=idfn)
def test_random_wrong_lattice(setup_cte, params, absorption, problem):
    cte = setup_cte

    cte['lattice']['S_conc'] = params[0]
    cte['lattice']['A_conc'] = params[1]
    cte['lattice']['N_uc'] = params[2]
    cte['states']['sensitizer_states'] = params[3]
    cte['states']['activator_states'] = params[4]

    setup_func = precalculate.setup_microscopic_eqs
    if problem == 'setup_average_eqs':
        setup_func = precalculate.setup_average_eqs

    if absorption == 'NIR_980': # sensitizer absorbs
        cte['excitations']['NIR_980']['active'] = True
        cte['excitations']['Vis_473']['active'] = False
        cte['excitations']['NIR_800']['active'] = False
    elif absorption == 'Vis_473': # activator absorbs, normal GSA
        cte['excitations']['NIR_980']['active'] = False
        cte['excitations']['Vis_473']['active'] = True
        cte['excitations']['NIR_800']['active'] = False
    elif absorption == 'NIR_800': # activator absorbs, ESA
        cte['excitations']['NIR_980']['active'] = False
        cte['excitations']['Vis_473']['active'] = False
        cte['excitations']['NIR_800']['active'] = True

    with pytest.raises(lattice.LatticeError):
        with temp_bin_filename() as temp_filename:
            (cte, initial_population, index_S_i, index_A_j,
             absorption_matrix, decay_matrix, ET_matrix,
             N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = setup_func(setup_cte, full_path=temp_filename)

def test_get_lifetimes(setup_cte):

    cte = setup_cte
    cte['lattice']['S_conc'] = 10.5
    cte['lattice']['A_conc'] = 5.2
    cte['lattice']['N_uc'] = 7
    cte['states']['sensitizer_states'] = 2
    cte['states']['activator_states'] = 7

    with temp_bin_filename() as temp_filename:
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix, ET_matrix,
         N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(cte, full_path=temp_filename)

    tau_list = precalculate.get_lifetimes(cte)

    assert len(tau_list) == (cte['states']['sensitizer_states'] +
                            cte['states']['activator_states'] - 2)

def test_wrong_number_states(setup_cte):

    cte = setup_cte
    cte['lattice']['S_conc'] = 1
    cte['lattice']['A_conc'] = 1
    cte['lattice']['N_uc'] = 8
    cte['states']['sensitizer_states'] = 2
    cte['states']['activator_states'] = 7

    with temp_bin_filename() as temp_filename:
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix, ET_matrix,
         N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(cte, full_path=temp_filename)

        # change number of states
        cte['states']['activator_states'] = 10
        (cte, initial_population, index_S_i, index_A_j,
         total_abs_matrix, decay_matrix, ET_matrix,
         N_indices, jac_indices, coop_ET_matrix, coop_N_indices, coop_jac_indices) = precalculate.setup_microscopic_eqs(cte, full_path=temp_filename)
