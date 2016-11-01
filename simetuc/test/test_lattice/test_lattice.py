# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:44:07 2016

@author: Pedro
"""
from collections import OrderedDict

import pytest
import numpy as np
# pylint: disable=E1101
#import scipy.sparse as sparse

import lattice as lattice

@pytest.fixture(scope='module')
def setup_cte():
    '''Load the cte data structure'''

    cte = OrderedDict([
             ('lattice',
              OrderedDict([('name', 'bNaYF4'),
                           ('N_uc', 20),
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
              OrderedDict([('sensitizer_ion_label', 'Yb'),
                           ('sensitizer_states_labels', ['GS', 'ES']),
                           ('activator_ion_label', 'Tm'),
                           ('activator_states_labels',
                            ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2']),
                           ('sensitizer_states', 2),
                           ('activator_states', 7)])),
             ('excitations',
              OrderedDict([('Vis_473',
                            OrderedDict([('active', True),
                                         ('power_dens', '1e6'),
                                         ('t_pulse', '1e-8'),
                                         ('process', 'Tm(3H6) -> Tm(1G4)'),
                                         ('degeneracy', '13/9'),
                                         ('pump_rate', 0.00093)])),
                           ('NIR_980',
                            OrderedDict([('active', False),
                                         ('power_dens', '1e7'),
                                         ('t_pulse', '1e-8'),
                                         ('process', 'Yb(GS)->Yb(ES)'),
                                         ('degeneracy', '4/3'),
                                         ('pump_rate', 0.0044)])),
                           ('label', 'Vis_473'),
                           ('t_pulse', 1e-08),
                           ('power_dens', 1000000.0),
                           ('degeneracy', 1.4444444444444444),
                           ('pump_rate', 0.00093),
                           ('ion_type', 'A'),
                           ('init_state', 0),
                           ('final_state', 5)])),
             ('experimental_data',
              [None,
               None,
               None,
               'tau_Tm_3F4_exc_Vis_473.txt',
               None,
               'tau_Tm_3H4_exc_Vis_473.txt',
               None,
               'tau_Tm_1G4_exc_Vis_473.txt',
               'tau_Tm_1D2_exc_Vis_473.txt']),
             ('optimization_params', ['CR50']),
             ('decay',
              {'B_pos_value_A': [(2, 1, 0.4),
                (3, 1, 0.3),
                (4, 3, 0.999),
                (5, 1, 0.15),
                (5, 2, 0.16),
                (5, 3, 0.04),
                (5, 4, 0.0),
                (6, 1, 0.43)],
               'B_pos_value_S': [],
               'pos_value_A': [(1, 83.33333333333333),
                (2, 40000.0),
                (3, 500.0),
                (4, 500000.0),
                (5, 1315.7894736842104),
                (6, 14814.814814814814)],
               'pos_value_S': [(1, 400.0)]}),
             ('ET',
              OrderedDict([('CR50',
                            {'indices': [5, 0, 3, 2],
                             'mult': 6,
                             'type': 'AA',
                             'value': 887920884.0}),
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
                           ('BackET',
                            {'indices': [3, 0, 0, 1],
                             'mult': 6,
                             'type': 'AS',
                             'value': 4502.20614}),
                           ('EM',
                            {'indices': [1, 0, 0, 1],
                             'mult': 6,
                             'type': 'SS',
                             'value': 45022061400.0})]))])

    cte['no_console'] = True
    cte['no_plot'] = True
    return cte


def idfn(params):
    '''Returns the name of the test according to the parameters'''
    return '{}S_{}A_{}Nuc_{}Ss_{}As'.format(params[0], params[1], params[2],
                                            params[3], params[4])

# S_conc, A_conc, N_uc, S_states, A_states, test should return a exception
@pytest.mark.parametrize('params', [(10.5, 5.2, 7, 2, 7, True), # normal
                                    (0.1, 0.0, 40, 2, 7, True), # no A, change S and N_uc
                                    (5.0, 0.0, 10, 2, 7, True), # no A, change S and N_uc
                                    (20.0, 0.0, 8, 2, 7, True), # no A, change S and N_uc
                                    (100.0, 0.0, 5, 2, 7, True), # no A, change S and N_uc
                                    (0.0, 0.1, 40, 2, 7, True), # no S, change A and N_uc
                                    (0.0, 0.5, 25, 2, 7, True), # no S, change A and N_uc
                                    (0.0, 20.0, 6, 2, 7, True), # no S, change A and N_uc
                                    (0.0, 100.0, 3, 2, 7, True), # no S, change A and N_uc
                                    # TEST NEGATIVE AND EXCESSIVE CONCS AND N_UC
                                    (0.0, 0.0, 10, 2, 7, False), # no S nor A
                                    (25.0, 100.0, 10, 2, 7, False), # too much S+A
                                    (0.0, 0.0, 0, 2, 7, False), # no S, A, N_uc
                                    (25.0, 100.0, 0, 2, 7, False), # too much S+A, no N_uc
                                    (-25.0, 0.0, 10, 2, 7, False), # negative S
                                    (0.0, -50.0, 20, 2, 7, False), # negative A
                                    (125.0, 50.0, 5, 2, 7, False), # too much S
                                    (0.0, 50.0, -20, 2, 7, False), # negative N_uc
                                    (0.03, 0.0, 1, 2, 7, False), # most likely no doped ions created
                                    # TEST NUMBER OF ENERGY STATES
                                    (5.0, 5.0, 10, 5, 0, True), # no A_states
                                    (10.0, 1.0, 8, 0, 4, True), # no S_states
                                    (6.0, 6.0, 5, 0, 0, False), # no S_states, A_states
                                    (5.0, 5.0, 10, 15, 0, True)], # no A_states
                         ids=idfn)
def test_cte(setup_cte, params):
    '''Test a lattice with different concentrations of S and A; different number of unit cells;
        and different number of S and A energy states
    '''

    cte = setup_cte

    cte['lattice']['S_conc'] = params[0]
    cte['lattice']['A_conc'] = params[1]
    cte['lattice']['N_uc'] = params[2]
    cte['states']['sensitizer_states'] = params[3]
    cte['states']['activator_states'] = params[4]
    # True for normal results, False for exception
    normal_result = params[5]

    cte['lattice']['name'] = 'test_lattice'

    if normal_result:

        (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
        index_S_i, index_A_j,
        index_S_k, dist_S_k,
        index_S_l, dist_S_l,
        index_A_k, dist_A_k,
        index_A_l, dist_A_l) = lattice.generate(cte)

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


    else: # tests that result in an exception

        with pytest.raises(lattice.LatticeError):
            (dist_array, ion_type, doped_lattice, initial_population, lattice_info,
            index_S_i, index_A_j,
            index_S_k, dist_S_k,
            index_S_l, dist_S_l,
            index_A_k, dist_A_k,
            index_A_l, dist_A_l) = lattice.generate(cte)

