# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:22:54 2016

@author: Pedro
"""
import os

import pytest
import numpy as np

import simetuc.simulations as simulations
import simetuc.precalculate as precalculate


test_folder_path = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope='module')
def setup_cte():
    '''Load the cte data structure'''

    cte = {'ET': dict([('CR50',
               {'indices': [5, 0, 3, 2],
                'mult': 6,
                'type': 'AA',
                'value': 2893199540.0}),
              ('ETU53',
               {'indices': [5, 3, 6, 1],
                'mult': 6,
                'type': 'AA',
                'value': 254295690.0}),
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
                'value': 45022061400.0}),
              ('ETU1',
               {'indices': [1, 0, 0, 2],
                'mult': 6,
                'type': 'SA',
                'value': 10000.0})]),
         'decay': {'B_pos_value_A': [(2, 1, 0.4),
           (3, 1, 0.3),
           (3, 2, 0.1),
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
          'pos_value_S': [(1, 400.0)]},
         'excitations': {'NIR_1470': {'active': False,
           'degeneracy': [1.8],
           'final_state': [6],
           'init_state': [5],
           'ion_exc': ['A'],
           'power_dens': 10000000.0,
           'process': ['Tm(1G4) -> Tm(1D2)'],
           'pump_rate': [0.0002],
           't_pulse': 1e-08},
          'NIR_800': {'active': False,
           'degeneracy': [1.4444444444444444, 1.2222222222222223],
           'final_state': [3, 5],
           'init_state': [0, 2],
           'ion_exc': ['A', 'A'],
           'power_dens': 10000000.0,
           'process': ['Tm(3H6)->Tm(3H4)', 'Tm(3H5)->Tm(1G4)'],
           'pump_rate': [0.0044, 0.004],
           't_pulse': 1e-08},
          'NIR_980': {'active': False,
           'degeneracy': [1.3333333333333333],
           'final_state': [1],
           'init_state': [0],
           'ion_exc': ['S'],
           'power_dens': 10000000.0,
           'process': ['Yb(GS)->Yb(ES)'],
           'pump_rate': [0.0044],
           't_pulse': 1e-08},
          'Vis_473': {'active': True,
           'degeneracy': [1.4444444444444444],
           'final_state': [5],
           'init_state': [0],
           'ion_exc': ['A'],
           'power_dens': 1000000.0,
           'process': ['Tm(3H6) -> Tm(1G4)'],
           'pump_rate': [0.00093],
           't_pulse': 1e-08}},
         'lattice': {'A_conc': 0.3,
          'N_uc': 30,
          'S_conc': 0.3,
          'cell_par': [5.9738, 5.9738, 3.5297, 90.0, 90.0, 120.0],
          'name': 'bNaYF4',
          'sites_occ': [1.0, 0.5],
          'sites_pos': [(0.0, 0.0, 0.0),
           (0.6666666666666666, 0.3333333333333333, 0.5)],
          'spacegroup': 'P-6'},
         'no_console': False,
         'no_plot': False,
         'states': {'activator_ion_label': 'Tm',
          'activator_states': 7,
          'activator_states_labels': ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2'],
          'sensitizer_ion_label': 'Yb',
          'sensitizer_states': 2,
          'sensitizer_states_labels': ['GS', 'ES']}}

    return cte

@pytest.fixture(scope='module')
def setup_benchmark(setup_cte):
    test_filename = os.path.join(test_folder_path, 'data_240S_108A.hdf5')

    (cte, initial_population, index_S_i, index_A_j,
     total_abs_matrix, decay_matrix,
     UC_matrix, N_indices,
     jac_indices) = precalculate.setup_microscopic_eqs(setup_cte, full_path=test_filename)

    tf = (10*np.max(precalculate.get_lifetimes(cte))).round(8)  # total simulation time
    tf_p = 10e-9
    t0_sol = tf_p
    tf_sol = tf
    N_steps = 1000

    rtol = 1e-3
    atol = 1e-15

    # initial conditions after a pulse
    ic = np.zeros((cte['states']['energy_states']), dtype=np.float64)
    index_S_i = np.array(index_S_i)
    index_A_j = np.array(index_A_j)
    ic[index_S_i[index_S_i != -1]] = 1.0
    ic[index_A_j[index_A_j != -1]] = 0.99999054943581
    ic[index_A_j[index_A_j != -1]+1] = 9.19e-12
    ic[index_A_j[index_A_j != -1]+2] = 2.3e-10
    ic[index_A_j[index_A_j != -1]+3] = 3.25e-10
    ic[index_A_j[index_A_j != -1]+5] = 9.45e-06
    ic[index_A_j[index_A_j != -1]+6] = 0.0

    t_sol = np.logspace(np.log10(t0_sol), np.log10(tf_sol), N_steps, dtype=np.float64)
    args_sol = (t_sol, simulations._rate_eq, (decay_matrix, UC_matrix, N_indices),
                simulations._jac_rate_eq, (decay_matrix, UC_matrix, jac_indices),
                ic)
    kwargs_sol = {'method': 'bdf', 'rtol': rtol, 'atol': atol, 'nsteps': 1000, 'quiet': True}

    return (args_sol, kwargs_sol)

#@pytest.mark.benchmark(group="fast", min_rounds=100)
#def test_dyn_benchmark_small(setup_cte, benchmark):
#    '''Benchmark the dynamics for a simple system'''
#    test_filename = os.path.join(test_folder_path, 'data_2S_2A.hdf5')
#    sim = simulations.Simulations(setup_cte, full_path=test_filename)
#    benchmark(sim.simulate_dynamics)

@pytest.mark.skip
@pytest.mark.benchmark(group="slow")
def test_benchmark_ode_solve_large(setup_benchmark, benchmark):
    '''Benchmark the dynamics for a medium-sized system'''
    benchmark.pedantic(simulations._solve_ode, args=setup_benchmark[0],
                       kwargs=setup_benchmark[1],
                       rounds=20, iterations=1)

