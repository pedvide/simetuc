# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:22:18 2016

@author: Pedro
"""
import pytest
import numpy as np

import simetuc.optimize as optimize
import simetuc.simulations as simulations

@pytest.fixture(scope='function')
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
         'ions': {'activators': 113, 'sensitizers': 0, 'total': 113},
         'lattice': {'A_conc': 0.3,
          'N_uc': 20,
          'S_conc': 0.3,
          'cell_par': [5.9738, 5.9738, 3.5297, 90.0, 90.0, 120.0],
          'name': 'bNaYF4',
          'sites_occ': [1.0, 0.5],
          'sites_pos': [(0.0, 0.0, 0.0),
           (0.6666666666666666, 0.3333333333333333, 0.5)],
          'spacegroup': 'P-6'},
         'no_console': False,
         'no_plot': False,
         'optimization_processes': ['CR50'],
         'optimize_method': 'SLSQP',
         'simulation_params': {'N_steps': 1000,
          'N_steps_pulse': 100,
          'atol': 1e-15,
          'rtol': 0.001},
         'states': {'activator_ion_label': 'Tm',
          'activator_states': 7,
          'activator_states_labels': ['3H6', '3F4', '3H5', '3H4', '3F3', '1G4', '1D2'],
          'energy_states': 791,
          'sensitizer_ion_label': 'Yb',
          'sensitizer_states': 2,
          'sensitizer_states_labels': ['GS', 'ES']}}

    cte['no_console'] = False
    cte['no_plot'] = True
    return cte

@pytest.mark.parametrize('method', [None, 'COBYLA', 'L-BFGS-B',
                                    'TNC', 'SLSQP', 'brute_force', 'basin_hopping'])
def test_optim1(setup_cte, mocker, method):
    '''Test that the optimization works'''

    # mock the simulation by returning an error that goes to 0
    mocked_opt_dyn = mocker.patch('simetuc.optimize.optim_fun_factory')
    value = 20
    def minimize(x):
        nonlocal value
        if value != 0:
            value -= 1
        return value
    mocked_opt_dyn.return_value = minimize

    optimize.optimize_dynamics(setup_cte, method)

    assert mocked_opt_dyn.called

def test_optim_no_dict_params(setup_cte, mocker):
    '''Test that the optimization works without the optimization params being present in cte'''

    # mock the simulation by returning an error that goes to 0
    mocked_opt_dyn = mocker.patch('simetuc.optimize.optim_fun_factory')
    value = 20
    def minimize(x):
        nonlocal value
        if value != 0:
            value -= 1
        return value
    mocked_opt_dyn.return_value = minimize

    del setup_cte['optimization_processes']

    optimize.optimize_dynamics(setup_cte, method='SLSQP')

    assert mocked_opt_dyn.called


def test_opti_fun_factory(setup_cte, mocker):
    '''Test that the optimization works'''
    mocked_dyn = mocker.patch('simetuc.simulations.Simulations.simulate_dynamics')
    sim = simulations.Simulations(setup_cte)
    process_list = setup_cte['optimization_processes']
    x0 = np.array([setup_cte['ET'][process]['value'] for process in process_list])

    _update_ET_and_simulate = optimize.optim_fun_factory(sim, process_list, x0)
    _update_ET_and_simulate(x0*1.5)

    assert mocked_dyn.called

