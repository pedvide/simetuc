# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:22:54 2016

@author: Pedro
"""
import pytest
import os
import numpy as np

import simetuc.simulations as simulations
from simetuc.util import temp_config_filename

test_folder_path = os.path.dirname(os.path.abspath(__file__))

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
         'optimization_processes': ['CR50', 'ETU53'],
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

    cte['no_console'] = True
    cte['no_plot'] = False
    return cte

def test_sim(setup_cte):
    '''Test that the simulations work'''
    setup_cte['lattice']['S_conc'] = 0
    sim = simulations.Simulations(setup_cte)

    assert sim

def test_sim_dyn1(setup_cte):
    '''Test that the dynamics work'''
    setup_cte['lattice']['S_conc'] = 0
    sim = simulations.Simulations(setup_cte)

    assert sim.cte == setup_cte

    solution = sim.simulate_dynamics()
    assert solution

    solution.total_error
    solution.log_errors()
    solution.plot()
    solution.plot(state=6)
    solution.plot(state=1)

def test_sim_dyn2(setup_cte):
    '''Test that the dynamics have the right result for a simple system'''
    test_filename = os.path.join(test_folder_path, 'data_2S_2A.npz')
    sim = simulations.Simulations(setup_cte, full_path=test_filename)

    solution = sim.simulate_dynamics()
    assert solution.index_A_j == [-1, 2, -1, 11]
    assert solution.index_S_i == [0, -1, 9, -1]

    t_sol = np.load(os.path.join(test_folder_path, 't_sol_2S_2A.npy'))
    assert np.allclose(t_sol, solution.t_sol)

    y_sol = np.load(os.path.join(test_folder_path, 'y_sol_2S_2A.npy'))
    assert np.allclose(y_sol, solution.y_sol)

def test_sim_dyn_diff(setup_cte):
    '''Test that the two dynamics are different'''
    setup_cte['lattice']['S_conc'] = 0
    sim1 = simulations.Simulations(setup_cte)
    solution1 = sim1.simulate_dynamics()

    setup_cte['lattice']['S_conc'] = 0.3
    sim2 = simulations.Simulations(setup_cte)
    solution2 = sim2.simulate_dynamics()

    assert sim1 != sim2
    assert solution1 != solution2


def test_sim_dyn_save_hdf5(setup_cte):
    '''Test that the dynamics solution is saved a loaded correctly'''

    sim = simulations.Simulations(setup_cte)
    solution = sim.simulate_dynamics()

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.DynamicsSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.log_errors()
    sol_hdf5.plot()


def test_sim_dyn_save_txt(setup_cte):
    '''Test that the dynamics solution is saved a loaded correctly'''

    sim = simulations.Simulations(setup_cte)
    solution = sim.simulate_dynamics()

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_no_file_hdf5():
    '''Wrong filename'''
    with pytest.raises(OSError):

        simulations.DynamicsSolution().load(os.path.join(test_folder_path, 'wrongFile.hdf5'))

def test_sim_dyn_no_t_pulse(setup_cte):
    '''Test that the dynamics gives an error if t_pulse is not defined'''
    del setup_cte['excitations']['Vis_473']['t_pulse']
    sim = simulations.Simulations(setup_cte)

    with pytest.raises(KeyError):
        sim.simulate_dynamics()

def test_sim_steady(setup_cte):
    '''Test that the steady state solution is saved a loaded correctly'''
    setup_cte['excitations']['Vis_473']['t_pulse'] = 1e-08
    sim = simulations.Simulations(setup_cte)

    assert sim.cte == setup_cte

    solution = sim.simulate_steady_state()
    assert solution

    solution.log_populations()
    solution.plot()

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.SteadyStateSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_no_plot(setup_cte, recwarn):
    '''Test that no plot works'''
    setup_cte['no_plot'] = True
    sim = simulations.Simulations(setup_cte)

    solution = sim.simulate_dynamics()

    with pytest.warns(simulations.PlotWarning):
        solution.plot()
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(simulations.PlotWarning)
    assert issubclass(warning.category, simulations.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)

def test_sim_power_dep1(setup_cte):
    '''Test that the power dependence works'''
    sim = simulations.Simulations(setup_cte)

    assert sim.cte == setup_cte

    power_dens_list = np.logspace(1, 5, 5-1+1)
    solution = sim.simulate_power_dependence(power_dens_list)

    assert solution

    solution.plot()

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.PowerDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_power_dep2(setup_cte, recwarn):
    '''Power dep list is empty'''

    sim = simulations.Simulations(setup_cte)

    power_dens_list = []
    solution = sim.simulate_power_dependence(power_dens_list)

    with pytest.warns(simulations.PlotWarning):
        solution.plot()
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(simulations.PlotWarning)
    assert issubclass(warning.category, simulations.PlotWarning)
    assert 'Nothing to plot! The power_dependence list is emtpy!' in str(warning.message)

def test_sim_power_dep3(setup_cte, recwarn):
    '''A plot was requested, but no_plot is set'''

    setup_cte['no_plot'] = True
    sim = simulations.Simulations(setup_cte)

    power_dens_list = np.logspace(1, 2, 3)
    solution = sim.simulate_power_dependence(power_dens_list)

    with pytest.warns(simulations.PlotWarning):
        solution.plot()
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(simulations.PlotWarning)
    assert issubclass(warning.category, simulations.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)

def test_sim_power_dep4(setup_cte):
    '''Check save_txt'''

    setup_cte['no_plot'] = True
    sim = simulations.Simulations(setup_cte)

    power_dens_list = np.logspace(1, 2, 3)
    solution = sim.simulate_power_dependence(power_dens_list)

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_power_dep5(setup_cte):
    '''Check that the solutions have the right power_dens'''

    setup_cte['no_plot'] = True
    sim = simulations.Simulations(setup_cte)

    power_dens_list = np.logspace(1, 2, 3)
    solution = sim.simulate_power_dependence(power_dens_list)

    for num, power_dens in enumerate(power_dens_list):
        solution[num].power_dens == power_dens

def test_sim_conc_dep1(setup_cte):
    '''Test that the concentration dependence works'''
    sim = simulations.Simulations(setup_cte)

    assert sim.cte == setup_cte

    conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
    solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
    assert solution
    solution.plot()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()


    solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
    assert solution
    solution.plot()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution()
        assert not sol_hdf5
        sol_hdf5.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()

def test_sim_conc_dep2(setup_cte):
    '''Conc list has only A changing'''
    sim = simulations.Simulations(setup_cte)

    conc_list = [(0.0, 0.01), (0.0, 0.1), (0.0, 0.3)]
    solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
    assert solution
    solution.plot()

def test_sim_conc_dep3(setup_cte):
    '''Conc list has only S changing'''
    sim = simulations.Simulations(setup_cte)

    conc_list = [(0.01, 0.3), (0.1, 0.3), (0.3, 0.3)]
    solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
    assert solution
    solution.plot()

def test_sim_conc_dep4(setup_cte, recwarn):
    '''Conc list is empty'''

    sim = simulations.Simulations(setup_cte)

    conc_list = []
    solution = sim.simulate_concentration_dependence(conc_list)

    with pytest.warns(simulations.PlotWarning):
        solution.plot()
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(simulations.PlotWarning)
    assert issubclass(warning.category, simulations.PlotWarning)
    assert 'Nothing to plot! The concentration_dependence list is emtpy!' in str(warning.message)


def test_sim_conc_dep5(setup_cte, recwarn):
    '''A plot was requested, but no_plot is set'''
    setup_cte['no_plot'] = True
    sim = simulations.Simulations(setup_cte)

    conc_list = [(0.01, 0.3), (0.1, 0.3)]
    solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)

    with pytest.warns(simulations.PlotWarning):
        solution.plot()
    assert len(recwarn) == 1 # one warning
    warning = recwarn.pop(simulations.PlotWarning)
    assert issubclass(warning.category, simulations.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)


def test_sim_conc_dep_no_file():
    '''Wrong filename'''
    with pytest.raises(OSError):
        simulations.PowerDependenceSolution().load(os.path.join(test_folder_path, 'wrongFile.hdf5'))

