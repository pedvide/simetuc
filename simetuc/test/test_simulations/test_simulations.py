# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:22:54 2016

@author: Pedro
"""
import os

import pytest
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import h5py
import numpy as np

import simetuc.simulations as simulations
import simetuc.plotter as plotter
from simetuc.util import temp_config_filename, temp_bin_filename, IonType, DecayTransition
from simetuc.util import Excitation
from simetuc.settings import Settings
### TODO: Test loading exp data with different formats


test_folder_path = os.path.dirname(os.path.abspath(__file__))

def test_sim(setup_cte_sim):
    '''Test that the simulations work'''
    setup_cte_sim['lattice']['S_conc'] = 0
    sim = simulations.Simulations(setup_cte_sim)
    assert sim.cte
    assert sim

def test_sim_dyn(setup_cte_sim):
    '''Test that the dynamics work'''
    setup_cte_sim['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        assert sim.cte == setup_cte_sim

        solution = sim.simulate_dynamics()
        assert solution

    solution.plot()
    solution.plot(state=7)
    solution.plot(state=1)
    plotter.plt.close('all')

def test_sim_dyn_errors(setup_cte_sim):
    '''Test that the dynamics work'''
    setup_cte_sim['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_dynamics()

        solution.errors
        assert isinstance(solution.errors, np.ndarray)
        solution.log_errors()


def test_change_cte(setup_cte_sim):
    '''Test that the cte is copied into the Solution.'''
    dynamics_sol = simulations.DynamicsSolution(np.zeros((100,)), np.zeros((100,10)),
                                                [-1]*10, [-1]*10, setup_cte_sim)

    setup_cte_sim.excitations['Vis_473'][0].power_dens = 500

    dynamics_sol2 = simulations.DynamicsSolution(np.zeros((100,)), np.zeros((100,10)),
                                                 [-1]*10, [-1]*10, setup_cte_sim)

    assert dynamics_sol != dynamics_sol2
    assert dynamics_sol2.cte.excitations['Vis_473'][0].power_dens == 500


def test_sim_dyn_2S_2A(setup_cte_sim):
    '''Test that the dynamics have the right result for a simple system'''
    test_filename = os.path.join(test_folder_path, 'data_2S_2A.hdf5')
    sim = simulations.Simulations(setup_cte_sim, full_path=test_filename)

    solution = sim.simulate_dynamics()
    assert solution.index_A_j == [0, -1, -1, 11]
    assert solution.index_S_i == [-1, 7, 9, -1]

    with h5py.File(os.path.join(test_folder_path, 't_sol_2S_2A.hdf5')) as file:
         t_sol = np.array(file['t_sol'])
    assert np.allclose(t_sol, solution.t_sol)

    with h5py.File(os.path.join(test_folder_path, 'y_sol_2S_2A.hdf5')) as file:
         y_sol = np.array(file['y_sol'])
    assert np.allclose(y_sol, solution.y_sol)

def test_sim_dyn_wrong_state_plot(setup_cte_sim):
    '''Test that you can't plot a wrong state.'''
    setup_cte_sim['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with pytest.raises(ValueError):
        solution.plot(state=10)
    plotter.plt.close('all')

def test_sim_average_dyn(setup_cte_sim):
    '''Test average dynamics.'''
    setup_cte_sim['lattice']['S_conc'] = 0

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_avg_dynamics()
        assert solution

def test_sim_dyn_diff(setup_cte_sim):
    '''Test that the two dynamics are different'''
    setup_cte_sim['lattice']['S_conc'] = 0
    setup_cte_sim['lattice']['A_conc'] = 0.2
    with temp_bin_filename() as temp_filename:
        sim1 = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution1 = sim1.simulate_dynamics()
        solution1.total_error

    setup_cte_sim['ions'] = {}
    setup_cte_sim['lattice']['S_conc'] = 0.2
    setup_cte_sim['lattice']['A_conc'] = 0
    with temp_bin_filename() as temp_filename:
        sim2 = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution2 = sim2.simulate_dynamics()
        solution2.total_error

    assert sim1 != sim2
    assert solution1 != solution2


def test_sim_dyn_save_hdf5(setup_cte_sim, mocker):
    '''Test that the dynamics solution is saved a loaded correctly'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim.states['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_dynamics()
        assert mocked.call_count == 2

        with temp_bin_filename() as filename:
            solution.save(filename)
            sol_hdf5 = simulations.DynamicsSolution.load(filename)
        assert sol_hdf5

        assert sol_hdf5.cte == solution.cte
        assert np.allclose(sol_hdf5.y_sol, solution.y_sol)
        assert np.allclose(sol_hdf5.t_sol, solution.t_sol)
        assert sol_hdf5.index_S_i == solution.index_S_i
        assert sol_hdf5.index_A_j == solution.index_A_j
        assert sol_hdf5 == solution
        sol_hdf5.log_errors()
        sol_hdf5.plot()
        plotter.plt.close('all')


def test_sim_dyn_save_txt(setup_cte_sim):
    '''Test that the dynamics solution is saved a loaded correctly'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_no_file_hdf5():
    '''Wrong filename'''
    with pytest.raises(OSError):
        simulations.DynamicsSolution.load(os.path.join(test_folder_path, 'wrongFile.hdf5'))

def test_sim_dyn_no_t_pulse(setup_cte_sim):
    '''Test that the dynamics gives an error if t_pulse is not defined'''
    del setup_cte_sim['excitations']['Vis_473'][0].t_pulse

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        with pytest.raises(AttributeError):
            sim.simulate_dynamics()

def test_sim_steady1(setup_cte_sim):
    '''Test that the steady state solution is saved a loaded correctly'''
    setup_cte_sim['excitations']['Vis_473'][0].t_pulse = 1e-08
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        assert sim.cte == setup_cte_sim
        solution = sim.simulate_steady_state()
        assert solution

    solution.log_populations()
    solution.plot()
    solution.log_populations()  # redo

    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.SteadyStateSolution.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()
    plotter.plt.close('all')

def test_sim_steady2(setup_cte_sim):
    '''Test average steady state'''
    setup_cte_sim['excitations']['Vis_473'][0].t_pulse = 1e-08
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_avg_steady_state()
        assert solution

def test_sim_no_plot(setup_cte_sim):
    '''Test that no plot works'''
    setup_cte_sim['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_dynamics()

    with pytest.warns(plotter.PlotWarning) as warnings:
        solution.plot()
#    assert len(warnings) == 1 # one warning
    warning = warnings.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)

    plotter.plt.close('all')

@pytest.mark.parametrize('average', [True, False])
@pytest.mark.parametrize('excitation_name', ['NIR_800', 'Vis_473'])
def test_sim_power_dep(setup_cte_sim, mocker, average, excitation_name):
    '''Test that the power dependence works'''
    for exc_name, exc_list in setup_cte_sim.excitations.items():
        for exc in exc_list:
            if exc_name is excitation_name:
                exc.active = True
            else:
                exc.active = False

    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim.states['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        assert sim.cte == setup_cte_sim
        power_dens_list = np.logspace(1, 3, 3-1+1)
        solution = sim.simulate_power_dependence(power_dens_list, average=average)
        assert (mocked.call_count == 2*len(power_dens_list)) or (mocked.call_count == len(power_dens_list))
        assert solution

    solution.plot()

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

    with temp_config_filename('') as filename:
        solution.save(filename)
        solution_hdf5 = simulations.PowerDependenceSolution.load(filename)

    assert solution_hdf5
    for sol, sol_hdf5 in zip(solution.solution_list, solution_hdf5.solution_list):
        assert sol.y_sol.shape == sol_hdf5.y_sol.shape
        assert np.allclose(sol.t_sol, sol_hdf5.t_sol)
        assert np.allclose(sol.y_sol, sol_hdf5.y_sol)
        assert sol.cte == sol_hdf5.cte
        print(type(sol.index_S_i), type(sol_hdf5.index_S_i))
        print(sol.index_S_i, sol_hdf5.index_S_i)
        assert sol.index_S_i == sol_hdf5.index_S_i
        assert sol.index_A_j == sol_hdf5.index_A_j

        assert sol == sol_hdf5

    assert solution_hdf5 == solution
    solution_hdf5.plot()

    plotter.plt.close('all')

def test_sim_power_dep_save_txt(setup_cte_sim, mocker):
    '''Test that the power dep solution is saved as text correctly'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        power_dens_list = np.logspace(1, 2, 3)
        solution = sim.simulate_power_dependence(power_dens_list, average=True)

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_power_dep_empty_list(setup_cte_sim):
    '''Power dep list is empty'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        power_dens_list = []
        solution = sim.simulate_power_dependence(power_dens_list)

    with pytest.warns(plotter.PlotWarning) as warnings:
        solution.plot()
#    assert len(warnings) == 1 # one warning
    warning = warnings.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'Nothing to plot! The power_dependence list is emtpy!' in str(warning.message)
    plotter.plt.close('all')

def test_sim_power_dep_no_plot(setup_cte_sim, mocker):
    '''A plot was requested, but no_plot is set'''
    setup_cte_sim['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.simulations.Simulations.simulate_pulsed_steady_state')
        mocked.return_value = simulations.SteadyStateSolution(np.empty((1000,)),
                                                              np.empty((1000, 2*setup_cte_sim['states']['energy_states'])),
                                                              [], [], setup_cte_sim)

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        power_dens_list = np.logspace(1, 2, 3)
        solution = sim.simulate_power_dependence(power_dens_list)
        assert mocked.call_count == len(power_dens_list)

    with pytest.warns(plotter.PlotWarning) as warnings:
        solution.plot()
#    assert len(warnings) == 1 # one warning
    warning = warnings.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)
    plotter.plt.close('all')

def test_sim_power_dep_correct_power_dens(setup_cte_sim, mocker):
    '''Check that the solutions have the right power_dens'''
    setup_cte_sim['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        power_dens_list = np.logspace(1, 3, 3)
        solution = sim.simulate_power_dependence(power_dens_list)
        assert mocked.call_count == 2*len(power_dens_list)

    for num, pow_dens in enumerate(power_dens_list):
        assert solution[num].power_dens == pow_dens

def test_sim_power_dep_ESA():
    '''Make sure the simulated solution for a simple problem is equal to the theoretical one.'''
    test_filename = os.path.join(test_folder_path, 'data_0S_1A.hdf5')

    cte = {'version': 1,
           'decay': {'decay_A': {DecayTransition(IonType.A, 1, 0, decay_rate=1e3),
                           DecayTransition(IonType.A, 2, 0, decay_rate=1e6)},
                     'decay_S': {DecayTransition(IonType.S, 1, 0, decay_rate=1e1)},
                     'branching_S': {}, 'branching_A': {}},
           'excitations': {
                  'ESA': [Excitation(IonType.A, 0, 1, True, 0, 1e-3, 1e6, t_pulse=None),
                             Excitation(IonType.A, 1, 2, True, 0, 1e-3, 1e6, t_pulse=None)]},
           'energy_transfer': {},
           'lattice': {'A_conc': 0.3,
                       'N_uc': 20,
                       'S_conc': 0.0,
                       'a': 5.9738,
                       'alpha': 90.0,
                       'b': 5.9738,
                       'beta': 90.0,
                       'c': 3.5297,
                       'gamma': 120.0,
                       'name': 'bNaYF4',
                       'sites_occ': [1.0, 0.5],
                       'sites_pos': [(0.0, 0.0, 0.0), (2/3, 1/3, 0.5)],
                       'spacegroup': 'P-6'},
           'no_console': False,
           'no_plot': False,
           'simulation_params': {'N_steps': 1000,
                                 'N_steps_pulse': 100,
                                 'atol': 1e-15,
                                 'rtol': 0.001},
           'states': {'activator_ion_label': 'Tm',
                      'activator_states': 3,
                      'activator_states_labels': ['GS', 'ES1', 'ES2'],
                      'energy_states': 3,
                      'sensitizer_ion_label': 'Yb',
                      'sensitizer_states': 2,
                      'sensitizer_states_labels': ['GS', 'ES']}}
    simple_cte =  Settings.load_from_dict(cte)

    power_dens_list = np.logspace(1, 6, 6)

    sim = simulations.Simulations(simple_cte, full_path=test_filename)

    solution = sim.simulate_power_dependence(power_dens_list, average=True)

    # check that the ES1 and ES2 populations are close to the theoretical values
    for sol in solution:
        GS = sol.steady_state_populations[2]
        ES1 = sol.steady_state_populations[3]
        ES2 = sol.steady_state_populations[4]
        P = sol.power_dens*sol.cte.excitations['ESA'][0].pump_rate
        k1 = 1e3
        k2 = 1e6
        theo_ES1 = GS*P/(k1+P)
        theo_ES2 = GS*P**2/((k1+P)*k2)
#        print('ES1: {}, theo_ES1: {}'.format(ES1, theo_ES1))
#        print('ES2: {}, theo_ES2: {}'.format(ES2, theo_ES2))
        assert np.allclose(theo_ES1, ES1, rtol=1e-4)
        assert np.allclose(theo_ES2, ES2, rtol=1e-4)


def test_sim_conc_dep_steady(setup_cte_sim, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == 2*len(conc_list)

    assert solution
    solution.plot()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()
    plotter.plt.close('all')

def test_sim_conc_dep_dyn(setup_cte_sim, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
        # dynamics call solve_ode twice (pulse and relaxation)
        assert mocked.call_count == 2*len(conc_list)

    assert solution
    solution.plot()
    solution.log_errors()
    with temp_config_filename('') as filename:
        solution.save(filename)
        sol_hdf5 = simulations.ConcentrationDependenceSolution.load(filename)

    assert sol_hdf5
    assert sol_hdf5 == solution
    sol_hdf5.plot()
    plotter.plt.close('all')

def test_sim_conc_dep_save_txt(setup_cte_sim, mocker):
    '''Test that the conc dep solution is saved as text correctly'''
    with temp_bin_filename() as temp_filename:
        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False, average=True)

    with temp_config_filename('') as filename:
        solution.save_txt(filename)

def test_sim_conc_dep_list(setup_cte_sim, mocker):
    '''Test that the concentration dependence works'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=True)
        # dynamics call solve_ode twice (pulse and relaxation)
        assert mocked.call_count == 2*len(conc_list)

        for num, conc in enumerate(conc_list):
            assert solution[num].concentration == conc

def test_sim_conc_dep_only_A(setup_cte_sim, mocker):
    '''Conc list has only A changing'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        conc_list = [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == 2*len(conc_list)

    assert solution
    solution.plot()
    plotter.plt.close('all')

def test_sim_conc_dep_only_S(setup_cte_sim, mocker):
    '''Conc list has only S changing'''
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        conc_list = [(0.01, 0.3), (0.1, 0.3), (0.3, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == 2*len(conc_list)

    assert solution
    solution.plot()
    plotter.plt.close('all')

def test_sim_conc_dep_empty_conc(setup_cte_sim):
    '''Conc list is empty'''
    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        conc_list = []
        solution = sim.simulate_concentration_dependence(conc_list)

    with pytest.warns(plotter.PlotWarning) as warnings:
        solution.plot()
#    assert len(warnings) == 1 # one warning
    warning = warnings.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'Nothing to plot! The concentration_dependence list is emtpy!' in str(warning.message)
    plotter.plt.close('all')


def test_sim_conc_dep_no_plot(setup_cte_sim, mocker):
    '''A plot was requested, but no_plot is set'''
    setup_cte_sim['no_plot'] = True
    with temp_bin_filename() as temp_filename:
        mocked = mocker.patch('simetuc.odesolver._solve_ode')
        # the num_states changes when the temp lattice is created,
        # allocate 2x so that we're safe. Also make the num_points 1000.
        mocked.return_value = np.random.random((1000, 2*setup_cte_sim['states']['energy_states']))

        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        conc_list = [(0.01, 0.3), (0.1, 0.3)]
        solution = sim.simulate_concentration_dependence(conc_list, dynamics=False)
        assert mocked.call_count == 2*len(conc_list)

    with pytest.warns(plotter.PlotWarning) as warnings:
        solution.plot()
#    assert len(warnings) == 1 # one warning
    warning = warnings.pop(plotter.PlotWarning)
    assert issubclass(warning.category, plotter.PlotWarning)
    assert 'A plot was requested, but no_plot setting is set' in str(warning.message)
    plotter.plt.close('all')


def test_sim_conc_dep_no_file():
    '''Wrong filename'''
    with pytest.raises(OSError):
        simulations.PowerDependenceSolution.load(os.path.join(test_folder_path, 'wrongFile.hdf5'))


@pytest.mark.parametrize('N_samples', [1, 2, 10])
def test_sim_sample_dynamics(setup_cte_sim, mocker, N_samples):
    '''Test that sampling the dynamics works'''
    setup_cte_sim['lattice']['S_conc'] = 0

    mocked = mocker.patch('simetuc.odesolver._solve_ode')
    # the num_states changes when the temp lattice is created,
    # allocate 2x so that we're safe. Also make the num_points 1000.
    mocked.return_value = np.random.random((1000, 2*setup_cte_sim.states['energy_states']))

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        assert sim.cte == setup_cte_sim

        solution = sim.sample_simulation(sim.simulate_dynamics, N_samples=N_samples)
        assert solution
        assert mocked.call_count == 2*N_samples

@pytest.mark.parametrize('N_samples', [1, 2, 10])
def test_sim_sample_conc_dynamics(setup_cte_sim, mocker, N_samples):
    '''Test that sampling the dynamics works'''
    setup_cte_sim['lattice']['S_conc'] = 0

    mocked = mocker.patch('simetuc.odesolver._solve_ode')
    # the num_states changes when the temp lattice is created,
    # allocate 2x so that we're safe. Also make the num_points 1000.
    mocked.return_value = np.random.random((1000, 2*setup_cte_sim.states['energy_states']))

    with temp_bin_filename() as temp_filename:
        sim = simulations.Simulations(setup_cte_sim, full_path=temp_filename)
        assert sim.cte == setup_cte_sim
        conc_list = [(0, 0.3), (0.1, 0.3), (0.1, 0)]
        solution = sim.sample_simulation(sim.simulate_concentration_dependence, N_samples=N_samples,
                                         concentrations=conc_list)
        assert solution
        assert mocked.call_count == 2*len(conc_list)*N_samples
