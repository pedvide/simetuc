# -*- coding: utf-8 -*-
"""
Plot different solutions to rate equations problems and lattices

Created on Thu Dec  1 11:46:29 2016

@author: Pedro
"""

from typing import List

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from simetuc.util import Conc


class PlotWarning(UserWarning):
    '''Warning for empty plots'''
    pass


def plot_avg_decay_data(t_sol: np.ndarray, list_sim_data: List[np.array],
                        list_exp_data: List[np.array] = None,
                        state_labels: List[str] = None,
                        concentration: Conc = None,
                        atol: float = 1e-15,
                        colors: str = 'rk') -> None:
    ''' Plot the list of simulated and experimental data (optional) against time in t_sol.
        If concentration is given, the legend will show the concentrations
        along with the state_labels (it can get long).
        colors is a string with two chars. The first is the sim color,
        the second the exp data color.
    '''
    num_plots = len(list_sim_data)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    # optional lists default to list of None
    list_exp_data = list_exp_data or [None]*num_plots
    state_labels = state_labels or [None]*num_plots

    if concentration is not None:
        conc_str = '_' + str(concentration.S_conc) + 'S_' + str(concentration.A_conc) + 'A'
        state_labels = [label+conc_str for label in state_labels]

    for num, (sim_data, exp_data, state_label)\
        in enumerate(zip(list_sim_data, list_exp_data, state_labels)):
        if sim_data is 0:
            continue
        if (np.isnan(sim_data)).any() or not np.any(sim_data > 0):
            continue

        plt.subplot(num_rows, num_cols, num+1)

        sim_color = colors[0]
        exp_color = colors[1]

        # no exp data: either a GS or simply no exp data available
        if exp_data is 0 or exp_data is None:
            # nonposy='clip': clip non positive values to a very small positive number
            plt.semilogy(t_sol*1000, sim_data, sim_color, label=state_label, nonposy='clip')
            plt.yscale('log', nonposy='clip')
            plt.axis('tight')
            # add some white space above and below
            margin_factor = np.array([0.7, 1.3])
            plt.ylim(*np.array(plt.ylim())*margin_factor)
            if plt.ylim()[0] < atol:
                plt.ylim(ymin=atol)  # don't show noise below atol
                # detect when the simulation goes above and below atol
                above = sim_data > atol
                change_indices = np.where(np.roll(above, 1) != above)[0]
                if change_indices.size > 0:
                    # last time it changes
                    max_index = change_indices[-1]
                    # show simData until it falls below atol
                    plt.xlim(xmax=t_sol[max_index]*1000)
            min_y = min(*plt.ylim())
            max_y = max(*plt.ylim())
            plt.ylim(ymin=min_y, ymax=max_y)
        else:  # exp data available
            # convert exp_data time to ms
            plt.semilogy(exp_data[:, 0]*1000, exp_data[:, 1]*np.max(sim_data),
                         exp_color, t_sol*1000, sim_data, sim_color, label=state_label)
            plt.axis('tight')
            plt.ylim(ymax=plt.ylim()[1]*1.2)  # add some white space on top
            plt.xlim(xmax=exp_data[-1, 0]*1000)  # don't show beyond expData

        plt.legend(loc="best", fontsize='small')
        plt.xlabel('t (ms)')


def plot_state_decay_data(t_sol: np.ndarray, sim_data_array: np.ndarray,
                          state_label: str = None, atol: float = 1e-15) -> None:
    ''' Plots a state's simulated data against time t_sol'''
    t_sol *= 1000  # convert to ms

    if sim_data_array is 0:
        return
    if (np.isnan(sim_data_array)).any() or not np.any(sim_data_array):
        return

    avg_sim = np.mean(sim_data_array, axis=1)

    # nonposy='clip': clip non positive values to a very small positive number
    plt.semilogy(t_sol, sim_data_array, 'k', nonposy='clip')
    plt.semilogy(t_sol, avg_sim, 'r', nonposy='clip', linewidth=5)
    plt.yscale('log', nonposy='clip')
    plt.axis('tight')
    # add some white space above and below
    margin_factor = np.array([0.7, 1.3])
    plt.ylim(*np.array(plt.ylim())*margin_factor)
    if plt.ylim()[0] < atol:
        plt.ylim(ymin=atol)  # don't show noise below atol
        # detect when the simulation goes above and below atol
        above = sim_data_array > atol
        change_indices = np.where(np.roll(above, 1) != above)[0]
        if change_indices.size > 0:
            # last time it changes
            max_index = change_indices[-1]
            # show simData until it falls below atol
            plt.xlim(xmax=t_sol[max_index])

    plt.legend([state_label], loc="best")
    plt.xlabel('t (ms)')


def plot_power_dependence(sim_data_arr: np.ndarray, power_dens_arr: np.ndarray,
                          state_labels: List[str]) -> None:
    ''' Plots the intensity as a function of power density for each state'''
    num_plots = len(state_labels)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    # calculate the slopes for each consecutive pair of points in the curves
    Y = np.log10(sim_data_arr)[:-1, :]
    X = np.log10(power_dens_arr)
    dX = (np.roll(X, -1, axis=0) - X)[:-1]
    # list of slopes
    slopes = [np.gradient(Y_arr, dX) for Y_arr in Y.T]
    slopes = np.around(slopes, 1)

    for num, state_label in enumerate(state_labels):  # for each state
        sim_data = sim_data_arr[:, num]
        if not np.any(sim_data):
            continue

        axis = plt.subplot(num_rows, num_cols, num+1)

        plt.loglog(power_dens_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
        plt.axis('tight')
        margin_factor = np.array([0.7, 1.3])
        plt.ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
        plt.xlim(*np.array(plt.xlim())*margin_factor)

        plt.legend(loc="best")
        plt.xlabel('Power density / W/cm^2')

        for i, txt in enumerate(slopes[num]):
            axis.annotate(txt, (power_dens_arr[i], sim_data[i]), xytext=(5, -7),
                          xycoords='data', textcoords='offset points')


def plot_concentration_dependence(sim_data_arr: np.ndarray, conc_arr: np.ndarray,
                                  state_labels: List[str]) -> None:
    '''Plots the concentration dependence of the steady state emission'''
    num_plots = len(state_labels)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    heatmap = False
    if len(conc_arr.shape) == 2:
        heatmap = True

    for num, state_label in enumerate(state_labels):  # for each state
        sim_data = sim_data_arr[:, num]
        if not np.any(sim_data):
            continue

        ax = plt.subplot(num_rows, num_cols, num+1)

        if not heatmap:
            plt.plot(conc_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
            plt.axis('tight')
            margin_factor = np.array([0.9, 1.1])
            plt.ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
            plt.xlim(*np.array(plt.xlim())*margin_factor)

            plt.legend(loc="best")
            plt.xlabel('Concentration (%)')
            # change axis format to scientifc notation
            xfmt = plt.ScalarFormatter(useMathText=True)
            xfmt.set_powerlimits((-1, 1))
            ax.yaxis.set_major_formatter(xfmt)
        else:
            x, y = conc_arr[:, 0], conc_arr[:, 1]
            z = sim_data

            # Set up a regular grid of interpolation points
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            # random grid
            interp_f = interpolate.Rbf(x, y, z, function='gaussian', epsilon=2)
            zi = interp_f(xi, yi)
#                zi = interpolate.griddata((x, y), z, (xi, yi), method='cubic')
#                interp_f = interpolate.interp2d(x, y, z, kind='linear')
#                zi = interp_f(xi, yi)

            plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                       extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
            plt.scatter(x, y, c=z)
            plt.xlabel('S concentration (%)')
            plt.ylabel('A concentration (%)')
            cb = plt.colorbar()
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
            cb.set_label(state_label)


def plot_lattice(doped_lattice: np.array, ion_type: np.array) -> None:
    '''Plot a lattice of x,y,z points with the color
        depending on the corresponding vcalue of ion_type
    '''
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
