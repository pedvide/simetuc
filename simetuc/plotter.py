# -*- coding: utf-8 -*-
"""
Plot different solutions to rate equations problems and lattices

Created on Thu Dec  1 11:46:29 2016

@author: Pedro
"""

from typing import List, Union, Tuple, Type

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl

from simetuc.util import Conc

A_TOL = 1e-20

ColorMap = Type[mpl.colors.Colormap]


class PlotWarning(UserWarning):
    '''Warning for empty plots'''
    pass


def plot_avg_decay_data(t_sol: Union[np.ndarray, List[np.array]],
                        list_sim_data: List[np.array],
                        list_exp_data: List[np.array] = None,
                        state_labels: List[str] = None,
                        concentration: Conc = None,
                        atol: float = A_TOL,
                        colors: Union[str, Tuple[ColorMap, ColorMap]] = 'rk',
                        fig: mpl.figure.Figure = None,
                        title: str = '') -> None:
    ''' Plot the list of simulated and experimental data (optional) against time in t_sol.
        If concentration is given, the legend will show the concentrations.
        colors is a string with two chars. The first is the sim color,
        the second the exp data color.
    '''
    num_plots = len(list_sim_data)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    # optional lists default to list of None
    list_exp_data = list_exp_data or [None]*num_plots
    state_labels = state_labels or ['']*num_plots

    list_t_sim = t_sol if len(t_sol) == num_plots else [t_sol]*num_plots  # type: List[np.array]

    if concentration:
        conc_str = '_' + str(concentration.S_conc) + 'S_' + str(concentration.A_conc) + 'A'
#        state_labels = [label+conc_str for label in state_labels]
    else:
        conc_str = ''

    sim_color = colors[0]
    exp_color = colors[1]
    exp_size = 2  # marker size
    exp_marker = '.'

    if fig is None:
        fig = plt.figure()

    fig.suptitle(title + '. Time in ms.')

    list_axes = fig.get_axes()  # type: List
    if not list_axes:
        for num in range(num_plots):
            fig.add_subplot(num_rows, num_cols, num+1)
        list_axes = fig.get_axes()

    for sim_data, t_sim, exp_data, state_label, axes\
        in zip(list_sim_data, list_t_sim, list_exp_data, state_labels, list_axes):

        if state_label: 
            axes.set_title(state_label.replace('_', ' '),
                           {'horizontalalignment': 'center',
                            'verticalalignment': 'center',
                            'fontweight': 'bold',
                            'fontsize': 10})

        if sim_data is None or np.isnan(sim_data).any() or not np.any(sim_data > 0):
            continue

        # no exp data: either a GS or simply no exp data available
        if exp_data is 0 or exp_data is None:
            # nonposy='clip': clip non positive values to a very small positive number
            axes.semilogy(t_sim*1000, sim_data, color=sim_color, label=state_label+conc_str)
            
            axes.axis('tight')
            axes.set_xlim(left=t_sim[0]*1000.0)
            # add some white space above and below
            margin_factor = np.array([0.7, 1.3])
            axes.set_ylim(*np.array(axes.get_ylim())*margin_factor)
            if axes.set_ylim()[0] < atol:
                axes.set_ylim(bottom=atol)  # don't show noise below atol
                # detect when the simulation goes above and below atol
                above = sim_data > atol
                change_indices = np.where(np.roll(above, 1) != above)[0]
                # make sure change_indices[-1] happens when the population is going BELOW atol
                if change_indices.size > 1 and sim_data[change_indices[-1]] < atol:  # pragma: no cover
                    # last time it changes
                    max_index = change_indices[-1]
                    # show simData until it falls below atol
                    axes.set_xlim(right=t_sim[max_index]*1000)
            min_y = min(*axes.get_ylim())
            max_y = max(*axes.get_ylim())
            axes.set_ylim(bottom=min_y, top=max_y)
        else:  # exp data available
            sim_handle, = axes.semilogy(t_sim*1000, sim_data, color=sim_color,
                                       label=state_label+conc_str, zorder=10)
            # convert exp_data time to ms
            exp_handle, = axes.semilogy(exp_data[:, 0]*1000, exp_data[:, 1]*np.max(sim_data),
                                       color=exp_color, marker=exp_marker,
                                       linewidth=0, markersize=exp_size, zorder=1)
            axes.axis('tight')
            axes.set_ylim(top=axes.get_ylim()[1]*1.2)  # add some white space on top
            tmin = min(exp_data[-1, 0], t_sim[0])
            axes.set_xlim(left=tmin*1000.0, right=exp_data[-1, 0]*1000)  # don't show beyond expData

    if conc_str:
        list_axes[0].legend(loc="best", fontsize='small')
        curr_handles, curr_labels = list_axes[0].get_legend_handles_labels()
        new_labels = [label.replace(state_labels[0]+'_', '').replace('_', ', ') for label in curr_labels]
        list_axes[0].legend(curr_handles, new_labels, markerscale=5, loc="best", fontsize='small')
        
    fig.subplots_adjust(top=0.918, bottom=0.041,
                        left=0.034, right=0.99,
                        hspace=0.275, wspace=0.12)


def plot_state_decay_data(t_sol: np.ndarray, sim_data_array: np.ndarray,
                          state_label: str = None, atol: float = A_TOL) -> None:
    ''' Plots a state's simulated data against time t_sol'''

    if sim_data_array is None:  # pragma: no cover
        return
    if (np.isnan(sim_data_array)).any() or not np.any(sim_data_array):
        return

    avg_sim = np.mean(sim_data_array, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # nonposy='clip': clip non positive values to a very small positive number
    ax.semilogy(t_sol*1000, sim_data_array, 'k')
    ax.semilogy(t_sol*1000, avg_sim, 'r', linewidth=5)
    plt.yscale('log', nonposy='clip')
    plt.axis('tight')
    plt.xlim(xmin=0.0)
    # add some white space above and below
    margin_factor = np.array([0.7, 1.1])
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
            plt.xlim(xmax=t_sol[max_index]*1000)

    plt.legend([state_label], loc="best")
    plt.xlabel('t (ms)')


def plot_power_dependence(sim_data_arr: np.ndarray, power_dens_arr: np.ndarray,
                          state_labels: List[str]) -> None:
    ''' Plots the intensity as a function of power density for each state'''

    non_zero_data = np.array([np.any(sim_data_arr[:, num]) for num in range(sim_data_arr.shape[1])])
    sim_data_arr = sim_data_arr[:, non_zero_data]
    state_labels = np.array(state_labels)[non_zero_data]

    num_plots = len(state_labels)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    # calculate the slopes for each consecutive pair of points in the curves
    Y = np.log10(sim_data_arr)[:-1, :]
    X = np.log10(power_dens_arr)
    dX = list((np.roll(X, -1, axis=0) - X)[:-1])
    # list of slopes
    slopes = [np.gradient(Y_arr, dX[0]) for Y_arr in Y.T]
    slopes = np.around(slopes, 1)

    fig = plt.figure()
    for num in range(num_plots):
        fig.add_subplot(num_rows, num_cols, num+1)
    list_axes = fig.get_axes()

    for num, (state_label, ax) in enumerate(zip(state_labels, list_axes)):  # for each state
        sim_data = sim_data_arr[:, num]
        if not np.any(sim_data):  # pragma: no cover
            continue

        ax.loglog(power_dens_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
        plt.axis('tight')
        margin_factor = np.array([0.7, 1.3])
        plt.ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
        plt.xlim(*np.array(plt.xlim())*margin_factor)

        ax.legend(loc="best")
        plt.xlabel('Power density (W/cm\u00B2)')

        for i, txt in enumerate(slopes[num]):
            ax.annotate(txt, (power_dens_arr[i], sim_data[i]), xytext=(5, -7),
                        xycoords='data', textcoords='offset points')


def plot_concentration_dependence(sim_data_arr: np.ndarray, conc_arr: np.ndarray,
                                  state_labels: List[str],
                                  ion_label: Union[str, Tuple[str, str]] = None) -> None:
    '''Plots the concentration dependence of the steady state emission'''
    num_plots = len(state_labels)
    num_rows = 3
    num_cols = int(np.ceil(num_plots/3))

    fig = plt.figure()

    heatmap = False
    if len(conc_arr.shape) == 2:
        heatmap = True

    for num, state_label in enumerate(state_labels):  # for each state
        sim_data = sim_data_arr[:, num]
        if not np.any(sim_data):
            continue

        ax = fig.add_subplot(num_rows, num_cols, num+1)
        
        if state_label: 
            ax.set_title(state_label.replace('_', ' '),
                         {'horizontalalignment': 'center',
                          'verticalalignment': 'center',
                          'fontweight': 'bold', 'fontsize': 10})

        if not heatmap:
            ax.semilogy(conc_arr, sim_data, '.-r', mfc='k', ms=10, label=state_label)
            plt.axis('tight')
            margin_factor = np.array([0.9, 1.1])
            ax.set_ylim(*np.array(plt.ylim())*margin_factor)  # add some white space on top
            ax.set_xlim(*np.array(plt.xlim())*margin_factor)

            ion_label = ion_label if ion_label else ''
            ax.set_xlabel(f'{ion_label} concentration (%)')
            # change axis format to scientifc notation
#            xfmt = plt.ScalarFormatter(useMathText=True)
#            xfmt.set_powerlimits((-1, 1))
#            ax.yaxis.set_major_formatter(xfmt)
        else:
            x, y = conc_arr[:, 0], conc_arr[:, 1]
            z = sim_data

            # Set up a regular grid of interpolation points
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            # random grid
            interp_f = interpolate.Rbf(x, y, z, function='gaussian', epsilon=15)
            zi = interp_f(xi, yi)
#                zi = interpolate.griddata((x, y), z, (xi, yi), method='cubic')
#                interp_f = interpolate.interp2d(x, y, z, kind='linear')
#                zi = interp_f(xi, yi)

            plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                      extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
            ax.scatter(x, y, c=z, edgecolors='r', linewidth=0.25)
            ion_label = ion_label if ion_label else 'SA'
            ax.set_xlabel(f'{ion_label[0]} concentration (%)')
            ax.set_ylabel(f'{ion_label[1]} concentration (%)')
            cb = plt.colorbar()
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()
            cb.set_label('Emission intensity')
            
#        plt.tight_layout()
    fig.subplots_adjust(hspace=0.35, wspace=0.26)


def plot_lattice(doped_lattice: np.array, ion_type: np.array) -> None:
    '''Plot a lattice of x,y,z points with the color
        depending on the corresponding value of ion_type
    '''
    from mpl_toolkits.mplot3d import proj3d

    def orthogonal_proj(zfront: float, zback: float) -> np.array:  # pragma: no cover
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

    S_ion = 0
    A_ion = 1
    # plot first S then A ions
    if np.any(ion_type == S_ion):
        axis.scatter(doped_lattice[ion_type==S_ion, 0], doped_lattice[ion_type==S_ion, 1],
                     doped_lattice[ion_type==S_ion, 2], c='r', marker='o', label='S')
    if np.any(ion_type == A_ion):
        axis.scatter(doped_lattice[ion_type==A_ion, 0], doped_lattice[ion_type==A_ion, 1],
                     doped_lattice[ion_type==A_ion, 2], c='B', marker='o', label='A')

    axis.set_xlabel('X (Å)')
    axis.set_ylabel('Y (Å)')
    axis.set_zlabel('Z (Å)')
    plt.axis('square')

    plt.legend(loc='best', scatterpoints=1)


#def plot_optimization_brute_force(param_values: np.array, error_values: np.array) -> None:
#    '''Plot all results from the brute force optimization'''
#    plt.plot(param_values, error_values, '.b-')
#    plt.xlabel('Param value')
#    plt.ylabel('RMS error')
