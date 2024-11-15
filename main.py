# This file is part of Auto LB Workflow.
#
# Auto LB Workflow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Auto LB Workflow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Auto LB Workflow. If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import sys
import yaml
import os


# from scalene import scalene_profiler
# mpl.use("Agg")


def check_interpo(num_components, list_components_class, list_components_scalability_df):
    for i in range(num_components):

        c1 = list_components_class[i]

        # Plot the SYPD
        fig1, ax1 = plt.subplots()
        c1.sypd.plot(x='nproc', y='SYPD', ax=ax1)

        # Plot the fitness
        fig2, ax2 = plt.subplots()
        c1.fitness.plot(x='nproc', y='fitness', ax=ax2)

        methods = ['linear', 'slinear', 'quadratic']  # , 'cubic']
        legend = methods.copy()
        legend.insert(0, 'Real')
        df1 = list_components_scalability_df[i]
        for m in methods:
            ### Start using interpolated data
            comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1[m]})
            t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD, c1.nproc_restriction, c1.ts_info, c1.ts_nproc,
                           TTS_r, ETS_r)

            t1.sypd.plot(x='nproc', y='SYPD', ax=ax1)
            t1.fitness.plot(x='nproc', y='fitness', ax=ax2)

        ax1.set_title(c1.name + " SYPD after interpolating")
        ax1.set_xlabel('nproc')
        ax1.set_ylabel('SYPD')
        ax1.legend(legend)

        ax2.set_title(c1.name + " Fitness after interpolating")
        ax2.set_xlabel('nproc')
        ax2.set_ylabel('fitness')
        ax2.legend(legend)
        fig_name = c1.name + "_SYPD_Fitness_after_interpolating.png"
        plt.savefig("./img/" + fig_name)
        plt.show()


def interpolate_data(component, nproc_step):
    nproc_start = component.nproc.min()
    nproc_end = component.nproc.max() + 1
    ## Interpolation
    methods = ['linear', 'slinear', 'quadratic']  # , 'cubic']
    legend = methods.copy()
    legend.insert(0, 'real')

    ## Interpolation
    xold = component.nproc.values
    if len(component.nproc_restriction) != 0:
        xnew = component.nproc_restriction.values
    else:
        xnew = np.arange(nproc_start, nproc_end, nproc_step)
        component.nproc_restriction = xnew
    if len(component.ts_info) != 0 and component.ts_nproc not in xnew:
        xnew = np.append(xnew, component.ts_nproc)

    x = np.sort(np.unique(np.concatenate((xold, xnew))))
    tmp_component = pd.Series([component.sypd.SYPD[component.nproc[component.nproc == n].index[0]]
                               if n in component.nproc.values else np.NaN for n in x])
    df = pd.DataFrame({'nproc': x, 'real': tmp_component})
    for m in methods:
        f = interpolate.interp1d(component.nproc, component.sypd.SYPD, kind=m, fill_value="extrapolate")
        ynew = f(x).round(2)
        df[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(component.nproc, component.sypd.SYPD, 'o')
        for m in methods:
            plt.plot(x, df[m])
        plt.legend(legend)
        plt.title("Check interpo " + component.name)
        fig_name = component.name + "_check_interpo.png"
        plt.savefig(fig_name)
        plt.show()

    # Filter only with the nproc restriction and ts_nproc (do not include nproc configs from scalability curve)
    # TODO: I think this is redundant: interpolate.interp1d does only need the scalability curve points. Then it its used
    # for whatever xnew values I want.
    df_with_data_interpolated = df[df.nproc.isin(xnew)]
    return component, df_with_data_interpolated


def print_result(num_components, list_components_class_interpolated, optimal_result):
    print("Optimal for TTS=%.1f, ETS=%.1f:" % (TTS_r, ETS_r))
    nproc_acc = 0
    chsy_acc = 0
    for component in list_components_class_interpolated:
        print("\n -------------------------------\n")
        print("Results for component %s:" % component.name)
        print("Number of processes: %i" % component.top_nproc)
        print("Fitness: %.2f" % component.get_fitness([component.top_nproc]).fitness)
        print("CHSY: %i" % (component.get_chsy(component.top_nproc)))
        print("SYPD: %.2f" % component.get_sypd(component.top_nproc))
        component.plot_scalability(component.top_nproc)
        component.plot_scalability_n(component.top_nproc)
        nproc_acc += component.top_nproc
        chsy_acc += component.get_chsy(component.top_nproc)

    if num_components > 1:
        # We have to add the cpl_cost to the CHSY
        chsy_acc += optimal_result['cpl_chsy']
        print("\n -------------------------------\n")
        print("Total number of processes: %i" % nproc_acc)
        print("Expected coupled CHSY: %i" % chsy_acc)
        print("Expected coupled SYPD: %.2f" % optimal_result['SYPD'])
        print("Expected coupling cost: %.2f %%, (%.2f CHSY)" % (
        optimal_result['cpl_cost'] * 100, optimal_result['cpl_chsy']))
        print("Coupled Objective Function: %.3f" % optimal_result['objective_f'])

    # Plot all fitness values with the optimal values obtained in the same plot
    fig, ax1 = plt.subplots()
    legend = list()
    for component in list_components_class_interpolated:
        component.fitness.plot(x='nproc', y='fitness', label=component.name, ax=ax1)
        # Get currrent color
        color = plt.gca().lines[-1].get_color()
        ax1.plot(component.top_nproc, component.get_fitness([component.top_nproc]).fitness, 'o',
                 label='Opt. %s: %i proc' % (component.name, component.top_nproc), c=color)
        # Text to be shown
        text = ' %.2f' % component.get_fitness([component.top_nproc]).fitness
        ax1.text(component.top_nproc, component.get_fitness([component.top_nproc]).fitness, text)
    plt.title("Fitness and optimal values")
    plt.legend()
    plt.ylabel("Fitness")
    fig_name = "Fitness_values.png"
    plt.savefig("./img/" + fig_name)
    plt.show()

    # Plot all scalability curves with the optimal values obtained in the same plot
    fig, ax1 = plt.subplots()
    first = True
    for component in list_components_class_interpolated:
        component.sypd.plot(x="nproc", y="SYPD", label=component.name, ax=ax1)
        # Get currrent color
        color = plt.gca().lines[-1].get_color()
        # Draw vertical line
        ax1.axvline(x=component.top_nproc, ls='-.', c='k', alpha=1.)
        # Draw optimal point (dot)
        ax1.plot(component.top_nproc, component.get_sypd(component.top_nproc), 'o', markersize=5, c=color,
                 label='Opt. %s: %i proc' % (component.name, component.top_nproc))
        # Text to be shown
        text = ' %.2f' % component.get_sypd(component.top_nproc)
        ax1.text(component.top_nproc, component.get_sypd(component.top_nproc), text)
    plt.title("Scalability and optimal values")
    plt.legend()
    plt.ylabel("SYPD")
    fig_name = "Scalability_results.png"
    plt.savefig("./img/" + fig_name)
    plt.show()

    # Save top configurations as txt file
    if num_components > 1:
        out_file = "nproc_config_0"
        f = open(out_file, "w")
        for component in list_components_class_interpolated:
            # f.write(component.name + '_nprocs_0=( ' + ''.join('%s ' % x[0] for x in optimal_result['top_configurations']) + ')\n')
            f.write(component.name + '_nprocs_0=( ' + ''.join('%s ' % x for x in component.top5_nproc) + ')\n')
            print("Top 5 configurations for %s: %s" % (component.name, component.top5_nproc))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("No configuration file provided!")
        exit(1)

    # Create the directory to save the plots
    os.makedirs("./img/", exist_ok=True)

    config_file = open(sys.argv[1])
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Load global parameters form the YAML config file
    num_components = len(config['Components'])
    TTS_r = config['General']['TTS_ratio']
    ETS_r = 1 - TTS_r
    nproc_step = config['General']['nproc_step']
    method = config['General']['interpo_method']  # ['linear', 'slinear', 'quadratic', 'cubic']
    max_nproc = config['General']['max_nproc']
    show_plots = config['General']['show_plots']

    from component_class import Component

    list_components_scalability_df = list()
    list_components_class = list()
    list_components_interpolated = list()
    list_components_class_interpolated = list()

    information_per_ts_provided = True
    # Load component data
    for component in config['Components']:
        if (component['timestep_info'] is not None) and (component['timestep_nproc'] is not None):
            ts_info_df = pd.read_csv(component['timestep_info'])
            ts_info_df.rename(columns={ts_info_df.columns[0]: "ts_id"}, inplace=True)
            ts_info_nproc = component['timestep_nproc']
        else:  # No information per ts provided. We pass an empty df and assume regular ts lengths
            ts_info_df = pd.DataFrame()
            ts_info_nproc = 0
            information_per_ts_provided = False

        # Load scalability curve
        component_df = pd.read_csv(component['file'])
        # Remove configurations that would surpass the nproc limitation (max_nproc)
        component_df = component_df[component_df.nproc <= max_nproc]
        list_components_scalability_df.append(component_df)  # Save this just for debugging
        # Make sure that the nproc restriction is inside the range of scalability provided for that component
        if component['nproc_restriction'] is not None:
            component['nproc_restriction'] = [np for np in component['nproc_restriction'] if
                                              component_df.nproc.min() <= np <= component_df.nproc.max()]
            print("Component %s has a nproc restriction:" % component['name'])
            print(component['nproc_restriction'])

        c = Component(component['name'], component_df.nproc, component_df.SYPD,
                      component['nproc_restriction'], ts_info_df, component['timestep_nproc'], TTS_r, ETS_r)
        # Interpolate the data
        component_class, df_component_interpolated = interpolate_data(c, nproc_step)
        list_components_class.append(component_class)
        list_components_interpolated.append(df_component_interpolated)

        # TODO: Select one of the methods
        comp_interpolated = pd.DataFrame({'nproc': df_component_interpolated.nproc,
                                          'SYPD': df_component_interpolated[method]})

        c1_n = Component(c.name, comp_interpolated.nproc, comp_interpolated.SYPD,
                         c.nproc_restriction, ts_info_df, ts_info_nproc, TTS_r, ETS_r)
        list_components_class_interpolated.append(c1_n)
        c1_n.plot_scalability()

    if show_plots:
        check_interpo(num_components, list_components_class_interpolated, list_components_interpolated)

    from brute_force import brute_force

    # scalene_profiler.start()
    optimal_result = brute_force(num_components, list_components_class_interpolated, max_nproc, show_plots)
    print_result(num_components, list_components_class_interpolated, optimal_result)
    # scalene_profiler.stop()
