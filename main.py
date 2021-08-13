import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import os, glob, sys, yaml
from cplx_opt import find_optimal
from mpl_toolkits import mplot3d


def check_interpo():

    c1.plot_scalability_n()
    c2.plot_scalability_n()

    # Plot the SYPD
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(c1.nproc, c1.sypd)
    ax2.plot(c2.nproc, c2.sypd)

    # Plot the fitness
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax3.plot(c1.nproc, c1.compute_fitness())
    ax4.plot(c2.nproc, c2.compute_fitness())

    methods = ['linear', 'slinear', 'quadratic', 'cubic']
    legend = methods.copy()
    legend.insert(0, 'Real')
    for m in methods:
        ### Start using interpolated data
        comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1[m]})
        comp2_new = pd.concat({'nproc': df2.nproc, 'SYPD': df2[m]})

        t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD, TTS_r, ETS_r)
        t2 = Component('NEMO_n', comp2_new.nproc, comp2_new.SYPD, TTS_r, ETS_r)

        ax1.plot(t1.nproc, t1.sypd)
        ax2.plot(t2.nproc, t2.sypd)
        ax3.plot(t1.nproc, t1.compute_fitness())
        ax4.plot(t2.nproc, t2.compute_fitness())


    ax1.set_title("SYPD after interpo for IFS")
    ax1.set_xlabel('nproc')
    ax1.set_ylabel('SYPD')
    ax1.legend(legend)

    ax2.set_title("SYPD after interpo for NEMO")
    ax2.set_xlabel('nproc')
    ax2.set_ylabel('SYPD')
    ax2.legend(legend)

    ax3.set_title("Fitness after interpo for IFS")
    ax3.set_xlabel('nproc')
    ax3.set_ylabel('fitness')
    ax3.legend(legend)

    ax4.set_title("Fitness after interpo for NEMO")
    ax4.set_xlabel('nproc')
    ax4.set_ylabel('fitness')
    ax4.legend(legend)

    plt.show()



def interpolate_data(component):
    step = 1
    start = 48
    ## Interpolation
    methods = ['linear', 'slinear', 'quadratic', 'cubic']
    legend = methods.copy()
    legend.insert(0, 'real')

    ## Interpolation
    elpin_cores = component.nproc_restriction
    # TODO: Use elpin nproc
    # xnew = pd.Series(elpin_cores)
    xnew = np.arange(start, component.nproc.max() + 1, step)
    tmp_component = pd.Series([component.sypd.SYPD[component.nproc[component.nproc == n].index[0]]
                               if n in component.nproc.values else np.NaN for n in xnew])
    df = pd.DataFrame({'nproc': xnew, 'real': tmp_component})
    for m in methods:
        f = interpolate.interp1d(component.nproc, component.sypd.SYPD, kind=m, fill_value="extrapolate")
        ynew = f(xnew).round(2)
        df[m] = pd.DataFrame({m: ynew})

    if show_plots:
        plt.plot(component.nproc, component.sypd.SYPD, 'o')
        for m in methods:
            plt.plot(xnew, df[m])
        plt.legend(legend)
        plt.title("Check interpo " + component.name)
        plt.show()

    return df


def print_result(list_components_class_interpolated, optimal_result):
    c1_n = list_components_class_interpolated[0]
    c2_n = list_components_class_interpolated[1]
    print("Optimal for TTS=%.1f, ETS=%.1f: \n" % (TTS_r, ETS_r))
    print("Number of processes for %s: %.2f" % (c1_n.name, optimal_result['nproc_' + c1_n.name]))
    print("Number of processes for %s: %.2f" % (c2_n.name, optimal_result['nproc_' + c2_n.name]))

    print("Fitness %s: %.2f" % (c1_n.name, optimal_result['fitness_' + c1_n.name]))
    print("Fitness %s: %.2f" % (c2_n.name, optimal_result['fitness_' + c2_n.name]))
    print("Objective function: %f" % optimal_result['objective_f'])

    print("Expected coupled SYPD: %.2f" % optimal_result['SYPD'])
    print("Expected coupled CHPSY: %i" % (c1_n.get_chpsy(optimal_result['nproc_' + c1_n.name])
                                          + c2_n.get_chpsy(optimal_result['nproc_' + c2_n.name])))
    print("%s CHPSY: %i" % (c1_n.name, c1_n.get_chpsy(optimal_result['nproc_' + c1_n.name])))
    print("%s CHPSY: %i" % (c2_n.name, c2_n.get_chpsy(optimal_result['nproc_' + c2_n.name])))

    plt.plot(c1_n.nproc, c1_n.fitness.fitness)
    plt.plot(c2_n.nproc, c2_n.fitness.fitness)
    plt.plot(optimal_result['nproc_' + c1_n.name], c1_n.get_fitness(optimal_result['nproc_' + c1_n.name]), 'o')
    plt.plot(optimal_result['nproc_' + c2_n.name], c2_n.get_fitness(optimal_result['nproc_' + c2_n.name]), 'o')
    plt.title("Fitness value")
    plt.legend([c1_n.name, c2_n.name])
    plt.show()

    c1_n.plot_scalability(optimal_result['nproc_' + c1_n.name])
    c2_n.plot_scalability(optimal_result['nproc_' + c2_n.name])


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("No configuration file provided!")
        exit(1)

    config_file = open(sys.argv[1])
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Load parameters form the YAML config file
    num_components = len(config['Components'])
    TTS_r = config['General']['TTS_ratio']
    ETS_r = 1 - TTS_r
    nodesize = config['General']['node_size']
    method = config['General']['interpo_method']  # ['linear', 'slinear', 'quadratic', 'cubic']
    max_nproc = config['General']['max_nproc']
    show_plots = config['General']['show_plots']

    elPin_cores = pd.Series([48, 92, 144, 192, 229, 285, 331, 380, 411, 476, 521, 563, 605, 665, 694, 759, 806, 826, 905,
                             1008, 1012, 1061, 1129, 1164, 1240, 1275, 1427, 1476, 1632, 1650, 1741, 1870])

    from component_class import Component
    list_components_scalability_df = list()
    list_components_class = list()
    list_components_interpolated = list()
    list_components_class_interpolated = list()
    for component in config['Components']:
        component_df = pd.read_csv(component['File'])
        list_components_scalability_df.append(component_df)
        component_class = Component(component['Name'], component_df.nproc, component_df.SYPD,
                                    component['nproc_restriction'], TTS_r, ETS_r)
        list_components_class.append(component_class)
        # Interpolate the data
        df_component_interpolated = interpolate_data(component_class)
        list_components_interpolated.append(interpolate_data(component_class))

    # TODO: Select one of the methods
        comp_interpolated = pd.DataFrame({'nproc': df_component_interpolated.nproc,
                                           'SYPD': df_component_interpolated[method]})

        c1_n = Component(component['Name'], comp_interpolated.nproc, comp_interpolated.SYPD,
                         component['nproc_restriction'], TTS_r, ETS_r)
        list_components_class_interpolated.append(c1_n)

    if show_plots:
        list_components_interpolated[0].plot_fitness()
        list_components_interpolated[1].plot_fitness()
        plt.title("Fitness")
        plt.legend([list_components_interpolated[0].name, list_components_interpolated[1].name])
        plt.show()

        #check_interpo()

    # Run LP model
    # find_optimal(c1_n, c2_n)

    from brute_force import brute_force, brute_force_old
    optimal_result = brute_force(list_components_class_interpolated, max_nproc)
    #
    # from iLP import solve_ilp
    # solve_ilp(c1_n, c2_n)

    print_result(list_components_class_interpolated, optimal_result)
