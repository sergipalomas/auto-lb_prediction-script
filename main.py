import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import sys, yaml
from cplx_opt import find_optimal
from mpl_toolkits import mplot3d


def check_interpo(num_components, list_components_class, list_components_scalability_df):

    for i in range(num_components):

        c1 = list_components_class[i]

        # Plot the SYPD
        fig1, ax1 = plt.subplots()
        c1.sypd.plot(x='nproc', y='SYPD', ax=ax1)

        # Plot the fitness
        fig2, ax2 = plt.subplots()
        c1.fitness.plot(x='nproc', y='fitness', ax=ax2)

        methods = ['linear', 'slinear', 'quadratic', 'cubic']
        legend = methods.copy()
        legend.insert(0, 'Real')
        df1 = list_components_scalability_df[i]
        for m in methods:
            ### Start using interpolated data
            comp1_new = pd.concat({'nproc': df1.nproc, 'SYPD': df1[m]})
            t1 = Component('IFS_n', comp1_new.nproc, comp1_new.SYPD, c1.nproc_restriction, TTS_r, ETS_r)

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
    if component.nproc_restriction.shape[0] != 0:
        xnew = np.array(component.nproc_restriction)
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


def print_result(num_components, list_components_class_interpolated, optimal_result):

    print("Optimal for TTS=%.1f, ETS=%.1f:" % (TTS_r, ETS_r))
    chpsy_acc = 0
    for component in list_components_class_interpolated:
        opt_nproc = optimal_result['nproc_' + component.name]
        print("\n -------------------------------\n")
        print("Results for component %s:" % component.name)
        print("Number of processes: %.2f" % opt_nproc)
        print("Fitness: %.2f" % (component.get_fitness(opt_nproc)))
        print("CHPSY: %i" % (component.get_chpsy(opt_nproc)))
        print("SYPD: %.2f" % component.get_sypd(opt_nproc))
        component.plot_scalability(opt_nproc)
        component.plot_scalability_n(opt_nproc)
        chpsy_acc += component.get_chpsy(opt_nproc)

    print("\n -------------------------------\n")
    print("Expected coupled CHPSY: %i" % chpsy_acc)
    print("Expected coupled SYPD: %.2f" % optimal_result['SYPD'])
    print("Coupled Objective Function: %.3f" % optimal_result['objective_f'])

    fig, ax1 = plt.subplots()
    legend = list()
    for i in range(num_components):
        c1_n = list_components_class_interpolated[i]
        c1_n.fitness.plot(x='nproc', y='fitness', legend=True, ax=ax1)
        ax1.plot(optimal_result['nproc_' + c1_n.name], c1_n.get_fitness(optimal_result['nproc_' + c1_n.name]), 'o')
        legend.append(c1_n.name)
        legend.append("optimal " + c1_n.name)
    plt.title("Fitness values")
    plt.legend(legend, loc=(0, 1.05))
    plt.show()


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
        # Make sure that the nproc restriction is inside the range of scalability provided for that component
        if component['nproc_restriction'] != None:
            component['nproc_restriction'] = [np for np in component['nproc_restriction'] if
                                              np <= component_df.nproc.max() and np >= component_df.nproc.min()]
            print("Component %s has a nproc restriction:" % component['Name'])
            print("[", *component['nproc_restriction'],"]")

        component_class = Component(component['Name'], component_df.nproc, component_df.SYPD,
                                    component['nproc_restriction'], TTS_r, ETS_r)
        list_components_class.append(component_class)
        # Interpolate the data
        df_component_interpolated = interpolate_data(component_class)
        list_components_interpolated.append(df_component_interpolated)

    # TODO: Select one of the methods
        comp_interpolated = pd.DataFrame({'nproc': df_component_interpolated.nproc,
                                           'SYPD': df_component_interpolated[method]})

        c1_n = Component(component['Name'], comp_interpolated.nproc, comp_interpolated.SYPD,
                         component['nproc_restriction'], TTS_r, ETS_r)
        list_components_class_interpolated.append(c1_n)

    if show_plots:
        check_interpo(num_components, list_components_class_interpolated, list_components_interpolated)

    # Run LP model
    # find_optimal(c1_n, c2_n)

    # from iLP import solve_ilp
    # solve_ilp(c1_n, c2_n)


    from brute_force import brute_force, brute_force_old
    optimal_result = brute_force(num_components, list_components_class_interpolated, max_nproc, show_plots)

    print_result(num_components, list_components_class_interpolated, optimal_result)
